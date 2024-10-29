import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

# ------------------------------------------------------------------------------------ #
# `pyrootutils.setup_root(...)` is recommended at the top of each start file
# to make the environment more robust and consistent
#
# the line above searches for ".git" or "pyproject.toml" in present and parent dirs
# to determine the project root dir
#
# adds root dir to the PYTHONPATH (if `pythonpath=True`)
# so this file can be run from any place without installing project as a package
#
# sets PROJECT_ROOT environment variable which is used in "configs/paths/default.yaml"
# this makes all paths relative to the project root
#
# additionally loads environment variables from ".env" file (if `dotenv=True`)
#
# you can get away without using `pyrootutils.setup_root(...)` if you:
# 1. move this file to the project root dir or install project as a package
# 2. modify paths in "configs/paths/default.yaml" to not use PROJECT_ROOT
# 3. always run this file from the project root dir
#
# https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from typing import List, Optional, Tuple

import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from models.module import PatchClassificationModule
from models.model import construct_PPNet
from models.push import push_prototypes
import os

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    #log.info(f"Instantiating model <{cfg.model._target_}>")
    #model: LightningModule = hydra.utils.instantiate(cfg.model)


    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)


    # if cfg.get("train"):
    #     log.info("Starting training!")
    #     trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    # train_metrics = trainer.callback_metrics
    ppnet = construct_PPNet(cfg.model)

    log.info(f"Instantiating model <{cfg.model._target_}> for joint step")
    # Instantiate the PatchClassificationModule using Hydra's config
    module: LightningModule = PatchClassificationModule(cfg, model_dir = cfg.results_dir, ppnet = ppnet, training_phase=1,max_steps=cfg.joint_steps)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": module,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # if cfg.trainer.get("auto_lr_find"):
    #     print("auto_lr_find")
    #     lr_finder = trainer.tuner.lr_find(model=model, datamodule=datamodule)
    #     # Plot with
    #     fig = lr_finder.plot(suggest=True)
    #     fig.show()
    #     # Pick point based on plot, or get suggestion
    #     new_lr = lr_finder.suggestion()
    #     print(f"New LR: {new_lr}")
    #     # update hparams of the model
    #     model.hparams.lr = new_lr

    global_step = trainer.global_step if trainer is not None else 0
    current_epoch = trainer.current_epoch if trainer is not None else 0 


    cfg.trainer.max_steps = cfg.joint_steps
    cfg.trainer.max_epochs = cfg.joint_epochs
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    #trainer = Trainer(logger=logger, enable_progress_bar=True,
    #                min_steps=1, max_steps=cfg.joint_steps, val_check_interval=1,gpus=0) #to do change this

    trainer.fit_loop.current_epoch = current_epoch + 1
    trainer.fit_loop.global_step = global_step + 1
    trainer.fit(model=module, datamodule=datamodule)

    ckpt_path = trainer.checkpoint_callback.best_model_path

    print("best model joint")
    print(ckpt_path)

    log.info('SAVING PROTOTYPES')
    ppnet = ppnet.cuda()
    module.eval()
    torch.set_grad_enabled(False)

    cfg.datamodule.push_prototypes = True
    push_dataset: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # Call setup to initialize the datasets
    push_dataset.setup(stage='fit')

    push_prototypes(
        push_dataset,
        prototype_network_parallel=ppnet,
        prototype_layer_stride=1,
        root_dir_for_saving_prototypes=module.prototypes_dir,
        prototype_img_filename_prefix='prototype-img',
        prototype_self_act_filename_prefix='prototype-self-act',
        proto_bound_boxes_filename_prefix='bb',
        save_prototype_class_identity=True,
        pascal=False,
        log=log,
        input_var=cfg.input_vars
    )

    torch.save(obj=ppnet, f=os.path.join(cfg.results_dir, f'checkpoints/push_last.pth'))
    torch.save(obj=ppnet, f=os.path.join(cfg.results_dir, f'checkpoints/push_best.pth'))

    ppnet = torch.load(os.path.join(cfg.results_dir, f'checkpoints/push_last.pth'))
    ppnet = ppnet.cuda()

    log.info('LAST LAYER FINE-TUNING')
    torch.set_grad_enabled(True)
    # callbacks = [
    #     EarlyStopping(monitor='val/accuracy', patience=early_stopping_patience_last_layer, mode='max')
    # ]

    cfg.datamodule.push_prototypes = False

    datamodule : LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    module = PatchClassificationModule( cfg,model_dir = cfg.results_dir, ppnet = ppnet, training_phase=2,max_steps=cfg.finetune_steps)

    #callbacks = [cb for cb in callbacks if not isinstance(cb, RichProgressBar)]

    current_epoch = trainer.current_epoch if trainer is not None else 0

    cfg.trainer.max_steps = cfg.finetune_steps + cfg.joint_steps  # Total steps with fine-tuning
    cfg.trainer.max_epochs = cfg.joint_epochs + cfg.finetune_epochs  # Total epochs including fine-tuning
    #trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger) 
    trainer.fit_loop.current_epoch = trainer.current_epoch
    trainer.fit_loop.global_step = trainer.global_step + 1

    trainer.fit(model=module, datamodule=datamodule)


    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=module, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics



    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="prototype.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
