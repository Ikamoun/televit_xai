"""
Pytorch Lightning Module for training prototype segmentation model on Cityscapes and SUN datasets
"""
import os
from collections import defaultdict
from typing import Dict, Optional
from omegaconf import DictConfig
import hydra
import time

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import numpy as np

from .components.lr_scheduler import PolynomialLR
from .components.utils import get_params
from .components.helpers import list_of_distances
from .model import PPNet
from .model import construct_PPNet
import segmentation_models_pytorch as smp


from .train_and_test import warm_only, joint, last_only
from torchmetrics import AUROC, AveragePrecision, F1Score, Precision, Recall
import logging
from src import utils


logging.basicConfig(level=logging.WARNING)
log = utils.get_pylogger(__name__)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def reset_metrics() -> Dict:
    return {
        'n_correct': 0,
        'n_batches': 0,
        'n_patches': 0,
        'cross_entropy': 0,
        'kld_loss': 0,
        'loss': 0,
        'f1': 0,
        'auprc': 0,
        'precision': 0,
        'recall': 0
    }

# done defined a function for bn freeze
def freeze_bn_layers(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):  # Change to nn.BatchNorm1d for 1D data
            for param in module.parameters():
                param.requires_grad = False  # Freeze parameters

count = [155134269, 5462083]
def weighted_cross_entropy(inputs, targets, class_counts = count ): # [325134269, 5462083]
    class_counts = torch.tensor(class_counts, dtype=torch.float32).to(inputs.device)
    class_weights = 1 /  class_counts
    weights = class_weights / class_weights.sum()

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    loss = criterion(inputs, targets)
    return loss


# noinspection PyAbstractClass
# PatchClassificationModule adapted to work with Hydra config
#@gin.configurable(denylist=['model_dir', 'ppnet', 'training_phase', 'max_steps'])

#done changed the parameters to get the input a dic from hydra
class PatchClassificationModule(LightningModule):
    def __init__(self, cfg: DictConfig, model_dir, ppnet, training_phase=1,max_steps = 10000):
        super().__init__()

        self.model_dir = model_dir
        self.prototypes_dir = os.path.join(self.model_dir, 'prototypes')
        self.checkpoints_dir = os.path.join(self.model_dir, 'checkpoints')
        self.image_size = cfg.model.image_size

        self.ppnet = ppnet
        self.training_phase = training_phase
        self.max_steps = max_steps
        self.poly_lr_power = cfg.poly_lr_power
        self.loss_weight_crs_ent = cfg.loss_weight_crs_ent
        self.loss_weight_l1 = cfg.loss_weight_l1
        self.loss_weight_kld = cfg.loss_weight_kld
        self.joint_optimizer_lr_features = cfg.joint_optimizer_lr_features
        self.joint_optimizer_lr_add_on_layers = cfg.joint_optimizer_lr_add_on_layers
        self.joint_optimizer_lr_prototype_vectors = cfg.joint_optimizer_lr_prototype_vectors
        self.joint_optimizer_weight_decay = cfg.joint_optimizer_weight_decay
        self.warm_optimizer_lr_add_on_layers = cfg.warm_optimizer_lr_add_on_layers
        self.warm_optimizer_lr_prototype_vectors = cfg.warm_optimizer_lr_prototype_vectors
        self.warm_optimizer_weight_decay = cfg.warm_optimizer_weight_decay
        self.last_layer_optimizer_lr = cfg.last_layer_optimizer_lr
        self.ignore_void_class = cfg.ignore_void_class
        self.iter_size = cfg.iter_size

        self._device = torch.device("cuda:1")
        os.makedirs(self.prototypes_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.start_step = None

        # Initialize training, validation, etc.
        self.metrics = {}
        for split_key in ['train', 'val', 'test', 'train_last_layer']:
            self.metrics[split_key] = reset_metrics()

        # initialize configure_optimizers()
        self.optimizer_defaults = None
        self.start_step = None

        # we use optimizers manually
        self.automatic_optimization = False
        self.best_loss = 10

        # Initialize optimizers based on training phase
        #done changed the log to have the same as the unet one
        if self.training_phase == 0:
            warm_only(model=self.ppnet)
            log.info(f'WARM-UP TRAINING START. ({self.max_steps} steps)')
        elif self.training_phase == 1:
            joint(model=self.ppnet)
            log.info(f'JOINT TRAINING START. ({self.max_steps} steps)')
        else:
            last_only(model=self.ppnet)
            log.info('LAST LAYER TRAINING START.')

        self.ppnet.prototype_class_identity = self.ppnet.prototype_class_identity.cuda()
        self.lr_scheduler = None
        self.iter_steps = 0
        self.batch_metrics = defaultdict(list)



    def forward(self, x):
        return self.ppnet(x)

    def _step(self, split_key: str, batch):
        optimizer = self.optimizers()
        if split_key == 'train' and self.iter_steps == 0:
            optimizer.zero_grad()

        if self.start_step is None:
            self.start_step = self.trainer.global_step

        #freeze_bn_layers(self.ppnet.features)

        prototype_class_identity = self.ppnet.prototype_class_identity.cuda()
        metrics = self.metrics[split_key]
        image, mcs_target = batch

        image = image.to(self._device).to(torch.float32)
        mcs_target = mcs_target.cpu().detach().numpy().astype(np.float32)

        mcs_model_outputs = self.ppnet.forward(image, return_activations=False)
        if not isinstance(mcs_model_outputs, list):
            mcs_model_outputs = [mcs_model_outputs]

        #done added f1 aupcr
        mcs_loss, mcs_cross_entropy, mcs_kld_loss, mcs_cls_act_loss, mcs_auprc, mcs_f1, mcs_precision, mcs_recall = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for output, patch_activations in mcs_model_outputs:
            #for sample_target in mcs_target:
            #    target.append(resize_label(sample_target, size=(output.shape[2], output.shape[1])).to(self.device))
            #target = torch.stack(torch.from_numpy(mcs_target), dim=0)
            target =  mcs_target #done no multiscale here
            # we flatten target/output - classification is done per patch

            #to do : should we keep this or need ?
            output = output.reshape(-1, output.shape[-1])
            target_img = torch.from_numpy(target.reshape(target.shape[0], -1)).to(output.device) # (batch_size, img_size)
            target = target.flatten()

            patch_activations = patch_activations.permute(0, 2, 3, 1)
            patch_activations_img = patch_activations.reshape(patch_activations.shape[0], -1, patch_activations.shape[-1]) # (batch_size, img_size, num_proto)

            target = torch.from_numpy(target)
            target = target.to(output.device)

            #  cross entropy
            cross_entropy = torch.nn.functional.cross_entropy(
                output,
                target.long())
            #  Dice loss
            dice_loss =smp.losses.DiceLoss(mode='multiclass')(
                output,
                target.long())

            #probability of fire 
            preds = torch.nn.functional.softmax(output)[:,1]
            y_pred = (preds > 0.5).int()
            # # # Count the number of pixels classified as class 1
            # num_class_1_pixels = class_1_pixels.sum()
            # print(num_class_1_pixels)

            # preds = torch.nn.functional.softmax(output, dim=1)[:, 1]
            # # Now, apply a threshold to classify pixels as class 1 if probability > 0.5z
            print("pred")
            class_1_pixels = (preds > 0.5)
            num_class_1_pixels = class_1_pixels.sum()
            print(num_class_1_pixels)

            # Calculate precision and recall
            precision_mod = Precision(num_classes=1, average='micro', multiclass=False).to(output.device)
            recall_mod = Recall(num_classes=1, average='micro', multiclass=False).to(output.device)

            precision = precision_mod(y_pred.flatten(), target.long().flatten())
            recall = recall_mod(y_pred.flatten(),  target.long().flatten())

            # AUPRC and f1
            preds = torch.nn.functional.softmax(output)[:,1]
            auprc = AveragePrecision(pos_label=1, compute_on_cpu=True)
            f1 = F1Score().to(output.device)
            
            auprc_value = auprc(preds.flatten(), target.long().flatten())
            print(preds.flatten().max())
            f1_value = f1(preds.flatten(), target.long().flatten())

            #calculate KLD over class pixels between prototypes from same class

            kld_loss = []
            for img_i in range(len(target_img)):
                for cls_i in torch.unique(target_img[img_i]).cpu().detach().numpy():
                    cls_i = int(cls_i)
                    if cls_i < 0 or cls_i >= self.ppnet.prototype_class_identity.shape[1]:
                        continue
                    cls_protos = torch.nonzero(self.ppnet.prototype_class_identity[:, cls_i]). \
                        flatten().cpu().detach().numpy()

                    if len(cls_protos) == 0:
                        continue

                    cls_mask = (target_img[img_i] == cls_i)
                    log_cls_activations = [torch.masked_select(patch_activations_img[img_i, :, i], cls_mask)
                                           for i in cls_protos]
                    log_cls_activations = [torch.nn.functional.log_softmax(act, dim=0) for act in log_cls_activations]
                    for i in range(len(cls_protos)):
                        if len(cls_protos) < 2 or len(log_cls_activations[0]) < 2:
                            # no distribution over given class
                            continue

                        log_p1_scores = log_cls_activations[i]
                        for j in range(i + 1, len(cls_protos)):
                            log_p2_scores = log_cls_activations[j]

                            # add kld1 and kld2 to make 'symmetrical kld'
                            kld1 = torch.nn.functional.kl_div(log_p1_scores, log_p2_scores,
                                                              log_target=True, reduction='sum')
                            kld2 = torch.nn.functional.kl_div(log_p2_scores, log_p1_scores,
                                                              log_target=True, reduction='sum')


                            kld = (kld1 + kld2) / 2.0
                            kld_loss.append(kld)


            if len(kld_loss) > 0:
                kld_loss = torch.stack(kld_loss)
                # to make 'loss' (lower == better) take exponent of the negative (maximum value is 1.0, for KLD == 0.0)
                kld_loss = torch.exp(-kld_loss)
                kld_loss = torch.mean(kld_loss)
            else:
                kld_loss = 0.0

            output_class = torch.argmax(output, dim=-1)
            # to do should change this as we don't have a class

            is_correct = output_class == target

            if hasattr(self.ppnet, 'nearest_proto_only') and self.ppnet.nearest_proto_only:
                l1_mask = 1 - torch.eye(self.ppnet.num_classes, device=self._device)
            else:
                l1_mask = 1 - torch.t(prototype_class_identity)

            l1 = (self.ppnet.last_layer.weight * l1_mask).norm(p=1)

            loss = (self.loss_weight_crs_ent * cross_entropy)
            #         self.loss_weight_kld * kld_loss +
            #         self.loss_weight_l1 * l1)

            mcs_loss += loss / len(mcs_model_outputs)
            mcs_cross_entropy += cross_entropy / len(mcs_model_outputs)
            mcs_auprc += auprc_value/ len(mcs_model_outputs)
            mcs_f1 += f1_value/ len(mcs_model_outputs)
            mcs_precision += precision/ len(mcs_model_outputs)
            mcs_recall += recall/ len(mcs_model_outputs)
            mcs_kld_loss += kld_loss / len(mcs_model_outputs)
            metrics['n_correct'] += torch.sum(is_correct)
            metrics['n_patches'] += output.shape[0]

        self.batch_metrics['loss'].append(mcs_loss.item())
        self.batch_metrics['cross_entropy'].append(mcs_cross_entropy.item())
        self.batch_metrics['kld_loss'].append(mcs_kld_loss.item())
        self.batch_metrics['auprc'].append(mcs_auprc.item())
        self.batch_metrics['f1'].append(mcs_f1.item())
        self.batch_metrics['precision'].append(mcs_precision.item())
        self.batch_metrics['recall'].append(mcs_recall.item())
        self.iter_steps += 1

        if split_key == 'train':

            self.manual_backward(mcs_loss/ self.iter_size)

            if self.iter_steps == self.iter_size:
                self.iter_steps = 0
                optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            lr = get_lr(optimizer)
            self.log('lr', lr, on_step=True)

        elif self.iter_steps == self.iter_size:
            self.iter_steps = 0

        if self.iter_steps == 0:
            for key, values in self.batch_metrics.items():
                mean_value = float(np.mean(self.batch_metrics[key]))
                metrics[key] += mean_value
                if key == 'loss':
                    self.log('train_loss_step', mean_value, on_step=True, prog_bar=True)
            metrics['n_batches'] += 1

            self.batch_metrics = defaultdict(list)

        return loss, preds, target.long(), image


    def training_step(self, batch, batch_idx):
        self._step('train', batch)
        pass

    def validation_step(self, batch, batch_idx):
        loss, preds, targets, inputs = self._step('val', batch)

        return {"loss": loss, "preds": preds.detach().cpu(), "targets": targets.detach().cpu(),
                "inputs": inputs.detach().cpu()}
        

    def test_step(self, batch, batch_idx):
        return self._step('test', batch)

    def on_train_epoch_start(self):
        # reset metrics
        for split_key in self.metrics.keys():
            self.metrics[split_key] = reset_metrics()


    def on_validation_epoch_end(self):
        val_loss =  self.metrics['val']['cross_entropy']

        self.log('training_stage', float(self.training_phase))
        if self.training_phase == 0:
            stage_key = 'warmup'
        elif self.training_phase == 1:
            stage_key = 'nopush'
        else:
            stage_key = 'push'

        torch.save(obj=self.ppnet, f=os.path.join(self.checkpoints_dir, f'{stage_key}_last.pth'))


        # done change best model on loss as it is in the unet model
        #if val_acc > self.best_acc:
        if val_loss < self.best_loss:
            log.info(f'Saving best model, loss: ' + str(val_loss))
            self.best_acc = val_loss
            torch.save(obj=self.ppnet, f=os.path.join(self.checkpoints_dir, f'{stage_key}_best.pth'))

    def _epoch_end(self, split_key: str):
        metrics = self.metrics[split_key]
        if len(self.batch_metrics) > 0:
            for key, values in self.batch_metrics.items():
                mean_value = float(np.mean(self.batch_metrics[key]))
                metrics[key] += mean_value
            metrics['n_batches'] += 1

        n_batches = metrics['n_batches']
        print(n_batches)

        self.batch_metrics = defaultdict(list)

        for key in ['loss', 'cross_entropy', 'kld_loss','auprc','f1', 'precision', 'recall']:
            self.log(f'{split_key}/{key}', metrics[key] / n_batches)

        self.log(f'{split_key}/accuracy', metrics['n_correct'] / metrics['n_patches'])
        self.log('l1', self.ppnet.last_layer.weight.norm(p=1).item())
        if hasattr(self.ppnet, 'nearest_proto_only') and self.ppnet.nearest_proto_only:
            self.log('gumbel_tau', self.ppnet.gumbel_tau)

    def training_epoch_end(self, step_outputs):
        return self._epoch_end('train')

    def validation_epoch_end(self, step_outputs):
        p = self.ppnet.prototype_vectors.view(self.ppnet.prototype_vectors.shape[0], -1).cpu()
        with torch.no_grad():
            p_avg_pair_dist = torch.mean(list_of_distances(p, p))
        self.log('p dist pair', p_avg_pair_dist.item())

        return self._epoch_end('val')

    def test_epoch_end(self, step_outputs):
        print("test_epoch_end")
        print()    
        return self._epoch_end('test')

    # def configure_optimizers(self):
        # if self.training_phase == 0:  # warmup
            # aspp_params = [
            #     self.ppnet.features.base.aspp.c0.weight,
            #     self.ppnet.features.base.aspp.c0.bias,
            #     self.ppnet.features.base.aspp.c1.weight,
            #     self.ppnet.features.base.aspp.c1.bias,
            #     self.ppnet.features.base.aspp.c2.weight,
            #     self.ppnet.features.base.aspp.c2.bias,
            #     self.ppnet.features.base.aspp.c3.weight,
            #     self.ppnet.features.base.aspp.c3.bias
            # ]
        #     optimizer_specs = \
        #         [
        #             {
        #                 'params': list(self.ppnet.add_on_layers.parameters()),
        #                 # + aspp_params,
        #                 'lr': self.warm_optimizer_lr_add_on_layers,
        #                 'weight_decay': self.warm_optimizer_weight_decay
        #             },
        #             {
        #                 'params': self.ppnet.prototype_vectors,
        #                 'lr': self.warm_optimizer_lr_prototype_vectors
        #             }
        #         ]
        # elif self.training_phase == 1:  # joint
        #     optimizer_specs = \
        #         [
        #             {
        #                 "params": get_params(self.ppnet.features, key="1x"),  # augmenter learning rate selon model
        #                 'lr': self.joint_optimizer_lr_features,
        #                 'weight_decay': self.joint_optimizer_weight_decay
        #             },
        #             {
        #                 "params": get_params(self.ppnet.features, key="10x"),
        #                 'lr': 10 * self.joint_optimizer_lr_features,
        #                 'weight_decay': self.joint_optimizer_weight_decay
        #             },
        #             {
        #                 "params": get_params(self.ppnet.features, key="20x"),
        #                 'lr': 10 * self.joint_optimizer_lr_features,
        #                 'weight_decay': self.joint_optimizer_weight_decay
        #             },
        #             {
        #                 'params': self.ppnet.add_on_layers.parameters(), # sigmoid
        #                 'lr': self.joint_optimizer_lr_add_on_layers,
        #                 'weight_decay': self.joint_optimizer_weight_decay
        #             },
        #             {
        #                 'params': self.ppnet.prototype_vectors,
        #                 'lr': self.joint_optimizer_lr_prototype_vectors
        #             }
        #         ]
        # else:  # last layer
        #     optimizer_specs = [
        #         {
        #             'params': self.ppnet.last_layer.parameters(),
        #             'lr': self.last_layer_optimizer_lr
        #         }
        #     ]

        # optimizer = torch.optim.Adam(optimizer_specs)  # data specific

        # if self.training_phase == 1:
        #     self.lr_scheduler = PolynomialLR(
        #         optimizer=optimizer,
        #         step_size=1,
        #         iter_max=self.max_steps // self.iter_size,
        #         power=self.poly_lr_power
        #     )

        # return optimizer

# optimze as in seasfire 

    def configure_optimizers(self):
        if self.training_phase == 1:  # joint
            optimizer_specs  = [
                {
                    'params': self.ppnet.features.parameters(),
                    'lr': self.joint_optimizer_lr_features,
                    'weight_decay': self.joint_optimizer_weight_decay
                },
                
                {
                        'params': self.ppnet.add_on_layers.parameters(), # sigmoid
                        'lr': self.joint_optimizer_lr_add_on_layers,
                        'weight_decay': self.joint_optimizer_weight_decay
                    },

                {
                    'params': self.ppnet.prototype_vectors,
                    'lr': self.joint_optimizer_lr_prototype_vectors
                }
            ]
        else:  # last layer
            optimizer_specs = [
                {
                    'params': self.ppnet.last_layer.parameters(),
                    'lr': self.last_layer_optimizer_lr
                }
            ]

        optimizer = torch.optim.Adam(optimizer_specs)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return optimizer

