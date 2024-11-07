from typing import Any, List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import AUROC, AveragePrecision, F1Score
import segmentation_models_pytorch as smp
import pytorch_lightning as pl


class plUNET(pl.LightningModule):
    def __init__(
            self,
            input_vars: list = None,
            nb_classes: int = 2,
            positional_vars: list = None,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
            loss='dice',
            encoder='efficientnet-b5'
    ):
        super().__init__()
        print(nb_classes)
        self.save_hyperparameters(logger=False)
        self.net = smp.UnetPlusPlus(encoder_name=encoder, in_channels=len(input_vars) , classes=nb_classes)
        print(self.net)
        if loss == 'dice':
            self.criterion = smp.losses.DiceLoss(mode='multiclass')
        elif loss == 'ce':
            self.criterion = torch.nn.CrossEntropyLoss()

        self.train_f1 = F1Score(compute_on_cpu=True)
        self.train_auprc = AveragePrecision(pos_label=1, num_classes=nb_classes-1, compute_on_cpu=True)
        #self.val_auc = AUROC(pos_label=1, num_classes=nb_classes, compute_on_cpu=True)
        self.val_f1 = F1Score(compute_on_cpu=True)
        self.val_auprc = AveragePrecision(pos_label=1, num_classes=nb_classes-1, compute_on_cpu=True)
        #self.test_auc = AUROC(pos_label=1, num_classes=nb_classes, compute_on_cpu=True)
        self.test_auprc = AveragePrecision(pos_label=1, num_classes=nb_classes-1, compute_on_cpu=True)
        self.test_f1 = F1Score()

    def forward(self, x: torch.Tensor):
        return self.net(x)


    def step(self, batch: Any):
        x, y = batch

        x = x.float()
        y = y.long()

        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        return loss, preds, y, x

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)
        self.train_auprc.update(preds.flatten(), targets.flatten())
        self.train_f1.update(preds.flatten(), targets.flatten())
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # Compute and log averaged metrics over the entire epoch
        self.log("train/auprc", self.train_auprc.compute(), prog_bar=False)
        self.log("train/f1", self.train_f1.compute(), prog_bar=False)
        
        # Reset metrics for the next epoch
        self.train_auprc.reset()
        self.train_f1.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)
        # log val metrics

        #self.val_auc.update(preds, targets)
        self.val_auprc.update(preds.flatten(), targets.flatten())
        self.val_f1.update(preds.flatten(), targets.flatten())
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        #self.log("val/auroc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds.detach().cpu(), "targets": targets.detach().cpu(),
                "inputs": inputs.detach().cpu()}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.step(batch)
        #self.test_auc.update(preds, targets)
        self.test_auprc.update(preds.flatten(), targets.flatten())
        self.test_f1.update(preds.flatten(), targets.flatten())

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        #self.log("test/auroc", self.test_auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds.detach().cpu(), "targets": targets.detach().cpu()}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, "monitor": "train/loss"}

#function for prototypes 
def unet_features(input_vars, nb_classes,  pretrained=False):

    """Constructs a UNet model without the final layer for feature extraction."""

    model = smp.UnetPlusPlus(encoder_name='efficientnet-b1', in_channels=len(input_vars) , classes=nb_classes)

    if pretrained:
        print("using pretrained")
        # Load pretrained weights
        checkpoint = torch.load('/home/ines/televit_xai/logs/train/runs/2024-10-09_11-26-14/checkpoints/epoch_015.ckpt')

        # Remove the final layer weights if necessary
        checkpoint.pop('final_conv.weight', None)
        checkpoint.pop('final_conv.bias', None)

        # Load the remaining weights
        model.load_state_dict(checkpoint, strict=False)

    # Remove the final layer for feature extraction
    model.final_conv = nn.Identity()  # Replace final layer with identity to remove it
    return model
