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
        self.net = smp.UnetPlusPlus(encoder_name=encoder, in_channels=len(input_vars) , classes=nb_classes)  # to do voir si pretrained  smp unet plus  #to do pb initialisation, reinitialiser protoseg  avec des exemples de feu # to do refaire afficahge et vérifier color  # to do : tenter une autre seed # to do mettre precisio recall metrics 
        if loss == 'dice':
            self.criterion = smp.losses.DiceLoss(mode='multiclass') #to do use a dice loss to balance # to do see number feu et number non feu 
        elif loss == 'ce':
            self.criterion = torch.nn.CrossEntropyLoss() # to do weighted cross entropy
        
            #self.criterion = torch.nn.functional.cross_entropy
        #self.val_auc = AUROC(pos_label=1, num_classes=nb_classes, compute_on_cpu=True)
        self.val_f1 = F1Score(compute_on_cpu=True)
        self.val_auprc = AveragePrecision(pos_label=1, num_classes=nb_classes-1, compute_on_cpu=True)
        #self.test_auc = AUROC(pos_label=1, num_classes=nb_classes, compute_on_cpu=True)
        self.test_auprc = AveragePrecision(pos_label=1, num_classes=nb_classes-1, compute_on_cpu=True)
        self.test_f1 = F1Score()

    def forward(self, x: torch.Tensor):
        return self.net(x)


# forward and get features 

    # def forward(self, x, return_features=False, layer = 3):
    # """
    # Forward pass of the model.
    
    # Args:
    #     x (tensor): Input image tensor.
    #     return_features (bool): If True, return the encoder feature maps.
    #                             If False, return the final logits (segmentation output).
    
    # Returns:
    #     Tensor: Either the encoder feature maps or the logits (final segmentation output).
    # """

    
    # # If return_features is True, return encoder feature maps
    # if return_features:
    #     return self.net.encoder(x)[layer]
    
    # # Otherwise, return the final logits (default behavior)
    # return self.net(x)


    # encoder_features[0]	(1, 32, 48, 48) 32
    # Block 1	encoder_features[1]	(1, 16, 48, 48)	16
    # Block 2	encoder_features[2]	(1, 24, 24, 24)	24
    # Block 3	encoder_features[3]	(1, 40, 12, 12)	40
    # Block 4	encoder_features[4]	(1, 80, 6, 6)	80
    # Block 5	encoder_features[5]	(1, 112, 6, 6)	112
    # Block 6	encoder_features[6]	(1, 192, 3, 3)	192
    # Final Block	encoder_features[7]	(1, 320, 3, 3)	320

    def step(self, batch: Any):
        x, y = batch

        # x_local = batch['x_local']
        # x_local_mask = batch['x_local_mask']
        # x_oci = batch['x_oci']
        # x_global = batch['x_global']
        # y_local = batch['y_local']
        # y_global = batch['y_global']
        # x = x_local
        # y = y_local

        # if this is the first batch
        # if self.global_step == 0:
        #     # print the shapes of the inputs and outputs
        #     print(f'x_local shape: {x_local.shape}')
        #     print(f'x_oci shape: {x_oci.shape}')
        #     print(f'x_global shape: {x_global.shape}')
        #     print(f'y_local shape: {y_local.shape}')
        #     print(f'x_local_mask shape: {x_local_mask.shape}')
        #     print(f'y_global shape: {y_global.shape}')

        x = x.float()
        # pad x of shape (batch_size, C, 80, 80) to (batch_size, 1, 96, 96)
        #x = torch.nn.functional.pad(x, (8, 8, 8, 8), mode='constant', value=0)

        # pad y of shape (batch_size, 80, 80) to (batch_size, 96, 96)
        #y = torch.nn.functional.pad(y, (8, 8, 8, 8), mode='constant', value=0)

        # x_oci = x_oci.float()
        # x_global = x_global.float()
        y = y.long()
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        return loss, preds, y, x

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, inputs = self.step(batch)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        pass

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


def unet_features(input_vars,nb_classes,  pretrained=False):

    """Constructs a UNet model without the final layer for feature extraction."""
    
    model = plUNET(input_vars = input_vars, nb_classes= nb_classes)
    
    if pretrained:
        print("using pretrained")
        # Load pretrained weights (assuming they exist)
        # For example, use torch.load to load weights from a file:
        
        #with 64
        checkpoint = torch.load('/home/ines/televit_xai/logs/train/runs/2024-10-09_11-26-14/checkpoints/epoch_015.ckpt')
        
        # Remove the final layer weights if necessary
        checkpoint.pop('final_conv.weight', None)
        checkpoint.pop('final_conv.bias', None)
        
        # Load the remaining weights
        model.load_state_dict(checkpoint, strict=False)
    
    # Remove the final layer for feature extraction
    model.final_conv = nn.Identity()  # Replace final layer with identity to remove it
    print(model)
    return model
