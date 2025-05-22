from torchgeo.trainers import SemanticSegmentationTask
from torchgeo.models import FarSeg, FCSiamConc
import torch.nn.functional as F
import torch

from pytorch_lightning import LightningModule

class UnetBinaryModule(LightningModule):
    def __init__(self, in_channels=4, num_classes=2, lr=1e-4):
        super().__init__()
        #self.model = FCSiamConc(in_channels=in_channels, classes=num_classes)
        self.model = FarSeg(classes=num_classes)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), y.float(), reduction='mean')
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), y.float(), reduction='mean')
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
