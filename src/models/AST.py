import torch 
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score
from transformers import ASTForAudioClassification

class AST_model(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        
        self.model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", num_labels=10,ignore_mismatched_sizes=True)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task='multiclass', num_classes=10)
        self.train_f1  = F1Score(task='multiclass', num_classes=10, average='macro')
        self.val_acc   = Accuracy(task='multiclass', num_classes=10)
        self.val_f1    = F1Score(task='multiclass', num_classes=10, average='macro')

    def shared_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        final_preds = torch.argmax(preds, dim=1)
        self.train_acc.update(final_preds, y)
        self.train_f1.update(final_preds, y)

        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/accuracy', self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/f1_macro',  self.train_f1,  on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        final_preds = torch.argmax(preds, dim=1)
        self.val_acc.update(final_preds, y)
        self.val_f1.update(final_preds, y)

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/accuracy', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/f1_macro',  self.val_f1,  on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        backbone_params = self.model.audio_spectrogram_transformer.parameters()
        head_params = self.model.classifier.parameters()
        
        lr_head = 5e-5      
        lr_backbone = 1e-5  
        
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr_backbone}, 
            {'params': head_params, 'lr': lr_head}          
        ], weight_decay=1e-4)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/f1_macro", 
            },
        }

    def forward(self, x):
        x = x.squeeze(1)
        x = x.transpose(1,2)
        x = x[:, :1024, :]
        x = self.model(input_values=x).logits
        return x


