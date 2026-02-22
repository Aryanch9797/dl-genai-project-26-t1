import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.num_heads = 8
        self.head_dim = embed_dim // self.num_heads
        self.weights_q = nn.Linear(embed_dim, embed_dim)
        self.weights_k = nn.Linear(embed_dim, embed_dim)
        self.weights_v = nn.Linear(embed_dim, embed_dim)
        
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.feed_forward_network = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim)
        )

    def multi_head_attention(self, x):
        B, N, C = x.shape  # B: Batch, N: seq_len, C: embed_dim
        # Generate Q, K, V and reshape to (B, num_heads, N, head_dim)
        q = self.weights_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.weights_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.weights_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Flash Attention when available; also applies scaled dot-product and attention dropout
        
        attention_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)
        # Merge heads back: (B, N, C)
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, N, C)

        return self.out_proj(attention_output)
    
    def forward(self, x):
        x_norm = self.layer_norm1(x)
        attention_output = self.multi_head_attention(x_norm)
        x = x + attention_output  

        x_norm2 = self.layer_norm2(x)
        feed_forward_output = self.feed_forward_network(x_norm2)
        x = x + feed_forward_output  

        return x 


class scratch_vision_transformer(pl.LightningModule):
    
    def __init__(self , num_layers, lr, weight_decay, patience, scheduler_patience, scheduler_factor, max_epochs):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
        self.max_epochs = max_epochs
        self.num_layers = num_layers
        self.positional_embedding = nn.Parameter(torch.randn(1, 657, 256))
        self.embedding_generator = nn.Conv2d(in_channels=1,
                                        out_channels=256, # embedding dimension
                                        kernel_size=(8,25),
                                        stride=(8,25),
                                        bias=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 256))
        self.layers = nn.ModuleList([
                    TransformerLayer(embed_dim=256) for _ in range(num_layers)
                ])  
        nn.init.normal_(self.positional_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier_layers = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 10)
        )
        self.train_acc = Accuracy(task='multiclass', num_classes=10)
        self.train_f1  = F1Score(task='multiclass', num_classes=10, average='macro')
        self.val_acc   = Accuracy(task='multiclass', num_classes=10)
        self.val_f1    = F1Score(task='multiclass', num_classes=10, average='macro')
        
    def generate_embeddings(self, x):
        batch = x.shape[0]
        x = self.embedding_generator(x)
        x = x.flatten(2)
        x = x.transpose(1,2)  # (batch_size, 656, embedding_dim)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens,x), dim=1)  # (batch_size, 657, embedding_dim)
        x = x + self.positional_embedding
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        final_preds = torch.argmax(preds, dim=1)
        self.train_acc.update(final_preds, y)
        self.train_f1.update(final_preds, y)
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        warmup_epochs = 5
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.01,      # Starts at 1% of the base lr 
            total_iters=warmup_epochs
        )
        max_epochs = self.max_epochs
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=(max_epochs - warmup_epochs), # The remaining epochs
            eta_min=1e-6                        # The final learning rate floor
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_f1"}

    def forward(self , x):
        x = self.generate_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        cls_token_output = x[:, 0, :]  # Extract the output corresponding to the [CLS] token
        output = self.classifier_layers(cls_token_output)
        return output