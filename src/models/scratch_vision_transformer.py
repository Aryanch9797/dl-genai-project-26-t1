import torch
import torch.nn as nn


class TransformerLayer(nn.Module):

    def __init__(self, embed_dim=256):
        super().__init__()
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
        B, N, C = x.shape  # B: Batch, N: 657, C: 256
        num_heads = 8
        head_dim = C // num_heads  # 256 // 8 = 32
        
        # 1. Generate Q, K, V
        # Shape (B, 657, 256)
        q = self.weights_q(x)
        k = self.weights_k(x)
        v = self.weights_v(x) 
        
        q = q.view(B, N, num_heads, head_dim)
        k = k.view(B, N, num_heads, head_dim)
        v = v.view(B, N, num_heads, head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attention = q @ k.transpose(-2, -1) / (head_dim ** 0.5)   # @ is matrix multiplication of the last two dimensions
        attention = torch.softmax(attention, dim=-1)
        attention_output = attention @ v 
        
        attention_output = attention_output.transpose(1, 2) # Batch, 657, 8 heads, 32 dim

        # .contiguous() ensures memory is aligned before reshaping
        attention_output = attention_output.contiguous().view(B, N, C)
        
        return self.out_proj(attention_output)
    
    def forward(self, x):
        x = self.layer_norm1(x)
        attention_output = self.multi_head_attention(x)
        x = x + attention_output  
        x = self.layer_norm2(x)
        feed_forward_output = self.feed_forward_network(x)
        x = x + feed_forward_output  

        return x 


class scratch_vision_transformer(nn.Module):
    
    def __init__(self , num_layers):
        super().__init__()
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
        self.classifier_layers = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 10)
        )
        
    def generate_embeddings(self, x):
        batch = x.shape[0]
        x = self.embedding_generator(x)
        x = x.flatten(2)
        x = x.transpose(1,2)  # (batch_size, 656, embedding_dim)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens,x), dim=1)  # (batch_size, 657, embedding_dim)
        x = x + self.positional_embedding
        return x

    def forward(self , x):
        x = self.generate_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        cls_token_output = x[:, 0, :]  # Extract the output corresponding to the [CLS] token
        output = self.classifier_layers(cls_token_output)
        return output