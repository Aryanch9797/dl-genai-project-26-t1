import torch
import torch.nn as nn

class CNN_mashup_model(nn.Module):
    def __init__(self, layers, n, sr ,dr,fl):
        """
            layers: Number of CNN layer
            n: size of kernel 
            sr: sample rate of music 
            dr: dropout rate
            fl: number of start filters
        """
        super().__init__()
        self.layers = layers
        self.n = n
        self.sr = sr
        self.dr = dr


        CNN_models = []
        self.prev_channel = 1 

        for i in range(self.layers):           
            out_channels = min(fl * (2**i), 512)             
            CNN_models.extend([
                nn.Conv2d(  in_channels=self.prev_channel, 
                            out_channels=out_channels, 
                            kernel_size=self.n, 
                            padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(self.dr)
            ])
            self.prev_channel = out_channels

        self.CNN_model = nn.Sequential(*CNN_models)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(in_features=self.prev_channel, out_features=10)

    def forward(self,x):
        
        x = self.CNN_model(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        
        return logits
    