
import torch
import torch.nn as nn
from tqdm import tqdm

def prediction(model,test_loader,device):
    model.eval()
    predictions = []
    
    with torch.inference_mode():
        for x in tqdm(test_loader):
            x = x.squeeze(0) # (batch, Num_Chunks, 1, Freq, Time) -->  (Num_Chunks, 1, Freq, Time)
            x = x.to(device)
            logits = model(x)
            avg_logits = torch.mean(logits, dim=0, keepdim=True)
            
            _, predicted_classes = torch.max(avg_logits, 1)
            predictions.append(predicted_classes.item())
            
    return predictions