import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm


def training_model(model, optimizer, train, val, epochs, patience, loss_fn, scheduler, device):
    """
    Custom training loop for pytorch model. with early stopping and model checkpointing and logging to wandb.
    """

    patience_counter = 0   # patience counter
    best_val = 0
    best_val_preds = []
    for e in tqdm(range(epochs)):
        if patience_counter > patience:
            print(f"Early stopping triggered")
            break
        
        # training loop
        model.train()
        
        train_losses = []
        all_train_preds = []
        all_train_targets = []
        
        for i, (x, y ) in enumerate(train):           
            x = x.to(device)
            y =  y.to(device)   
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds,y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            _, predicted_classes = torch.max(preds, 1) 
            all_train_preds.extend(predicted_classes.cpu().numpy())
            all_train_targets.extend(y.cpu().numpy())

        train_acc = accuracy_score(all_train_targets, all_train_preds)
        train_f1 = f1_score(all_train_targets, all_train_preds, average='macro')
        avg_train_loss = np.mean(train_losses)
        print(f"Epoch: {e}, Training_f1_macro: {train_f1}, Training_accuracy: {train_acc}, Avg_train_loss: {avg_train_loss}")

        # validation loop
        model.eval()
        all_val_preds = []
        all_val_targets = []
        with torch.inference_mode(): 
            for x, y  in val:
                x = x.to(device)
                y =  y.to(device)                
                preds = model(x)                
                _, predicted_classes = torch.max(preds, 1)
                all_val_preds.extend(predicted_classes.cpu().numpy())
                all_val_targets.extend(y.cpu().numpy())
        val_acc = accuracy_score(all_val_targets, all_val_preds)
        val_f1 = f1_score(all_val_targets, all_val_preds, average='macro')
        scheduler.step(val_f1)        
        print(f"Epoch: {e}, validation_f1_macro: {val_f1}, validation_accuracy: {val_acc}")

        # early stopping and model checkpointing
        if val_f1 > best_val:
            best_val = val_f1
            patience_counter = 0 
            torch.save(model.state_dict(), "best_model.pth")
            best_val_preds = all_val_preds
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")
            
        # Log metrics to wandb
        wandb.log({
                    "train/accuracy": train_acc,
                    "train/f1_macro": train_f1,
                    "train/loss": avg_train_loss,
                    "val/accuracy": val_acc,
                    "val/f1_macro": val_f1,
                    "epoch": e
                })
  
    wandb.finish()
    model.load_state_dict(torch.load('best_model.pth'))
    print(classification_report(all_val_targets,best_val_preds))
    sns.heatmap(confusion_matrix(all_val_targets,best_val_preds) , annot=True,fmt='d', cmap='Blues' )
    plt.show()
    
    return model