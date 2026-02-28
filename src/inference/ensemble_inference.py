import torch
import pandas as pd
from tqdm import tqdm

def ensemble_and_submit(models, loader, label_map, device, AST_model=None, AST_loader=None, ):
    """
    Ensembles three models via logit averaging and creates submission file.
    """
    for model in models:
        model.eval()
    if AST_model is not None:
        AST_model.eval()
    
    final_preds = []
    ids = []
    
    test_ids = loader.dataset.test_csv.index.tolist()
    print(f"Starting Ensemble on {device}...")

    with torch.inference_mode():
        
        if AST_model is not None and AST_loader is not None:
            combined_loader = zip(loader, AST_loader)
        else:
            combined_loader = loader

        for i, batch_data in enumerate(tqdm(combined_loader)):
            if AST_model is not None and AST_loader is not None:
                x,ast_x = batch_data  # Unzip the combined loader output
                # x = x[0]
                # ast_x = ast_x[0]
            else:
                x = batch_data
                # x = x[0] 
            
            x = x.squeeze(0) 
            x = x.to(device)
            
            logits_cnn = models[0](x)
            logits_resnet = models[1](x)
            avg_logits_cnn = torch.mean(logits_cnn, dim=0)        # calculating mean for all chunks
            avg_logits_resnet = torch.mean(logits_resnet, dim=0)
            final_logits = (avg_logits_cnn + avg_logits_resnet ) / 2  # average all three models  
            
            if AST_model is not None and AST_loader is not None:
                ast_x = ast_x.squeeze(0).to(device)
                logits_ast = AST_model(ast_x)
                avg_logits_ast = torch.mean(logits_ast, dim=0)  # Average over chunks for AST
                final_logits = (avg_logits_cnn + avg_logits_resnet + avg_logits_ast) / 3  # Average all three models
            
            pred_idx = torch.argmax(final_logits).item()
            predicted_label = label_map[pred_idx]
            
            final_preds.append(predicted_label)
            ids.append(test_ids[i])


    submission = pd.DataFrame({
        'id': ids,
        'genre': final_preds
    })
    

    submission.to_csv("submission.csv", index=False)
    print("Ensemble complete. Saved to 'submission.csv'")
    return submission