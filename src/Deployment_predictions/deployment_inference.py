import torch
import pandas as pd
from src.Deployment_predictions.deployment_data_perprocessor import process_audio
from src.models.scratch_CNN import CNN_mashup_model
from src.models.Resnet_50 import ResNet50_GenreClassifier


def load_models(cnn_path, renet_path):
    """
    Loads the CNN and ResNet models from the specified paths.
    """
    cnn_model = CNN_mashup_model(7,3,16000,0.2,32)
    resnet_model = ResNet50_GenreClassifier()

    cnn_model.load_state_dict(torch.load(cnn_path, map_location='cpu'))
    resnet_model.load_state_dict(torch.load(renet_path, map_location='cpu'))
    
    return cnn_model, resnet_model

def predict_ensemble(audio_path, cnn_model, resnet_model):
    """
    Predicts the genre of the given audio using an ensemble of CNN and ResNet models.
    """
    audio = process_audio(audio_path)
    audio = audio.squeeze(0) 

    label_to_genre = {
        0: "blues", 1: "classical", 2: "country", 3: "disco", 4: "hiphop",
        5: "jazz", 6: "metal", 7: "pop", 8: "reggae", 9: "rock"
    }

    cnn_model.eval()
    resnet_model.eval()
    
    with torch.inference_mode():
        logits_cnn = cnn_model(audio)
        logits_resnet = resnet_model(audio)
        
        avg_logits_cnn = torch.mean(logits_cnn, dim=0)
        avg_logits_resnet = torch.mean(logits_resnet, dim=0)
        
        final_logits = (avg_logits_cnn + avg_logits_resnet) / 2
        
        pred_idx = torch.argmax(final_logits).item()
        prediction_prob = torch.softmax(final_logits, dim=0)[pred_idx].item()
        prediction_genre = label_to_genre[pred_idx]
    
    return prediction_genre , prediction_prob
