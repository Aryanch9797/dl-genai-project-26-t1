import torch
import torchaudio
import math

def process_audio(audio_path, sr=16000, duration=10.24):
    """
    Processes a single audio file into normalized mel-spectrogram chunks.
    """
    num_samples = int(duration * sr)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=128,
        f_min=0,
        f_max=8000,
        normalized=True,
    )
    
    audio, orig_sr = torchaudio.load(audio_path)
    
    if orig_sr != sr:
        audio = torchaudio.functional.resample(audio, orig_sr, sr)
        
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
        
    total_samples = audio.shape[-1]
    num_chunks = math.ceil(total_samples / num_samples)
    
    test_data = []
    
    #  Slice into chunks 
    for i in range(num_chunks):
        start = i * num_samples
        end = start + num_samples
        audio_chunk = audio[:, start:end]
        
        # Pad the final chunk with silence (zeros) if it's too short
        if audio_chunk.shape[-1] < num_samples:
            padding_size = num_samples - audio_chunk.shape[-1]
            audio_chunk = torch.nn.functional.pad(audio_chunk, (0, padding_size))

        #  Generate Spectrogram and Normalize
        audio_mel = mel_transform(audio_chunk)
        mel_db = torch.log(audio_mel + 1e-6)
        
        mean = mel_db.mean()
        std = mel_db.std()
        
        mel_db = (mel_db - mean) / (std + 1e-6)
        
        test_data.append(mel_db)

    # Returns shape: [chunks, channel, height, width]
    return torch.stack(test_data)