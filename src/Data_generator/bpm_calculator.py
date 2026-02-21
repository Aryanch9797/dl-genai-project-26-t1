"""
This script calculates the BPM (beats per minute) for each audio file inside the provided path and save the results in a CSV for later use.
"""
def calculate_bpm(path):
    audio_files = glob.glob(path)
    bpm_data = []
    for path in tqdm(audio_files):
        audio , _ = audio , sr = librosa.load(path, sr=22050)
        onset_env = librosa.onset.onset_strength(y=audio, sr=22050) 
        current_bpm = librosa.feature.tempo(onset_envelope=onset_env)[0]
        if current_bpm <= 0: 
            current_bpm = 120.0
        bpm_data.append({
                "path": path,      
                "bpm": current_bpm  
            })

    df = pd.DataFrame(bpm_data)
    df.to_csv("bpm.csv",index=False)