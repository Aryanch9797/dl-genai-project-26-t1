class test_mashed_dataset(Dataset):
    def __init__(self, path, test_csv, sr, duration):
        self.path = path
        self.test_csv = test_csv
        self.sr = sr
        self.duration = duration
        self.num_samples = int(duration * sr)
        # Mel transform
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr, # number of samples per second 16000 in this case
            n_fft=400,        # 400 frequency bins the fourier tranform calculates
            win_length=400, # 25ms  # 400 samples are analyzed at one moment for creating spectrogram
            hop_length=160 , # 10ms  # 160 samples the window slides after each calculation
            n_mels=128,        # 128 frequency bands
            f_min=0,          # minimun frequency 
            f_max=8000,       # maximum frequency
            normalized=True,   # normalization
        )
        
        # self.db = torchaudio.transforms.AmplitudeToDB()
    def __len__(self):
        return len(self.test_csv)
    def __getitem__(self,index):
        filename = test_csv.iloc[index][0]
        path = self.path+filename
        
        """Load and preprocess audio."""
        audio, sr = torchaudio.load(path)
        
        if sr != self.sr:         # correct sample rate
            audio = torchaudio.functional.resample(audio, sr, self.sr)
        
        if audio.shape[0] > 1:    # ensure mono audio
            audio = audio.mean(0, keepdim=True)
        total_samples = audio.shape[-1]
        
        samples = math.ceil(total_samples/self.num_samples)
        test_data = []
                            
        for i in range(samples):  # breaking one test samples into 10.24 sec parts
            start = i*self.num_samples
            end = start+self.num_samples
            audio_chunk = audio[:,start:end]
            
            if audio_chunk.shape[-1] < self.num_samples:
                audio_chunk = torch.nn.functional.pad(audio_chunk, (0, self.num_samples- audio_chunk.shape[-1]))

            audio_mel = self.mel(audio_chunk)
            mel_db = torch.log(audio_mel + 1e-6)
            mean = mel_db.mean()
            std = mel_db.std()
            
            mel_db = (mel_db - mean) / (std + 1e-6)
                
            test_data.append(mel_db)

        return torch.stack(test_data)   # chunks, channel, height, width