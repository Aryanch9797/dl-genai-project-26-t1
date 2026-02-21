# creating a dataset class to use pre-created mel-spectrogram of audio mashup
class MelSpectrogramDataset(Dataset):
    def __init__(self, is_validation, train_paths,val_path):
        """
        is_validation: True for validation data
        train_paths: list of paths for training data
        val_path: Path to the root folder containing genre subfolders for validations
        """
        self.is_validation = is_validation
        self.train_paths = train_paths
        self.val_path = val_path
        self.genre_to_label = {
            "blues": 0, "classical": 1, "country": 2, "disco": 3, "hiphop": 4,
            "jazz": 5, "metal": 6, "pop": 7, "reggae": 8, "rock": 9
        }
        
        self.samples = []   
        if self.is_validation:
            for genre, label in self.genre_to_label.items():
                genre_path = os.path.join(val_path, genre, "*.npz")
                all_files = sorted(glob(genre_path))
                
                # Store pairs of (path, label)
                for f in all_files:
                    self.samples.append((f, label))
        else:
            for path in train_paths:
                for genre, label in self.genre_to_label.items():
                    genre_path = os.path.join(path, genre, "*.npz")
                    all_files = sorted(glob(genre_path))
                    
                    # Store pairs of (path, label)
                    for f in all_files:
                        self.samples.append((f, label))
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, label = self.samples[index]
        
        with np.load(file_path) as data:
            mel = data['mel_spectrogram']

        mel_tensor = torch.from_numpy(mel).float()
            
        if not self.is_validation:
            if random.random() > 0.5:
                mel_tensor = F.mask_along_axis(
                    mel_tensor, 
                    mask_param=10, 
                    mask_value=0, 
                    axis=0
                )
    
                mel_tensor = F.mask_along_axis(
                    mel_tensor, 
                    mask_param=10, 
                    mask_value=0, 
                    axis=1
                )
        # Normalization
        mean = mel_tensor.mean()
        std = mel_tensor.std()
        mel_tensor = (mel_tensor - mean)/ (std + 1e-6)
        mel_tensor = mel_tensor.unsqueeze(0)  #format channel, height, width

        return mel_tensor, torch.tensor(label, dtype=torch.long)