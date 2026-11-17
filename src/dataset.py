
from pathlib import Path
import torch
from torch.utils.data import Dataset
from src.preprocessing import extract_features

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

LABELS = list(EMOTIONS.values())
LABEL_TO_ID = {
    label: i for i, label in enumerate(LABELS)
}

class RAVDESSDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]

        parts = file_path.stem.split("-")

        emotion = EMOTIONS[parts[2]]
        label = LABEL_TO_ID[emotion]

        features = extract_features(str(file_path))

        features = torch.tensor(features).unsqueeze(0)

        return features, torch.tensor(label)


def load_files(data_dir):
    files = []

    for file_path in Path(data_dir).rglob("*.wav"):
        parts = file_path.stem.split("-")

        # Audio-only modality and speech channel
        if len(parts) != 7:
            continue

        if parts[0] == "03" and parts[1] == "01":
            files.append(file_path)

    return sorted(files)
