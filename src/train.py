
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

from src.dataset import RAVDESSDataset, load_files
from src.model import CNNBiLSTM

DATA_DIR = "data/RAVDESS"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

def main():
    files = load_files(DATA_DIR)

    if not files:
        raise RuntimeError("No RAVDESS speech files found.")

    actors = sorted({
        f.stem.split("-")[6] for f in files
    })

    train_actors, test_actors = train_test_split(
        actors, test_size=0.2, random_state=42
    )

    train_actors, val_actors = train_test_split(
        train_actors, test_size=0.2, random_state=42
    )

    train_files = [
        f for f in files if f.stem.split("-")[6] in train_actors
    ]
    val_files = [
        f for f in files if f.stem.split("-")[6] in val_actors
    ]
    test_files = [
        f for f in files if f.stem.split("-")[6] in test_actors
    ]

    print("Training samples:", len(train_files))
    print("Validation samples:", len(val_files))
    print("Testing samples:", len(test_files))

    train_loader = DataLoader(
        RAVDESSDataset(train_files),
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        RAVDESSDataset(val_files),
        batch_size=32
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CNNBiLSTM().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )

    best_loss = float("inf")

    for epoch in range(30):
        model.train()
        train_loss = 0.0

        for features, labels in tqdm(train_loader):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                val_loss += criterion(
                    outputs, labels
                ).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1}/30 | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss

            torch.save(
                model.state_dict(),
                MODEL_DIR / "best_model.pth"
            )

    # Save test file list for the evaluation stage
    with open(MODEL_DIR / "test_files.txt", "w") as f:
        for path in test_files:
            f.write(str(path) + "\n")

    print("Training complete.")

if __name__ == "__main__":
    main()
