
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from src.dataset import RAVDESSDataset, LABELS
from src.model import CNNBiLSTM

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    with open("models/test_files.txt") as f:
        files = [line.strip() for line in f if line.strip()]

    loader = DataLoader(
        RAVDESSDataset(files),
        batch_size=32
    )

    model = CNNBiLSTM().to(device)
    model.load_state_dict(
        torch.load("models/best_model.pth",
                   map_location=device,
                   weights_only=True)
    )
    model.eval()

    actual, predicted = [], []

    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features.to(device))
            predictions = outputs.argmax(dim=1).cpu()

            actual.extend(labels.tolist())
            predicted.extend(predictions.tolist())

    print("Accuracy:", accuracy_score(actual, predicted))
    print(classification_report(
        actual,
        predicted,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0
    ))

    cm = confusion_matrix(
        actual, predicted,
        labels=list(range(len(LABELS)))
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=LABELS,
        yticklabels=LABELS,
        cmap="Blues"
    )

    plt.xlabel("Predicted Emotion")
    plt.ylabel("Actual Emotion")
    plt.title("Speech Emotion Recognition")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    main()
