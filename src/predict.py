
import torch
import torch.nn.functional as F

from src.model import CNNBiLSTM
from src.preprocessing import extract_features
from src.dataset import LABELS

def predict_emotion(audio_path):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CNNBiLSTM().to(device)

    model.load_state_dict(
        torch.load(
            "models/best_model.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    features = extract_features(audio_path)

    x = torch.tensor(features).unsqueeze(0).unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        output = model(x)
        probabilities = F.softmax(output, dim=1)

    confidence, prediction = probabilities.max(dim=1)

    return {
        "emotion": LABELS[prediction.item()],
        "confidence": round(confidence.item() * 100, 2)
    }


if __name__ == "__main__":
    result = predict_emotion("sample.wav")

    print("Predicted emotion:", result["emotion"])
    print("Confidence:", result["confidence"], "%")
