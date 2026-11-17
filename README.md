🎙️ Speech Emotion Recognition Using Deep Learning

<p align="center">
<strong>Deep Learning | Audio Signal Processing | Emotion Classification</strong>
</p>

<p align="center">
An intelligent speech emotion recognition system that uses deep learning to identify human emotions from audio recordings.
</p>

<p align="center">
<img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/Librosa-Audio%20Processing-0A9396?style=for-the-badge" alt="Librosa">
<img src="https://img.shields.io/badge/Status-In%20Development-orange?style=for-the-badge" alt="Project Status">
</p>

📌 1. Project Overview

Human speech communicates more than just words. Vocal characteristics such as pitch, tone, intensity and rhythm can provide valuable information about emotional expression.

Speech Emotion Recognition (SER) is a machine learning task that aims to identify emotional states from speech signals.

This project develops a deep learning-based Speech Emotion Recognition system using audio signal processing and a hybrid Convolutional Neural Network–Bidirectional Long Short-Term Memory (CNN-BiLSTM) architecture.

The system extracts meaningful acoustic representations from speech recordings using Mel-Frequency Cepstral Coefficients (MFCCs) and Mel Spectrograms before learning spatial and temporal patterns for multi-class emotion classification.

🎯 Project Goals
Extract meaningful acoustic features from speech recordings.
Apply audio preprocessing and feature engineering.
Develop a CNN-BiLSTM deep learning architecture.
Classify speech into eight emotion categories.
Evaluate model performance using appropriate classification metrics.
Explore the application of deep learning in human emotion recognition.
🧠 2. Model Architecture

The proposed architecture combines CNN and BiLSTM networks to learn complementary characteristics of speech signals.

Component	Function
Audio Preprocessing	Prepare and standardise audio recordings
MFCC Extraction	Capture important spectral characteristics
Mel Spectrogram	Represent audio frequency information over time
CNN	Learn local acoustic patterns
BiLSTM	Capture forward and backward temporal dependencies
Classification Layer	Predict the corresponding emotion category
Architecture Workflow
text
          AUDIO INPUT
               |
               ▼
       AUDIO PREPROCESSING
          (Librosa)
               |
               ▼
       FEATURE EXTRACTION
        ┌──────────────┐
        │     MFCC     │
        │ Mel Spectrogram│
        └──────────────┘
               |
               ▼
          CNN LAYERS
      Acoustic Feature Learning
               |
               ▼
         BiLSTM LAYERS
       Temporal Modelling
               |
               ▼
      FULLY CONNECTED LAYER
               |
               ▼
       SOFTMAX CLASSIFIER
               |
               ▼
       PREDICTED EMOTION

😊 3. Emotion Categories

The model is designed to recognise eight emotional categories from speech recordings.

Emotion	Description
😐 Neutral	Emotionally neutral speech
😌 Calm	Calm and relaxed speech
😄 Happy	Positive emotional expression
😢 Sad	Sad emotional expression
😠 Angry	Angry or frustrated expression
😨 Fearful	Fear-related vocal expression
🤢 Disgust	Disgust-related expression
😲 Surprised	Surprise-related expression
📂 4. Dataset

This project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

RAVDESS is a publicly available emotional speech and song dataset containing recordings performed by professional actors.

The dataset provides labelled emotional speech samples suitable for developing and evaluating speech emotion recognition systems.

Dataset source: RAVDESS on Zenodo

🛠️ 5. Technology Stack
Technology	Application
Python	Core programming language
PyTorch	Deep learning model development
Librosa	Audio processing and feature extraction
NumPy	Numerical operations
Scikit-learn	Data processing and evaluation metrics
Matplotlib	Data visualisation
Seaborn	Statistical visualisation
⚙️ 6. Key Features
Audio Preprocessing: Load, process and prepare WAV audio recordings using Librosa.
Feature Extraction: Extract MFCC and Mel Spectrogram representations.
CNN Feature Learning: Learn acoustic patterns from extracted audio features.
BiLSTM Sequence Modelling: Capture temporal dependencies in speech signals.
Multi-Class Classification: Predict one of eight emotion categories.
Audio Prediction: Generate emotion predictions from WAV recordings.
Model Evaluation: Assess classification performance and visualise results.
📁 7. Project Structure
text
speech-emotion-recognition/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── speech_emotion_analysis.ipynb
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│
├── results/
│
├── requirements.txt
├── .gitignore
└── README.md

🚀 8. Installation and Setup
Step 1: Clone the repository
bash
git clone YOUR_GITHUB_REPOSITORY_URL

Step 2: Navigate to the project directory
bash
cd speech-emotion-recognition

Step 3: Create a virtual environment
bash
python -m venv venv

Step 4: Activate the environment

Windows:

bash
venv\Scripts\activate


Mac/Linux:

bash
source venv/bin/activate

Step 5: Install dependencies
bash
pip install -r requirements.txt

🏋️ 9. Model Training

Train the CNN-BiLSTM model using:

bash
python -m src.train

📊 10. Model Evaluation

Evaluate the trained model using:

bash
python -m src.evaluate


Evaluation will focus on classification performance across the eight emotion categories.

Planned evaluation visualisations include:

Confusion Matrix
Classification Report
Accuracy and Loss Curves
Per-Class Precision and Recall
🎧 11. Emotion Prediction

Generate predictions from an audio recording:

bash
python -m src.predict


The system processes the input WAV recording and produces a predicted emotion category.

🏆 12. Results and Performance

Model evaluation results and performance comparisons will be documented here after training.

Evaluation Metric	Result
Accuracy	To be determined
Precision	To be determined
Recall	To be determined
F1-Score	To be determined

Additional visualisations, including confusion matrices and training curves, will be added as experiments progress.

🔮 13. Future Improvements

Potential extensions to this project include:

Real-time speech emotion recognition.
Interactive Streamlit web application.
Support for additional speech emotion datasets.
Audio data augmentation.
Transformer-based speech models.
Hyperparameter optimisation.
Comparison of different deep learning architectures.
Model explainability and performance analysis.
⚠️ 14. Disclaimer

This system estimates vocal emotion patterns based on learned audio characteristics. It does not directly measure a person's internal emotional state.

Predictions may be influenced by speaker characteristics, recording conditions, language, cultural differences and other factors.

This project is intended for educational and research purposes and should not be used as a definitive assessment of an individual's emotional state.

📜 15. License

This project is licensed under the MIT License.

See the LICENSE file for details.

<p align="center">
<strong>Turning speech signals into meaningful emotional insights through deep learning.</strong>
</p>

<p align="center">
⭐ If you find this project interesting, consider giving the repository a star!
</p>
