import librosa
import numpy as np

SAMPLE_RATE = 22050
DURATION = 3.5
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

def extract_features(file_path):
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=DURATION
    )

    target_length = int(SAMPLE_RATE * DURATION)

    audio = librosa.util.fix_length(
        audio,
        size=target_length
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Normalize each recording
    log_mel = (
        log_mel - log_mel.mean()
    ) / (log_mel.std() + 1e-6)

    return log_mel.astype(np.float32)

import librosa
import numpy as np

SAMPLE_RATE = 22050
DURATION = 3.5
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

def extract_features(file_path):
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        duration=DURATION
    )

    target_length = int(SAMPLE_RATE * DURATION)

    audio = librosa.util.fix_length(
        audio,
        size=target_length
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Normalize each recording
    log_mel = (
        log_mel - log_mel.mean()
    ) / (log_mel.std() + 1e-6)

    return log_mel.astype(np.float32)
