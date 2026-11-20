"""One fixed-length librosa feature vector per recording."""

import librosa
import numpy as np

SAMPLE_RATE = 22050
HOP_LENGTH = 512
FFT_LENGTH = 2048
CHANNELS = [*(f"mfcc_{i:02d}" for i in range(1, 21)),
            *(f"chroma_{i:02d}" for i in range(12)),
            "spectral_centroid", "rms", "zero_crossing_rate"]
FEATURE_NAMES = [f"{name}_{stat}" for name in CHANNELS for stat in ("mean", "std")]
FEATURE_NAMES.append("tempo_bpm")
FEATURE_COUNT = len(FEATURE_NAMES)
CACHE_VERSION = f"v1-librosa-{librosa.__version__}"


def extract(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if y.size < FFT_LENGTH:
        raise ValueError(f"Recording is too short: {path}")
    magnitude = np.abs(librosa.stft(y, n_fft=FFT_LENGTH, hop_length=HOP_LENGTH))
    mel = librosa.feature.melspectrogram(S=magnitude**2, sr=sr, n_fft=FFT_LENGTH)
    frames = np.vstack([
        librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20),
        librosa.feature.chroma_stft(S=magnitude**2, sr=sr, n_fft=FFT_LENGTH),
        librosa.feature.spectral_centroid(S=magnitude, sr=sr, n_fft=FFT_LENGTH),
        librosa.feature.rms(y=y, frame_length=FFT_LENGTH, hop_length=HOP_LENGTH),
        librosa.feature.zero_crossing_rate(y, frame_length=FFT_LENGTH, hop_length=HOP_LENGTH),
    ])
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    tempo = librosa.feature.tempo(onset_envelope=onset, sr=sr, hop_length=HOP_LENGTH)[0]
    features = np.append(np.column_stack((frames.mean(axis=1), frames.std(axis=1))).ravel(),
                         tempo).astype(np.float32)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ValueError(f"Invalid feature vector: {path}")
    return features
