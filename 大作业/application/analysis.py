import os
import numpy as np
import librosa
from sklearn.preprocessing import RobustScaler
import argparse
from tqdm import tqdm



class AudioFeatureAnalysis:
    def __init__(self, sr=44100, n_mfcc=13):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.scaler = RobustScaler()

    def __call__(self, audio_path):
        # return self.extract(audio_path)
        return self._extractAudio(audio_path)

    def _extractAudio(self, audio_path):
        y, sr = librosa.load(audio_path, sr=self.sr)
        audio = np.vstack([y, y])
        print(audio)
        exit(0)
        return audio

    def _extract_mfcc(self, y, sr):
        # MFCC特征
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        return np.vstack([mfcc, delta_mfcc, delta2_mfcc])

    def _extract_pitch(self, y, sr):
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=70, fmax=400, fill_na=1
        )
        return f0.reshape(1, -1)

    def _extract_volume(self, y, sr):
        rms = librosa.feature.rms(y=y)
        delta_rms = librosa.feature.delta(rms)
        delta2_rms = librosa.feature.delta(rms, order=2)
        return np.vstack([rms, delta_rms, delta2_rms])

    def _extract_timbre(self, y, sr):
        S = np.abs(librosa.stft(y))
        timbre_features = np.vstack([
            librosa.feature.spectral_centroid(S=S),
            librosa.feature.spectral_bandwidth(S=S),
            librosa.feature.spectral_rolloff(S=S),
            librosa.feature.chroma_stft(S=S, sr=sr),
            librosa.feature.spectral_contrast(S=S, sr=sr)
        ])
        return timbre_features

    def extract(self, audio_path) -> np.ndarray:
        y, sr = librosa.load(audio_path, sr=self.sr)

        pitch = self._extract_pitch(y, sr)
        mfcc = self._extract_mfcc(y, sr)
        volume = self._extract_volume(y, sr)
        timbre = self._extract_timbre(y, sr)

        flatness = librosa.feature.spectral_flatness(y=y)
        zerocr = librosa.feature.zero_crossing_rate(y)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)

        features = [pitch, mfcc, volume, timbre, flatness, zerocr, mel]
        features = np.vstack(features)
        features = self.scaler.fit_transform(features)

        return features