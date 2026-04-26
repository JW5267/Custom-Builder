"""AI Audio Detector model.

This module implements an interpretable audio detector that extracts 72 features
split across spectral patterns, rhythm, and timbre fingerprints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window, stft
from scipy.stats import kurtosis, skew

EPS = 1e-10


@dataclass
class FeaturePack:
    """Container for model features."""

    spectral: np.ndarray
    rhythm: np.ndarray
    timbre: np.ndarray

    @property
    def all(self) -> np.ndarray:
        return np.concatenate([self.spectral, self.rhythm, self.timbre])


@dataclass
class AIAudioDetector:
    """72-feature detector for AI-generated audio artifacts."""

    sample_rate: int = 16_000
    n_fft: int = 1024
    hop_length: int = 256
    frame_length: int = 1024
    weights: np.ndarray = field(default_factory=lambda: np.linspace(-0.7, 0.7, 72))
    bias: float = 0.05

    def from_wav(self, path: str) -> Tuple[float, Dict[str, np.ndarray]]:
        sr, data = wavfile.read(path)
        audio = self._to_mono_float(data)
        if sr != self.sample_rate:
            audio = self._resample(audio, sr, self.sample_rate)
            sr = self.sample_rate
        return self.predict(audio, sr)

    def predict(self, audio: np.ndarray, sample_rate: int) -> Tuple[float, Dict[str, np.ndarray]]:
        pack = self.extract_features(audio, sample_rate)
        score = self._score(pack.all)
        return score, {
            "spectral": pack.spectral,
            "rhythm": pack.rhythm,
            "timbre": pack.timbre,
            "all": pack.all,
        }

    def extract_features(self, audio: np.ndarray, sample_rate: int) -> FeaturePack:
        audio = self._prepare_audio(audio)
        spec_mag, freqs, _ = self._magnitude_spectrogram(audio, sample_rate)
        spectral = self._spectral_features(spec_mag, freqs)
        rhythm = self._rhythm_features(audio, spec_mag, sample_rate)
        timbre = self._timbre_features(audio, spec_mag, sample_rate)

        assert spectral.shape[0] == 30, "Spectral feature vector must be length 30"
        assert rhythm.shape[0] == 24, "Rhythm feature vector must be length 24"
        assert timbre.shape[0] == 18, "Timbre feature vector must be length 18"

        return FeaturePack(spectral=spectral, rhythm=rhythm, timbre=timbre)

    def _score(self, features: np.ndarray) -> float:
        x = self._zscore(features)
        raw = float(np.dot(x, self.weights) + self.bias)
        return 1.0 / (1.0 + np.exp(-raw))

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float64).squeeze()
        if audio.ndim != 1:
            raise ValueError("Audio must be mono after preprocessing")
        peak = np.max(np.abs(audio)) + EPS
        return np.clip(audio / peak, -1.0, 1.0)

    def _magnitude_spectrogram(self, audio: np.ndarray, sample_rate: int):
        _, _, zxx = stft(
            audio,
            fs=sample_rate,
            window=get_window("hann", self.n_fft),
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            nfft=self.n_fft,
            boundary=None,
            padded=False,
        )
        return np.abs(zxx) + EPS, np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate), None

    def _spectral_features(self, spec_mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        power = spec_mag**2
        frame_energy = np.sum(power, axis=0) + EPS
        normalized = power / np.sum(power, axis=0, keepdims=True)

        centroid = np.sum(freqs[:, None] * normalized, axis=0)
        spread = np.sqrt(np.sum(((freqs[:, None] - centroid[None, :]) ** 2) * normalized, axis=0))
        rolloff = np.array([
            freqs[np.searchsorted(np.cumsum(normalized[:, i]), 0.85)] for i in range(normalized.shape[1])
        ])
        flatness = np.exp(np.mean(np.log(spec_mag + EPS), axis=0)) / (np.mean(spec_mag, axis=0) + EPS)

        band_edges = np.linspace(0, len(freqs) - 1, 11, dtype=int)
        band_energies = []
        for i in range(len(band_edges) - 1):
            b0, b1 = band_edges[i], band_edges[i + 1]
            band_energies.append(np.mean(power[b0:b1], axis=0))
        band_energies = np.array(band_energies)

        feats = [
            np.mean(centroid), np.std(centroid), skew(centroid), kurtosis(centroid),
            np.mean(spread), np.std(spread), skew(spread), kurtosis(spread),
            np.mean(rolloff), np.std(rolloff),
            np.mean(flatness), np.std(flatness),
            np.mean(frame_energy), np.std(frame_energy),
            skew(frame_energy), kurtosis(frame_energy),
            np.mean(np.diff(centroid)), np.std(np.diff(centroid)),
            np.mean(np.diff(flatness)), np.std(np.diff(flatness)),
        ]

        # 10 band means + 10 band stds = 20; keep first 10 for compactness
        band_means = np.mean(band_energies, axis=1)
        band_stds = np.std(band_energies, axis=1)
        feats.extend(band_means[:5])
        feats.extend(band_stds[:5])

        return np.array(feats[:30], dtype=np.float64)

    def _rhythm_features(self, audio: np.ndarray, spec_mag: np.ndarray, sample_rate: int) -> np.ndarray:
        envelope = np.mean(spec_mag, axis=0)
        envelope = envelope - np.mean(envelope)

        ac = np.correlate(envelope, envelope, mode="full")
        ac = ac[len(ac) // 2 :]
        ac = ac / (np.max(np.abs(ac)) + EPS)

        tempo_bins = np.arange(30, 241, 10)
        lags = (60.0 * sample_rate / (tempo_bins * self.hop_length)).astype(int)
        lags = np.clip(lags, 1, len(ac) - 1)
        tempo_strength = ac[lags]

        zcr_frames = self._framewise_zcr(audio)
        rms_frames = self._framewise_rms(audio)

        feats = [
            np.mean(envelope), np.std(envelope), skew(envelope), kurtosis(envelope),
            np.mean(np.diff(envelope)), np.std(np.diff(envelope)),
            np.max(tempo_strength), np.mean(tempo_strength), np.std(tempo_strength),
            tempo_bins[np.argmax(tempo_strength)],
            np.mean(zcr_frames), np.std(zcr_frames), skew(zcr_frames), kurtosis(zcr_frames),
            np.mean(rms_frames), np.std(rms_frames), skew(rms_frames), kurtosis(rms_frames),
            np.mean(np.diff(rms_frames)), np.std(np.diff(rms_frames)),
            np.mean(np.diff(zcr_frames)), np.std(np.diff(zcr_frames)),
        ]

        # Add 2 robust rhythm texture descriptors to reach 24
        feats.extend([
            np.percentile(envelope, 90) - np.percentile(envelope, 10),
            np.mean(np.abs(np.diff(np.signbit(np.diff(envelope)).astype(int)))),
        ])

        return np.array(feats[:24], dtype=np.float64)

    def _timbre_features(self, audio: np.ndarray, spec_mag: np.ndarray, sample_rate: int) -> np.ndarray:
        mfcc_like = self._dct(np.log(spec_mag + EPS), k=8)
        harmonic_ratio = self._harmonicity(audio, sample_rate)

        spectral_slope = self._spectral_slope(spec_mag, sample_rate)
        crest = np.max(spec_mag, axis=0) / (np.mean(spec_mag, axis=0) + EPS)

        feats = [
            np.mean(harmonic_ratio), np.std(harmonic_ratio),
            np.mean(spectral_slope), np.std(spectral_slope),
            np.mean(crest), np.std(crest), skew(crest), kurtosis(crest),
        ]

        for i in range(mfcc_like.shape[0]):
            feats.append(np.mean(mfcc_like[i]))
        for i in range(mfcc_like.shape[0]):
            feats.append(np.std(mfcc_like[i]))

        return np.array(feats[:18], dtype=np.float64)

    def _framewise_zcr(self, audio: np.ndarray) -> np.ndarray:
        frames = self._frame_audio(audio)
        signs = np.sign(frames)
        crossings = np.sum(np.abs(np.diff(signs, axis=1)), axis=1) / 2
        return crossings / frames.shape[1]

    def _framewise_rms(self, audio: np.ndarray) -> np.ndarray:
        frames = self._frame_audio(audio)
        return np.sqrt(np.mean(frames**2, axis=1) + EPS)

    def _frame_audio(self, audio: np.ndarray) -> np.ndarray:
        n = len(audio)
        if n < self.frame_length:
            audio = np.pad(audio, (0, self.frame_length - n))
            n = len(audio)
        n_frames = 1 + (n - self.frame_length) // self.hop_length
        idx = np.arange(self.frame_length)[None, :] + self.hop_length * np.arange(n_frames)[:, None]
        return audio[idx]

    def _dct(self, x: np.ndarray, k: int) -> np.ndarray:
        n = x.shape[0]
        i = np.arange(n)
        basis = np.array([np.cos(np.pi * (i + 0.5) * j / n) for j in range(k)])
        return basis @ x

    def _harmonicity(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        frames = self._frame_audio(audio)
        mins = int(sample_rate / 500)
        maxs = int(sample_rate / 60)
        out = []
        for frame in frames:
            ac = np.correlate(frame, frame, mode="full")
            ac = ac[len(ac) // 2 :]
            denom = ac[0] + EPS
            ac = ac / denom
            if maxs >= len(ac):
                maxs = len(ac) - 1
            out.append(np.max(ac[mins:maxs]))
        return np.array(out)

    def _spectral_slope(self, spec_mag: np.ndarray, sample_rate: int) -> np.ndarray:
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / sample_rate)
        x = np.log(freqs[1:] + 1.0)
        y = np.log(spec_mag[1:, :] + EPS)
        x0 = x - np.mean(x)
        denom = np.sum(x0**2) + EPS
        slopes = (x0[:, None] * (y - np.mean(y, axis=0, keepdims=True))).sum(axis=0) / denom
        return slopes

    def _zscore(self, x: np.ndarray) -> np.ndarray:
        return (x - np.mean(x)) / (np.std(x) + EPS)

    def _to_mono_float(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data)
        if data.ndim == 2:
            data = np.mean(data, axis=1)
        if np.issubdtype(data.dtype, np.integer):
            max_val = np.iinfo(data.dtype).max
            data = data.astype(np.float64) / max_val
        return data.astype(np.float64)

    def _resample(self, audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        if src_sr == dst_sr:
            return audio
        ratio = dst_sr / src_sr
        n = int(round(len(audio) * ratio))
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
        return np.interp(x_new, x_old, audio)


if __name__ == "__main__":
    # Demonstration with synthetic signal.
    sr = 16_000
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    synthetic = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
    synthetic += 0.05 * np.random.default_rng(42).standard_normal(len(t))

    detector = AIAudioDetector()
    prob, details = detector.predict(synthetic, sr)
    print(f"AI-likelihood score: {prob:.4f}")
    print(f"Feature vector size: {details['all'].shape[0]}")
