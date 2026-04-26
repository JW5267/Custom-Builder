# AI Audio Detector

`ai_audio_detector.py` provides a lightweight AI-audio detection model that extracts **72 engineered features**:

- **30 spectral features** (distribution, band energy, and dynamic descriptors)
- **24 rhythm features** (envelope statistics, tempo-lag strength, frame dynamics)
- **18 timbre features** (harmonicity, spectral slope, crest, and MFCC-like coefficients)

It outputs a score in `[0,1]` representing AI-likelihood.

## Quick start

```bash
python ai_audio_detector.py
```

Use in your own code:

```python
from ai_audio_detector import AIAudioDetector

detector = AIAudioDetector()
score, feature_groups = detector.predict(audio_array, sample_rate)
print(score, feature_groups["all"].shape)  # 72
```

From WAV:

```python
score, feature_groups = detector.from_wav("example.wav")
```
