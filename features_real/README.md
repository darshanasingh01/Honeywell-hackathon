# features_real: feature extraction for the REAL engine data (person B)

A separate, self-contained folder. It does **not** read, import or modify the existing synthetic feature set,
and every file it writes starts with `real_`, so nothing can overwrite the synthetic outputs.

Input: 320 real engine recordings (183 diesel, 137 petrol, 44.1 kHz mono WAV) and `EngineFaultDB_Final.csv`
(55,999 rows, 4 fault classes).
Output: for every audio window a log-mel spectrogram plus a feature row, and a windowed feature table for the CSV.

## 1. Install (Windows, once)

```powershell
cd features_real
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run

```powershell
python run_real_features.py --audio-dir "C:\path\to\audio" --csv "C:\path\to\EngineFaultDB_Final.csv" --out real_outputs
```

* `--audio-dir` is searched recursively. A file's fuel label comes from the **folder name**: any folder
  whose name contains `diesel` or `petrol` (any case). WAVs in other folders are skipped and counted.
* Identical files are dropped by content hash (for example the same recording in two ZIP parts).
  The same audio under both fuels is an error.
* Options: `--no-spectrograms` (skip the big .npz), `--win-s`, `--hop-s`, `--highpass`. Other settings are in
  `real_features/config.py`.
* Defaults give **5,692 audio windows** (0.5 s window, 0.25 s hop), the same count as the earlier ChatGPT run.
  Full run takes about a minute.

Tests: `pip install pytest` then `python -m pytest -q tests` (14 tests).

## 3. Outputs (`real_outputs/`)

| File | What it is |
|---|---|
| `real_audio_windows.csv` | one row per audio window: metadata + 49 features (below) |
| `real_audio_spectrograms.npz` | `spectrograms` float16 `(5692, 64, 21)` log-mel dB, plus `window_id`; row *i* matches row *i* of the CSV |
| `real_audio_manifest.csv` | one row per unique recording: key, REC number, fuel, label (0 diesel, 1 petrol), split, path |
| `real_audio_summary.json` | settings, counts, duration stats |
| `real_csv_windows.csv` | one row per 128-row window of the CSV: label `fault`, split, row range, 5 stats per signal |
| `real_csv_summary.json` | settings, counts per split and class |

Audio metadata columns: `window_id, recording_key, rec_num, fuel, label, split, start_s, end_s, padded, clipped`
(`clipped` = the whole recording hits full scale; 15 recordings do, flagged but kept).

Audio feature columns (49):
* time domain: `rms, peak, crest, zcr, skew, kurtosis`
* spectrum: `spec_centroid, spec_bandwidth, spec_rolloff85, spec_flatness, spec_entropy, dominant_hz`
* relative band energy: `band_20_100 ... band_16000_22050` (fractions that sum to 1)
* envelope (1-8 kHz band, Hilbert): `env_mod_depth, env_kurtosis, env_peak_hz, env_peak_ratio`
* MFCC 1-13: `mfccN_mean, mfccN_std`

CSV feature columns: for each of the 14 signals `<name>_mean, _std, _min, _max, _slope`
(slope = least-squares change per row).

Load example:
```python
import numpy as np, pandas as pd
df = pd.read_csv("real_outputs/real_audio_windows.csv")
spec = np.load("real_outputs/real_audio_spectrograms.npz", allow_pickle=True)["spectrograms"]
train = df.split == "train"
X_img, y = spec[train.to_numpy()], df.loc[train, "label"]
```

## 4. Things the next person must know

1. **The audio labels are fuel type (diesel vs petrol), not faults.** The recordings carry no fault labels.
2. **The audio and the CSV are not linked.** The CSV has no recording id, timestamp or audio reference, so
   audio rows and CSV rows cannot be joined. Treat them as two separate datasets.
3. **Split by recording, never by window.** Windows overlap 50% and come from the same recording. The `split`
   column is assigned per recording (stratified by fuel, seed 42, 70/15/15), so no recording is in two splits.
4. **CSV splits are by contiguous chunks.** The CSV is one unbroken block per fault class and neighbouring rows are
   near-copies (autocorrelation ~0.99). Each class is cut into 2000-row chunks, whole chunks go to
   train/val/test, and 128 rows are trimmed from each chunk edge so no window sits next to another split.
   Keep the file in its original row order. A shuffled file is rejected with an error.
5. **What fault codes 0-3 mean is not known.** Do not name them in the dashboard until that is confirmed.
6. **CAN decoding is not included.** There are no CAN frames in the real data; the CSV engine columns
   (MAP, TPS, RPM, ...) are the stand-in. A decoder can be added if the simulator emits real CAN frames.
7. **Order analysis needs RPM, and the audio files have none.** `audio_features.order_spectrum(seg, sr, rpm)` is
   provided and tested (for a roughly constant-RPM window) but not used by the pipeline.
   `env_peak_hz` is the strongest envelope modulation frequency in 5-500 Hz; it is **not** an RPM estimate.
8. Loudness is kept (`rms`, `peak`), so microphone gain differences between recordings can leak into models.
   Per-recording normalisation is a reasonable option for the modelling stage.

## 5. Adding this to the shared repo without touching the synthetic set

1. Copy the `features_real` folder into your clone of the repo, on your `features` branch.
2. `git status` and check it lists only **new** files (nothing modified or deleted).
3. `git add features_real/` (never `git add .`), commit, push, open a pull request.
   The `real_outputs/` folder is about 19 MB; if the team prefers a lighter repo, leave `real_outputs/` out and
   rerun step 2 on a shared drive instead.
