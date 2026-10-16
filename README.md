# Honeywell-hackathon
TEAM project for hackathon — Problem Statement 2: automobile engine signature extraction & pattern recognition.

## Person B — Signal processing (`feature_extraction/`)

Turns raw engine sensor data into window-level **feature rows + spectrograms** for the ML models (Person C) and the dashboard (Person D).

Sensors: microphone, knock sensor, **3-axis accelerometer** (`accel_x/y/z`) and crank pulse on the fast channel, plus ECU telemetry (now including **ignition timing**).

### Pipeline (run from repo root)
```bash
python -m feature_extraction.generate_dataset   # 6 seeds x 5 conditions -> data/raw/ CSVs + CAN .log
python -m feature_extraction.pipeline            # -> data/processed/{features.csv, spectrograms.npy, spectrogram_meta.json}
python -m feature_extraction.validate_b_outputs  # gate: must print PASS before handing off
python -m pytest tests/ -q                       # test suite
python -m feature_extraction.can_codec check     # CAN round-trip self-test
python -m feature_extraction.plot_sensors all    # accel spectrograms + ignition trace + signatures
```
Sampling: fast channels at **25.6 kHz** (mic, knock, `accel_x/y/z`, crank); ECU telemetry at 10 Hz, delivered **only through a decoded CAN/OBD-II log** (SAE J1979), never the raw telemetry CSV. Windows are **0.5 s with 0.25 s hop (50 % overlap)**; windows that straddle two labels are skipped; the pipeline raises if `fs != 25600`.

> The accelerometer is sampled at **25.6 kHz** so it rides the same fast-channel pipeline as the mic and knock sensor. This deviates from the ~4 kHz typical of a spec-sheet industrial accelerometer; it is an intentional prototype choice so the knock/valve ring physics stay in band.

### Outputs
| File | Shape / content |
|---|---|
| `data/processed/features.csv` | 1170 rows x 141 cols = **136 features + 5 meta** (`run_id, window_id, label, t_start, t_end`) |
| `data/processed/spectrograms.npy` | `(1170, 2, 65, 199)` float32, dB, per-window normalized. **Channel 0 = mic, 1 = knock sensor** (unchanged — accel axes are feature columns, not spectrogram channels). Row *i* ↔ features.csv row *i* |
| `data/processed/spectrogram_meta.json` | per-row meta + global settings (fs, nperseg, channels, feature column order) |

`data/` is git-ignored — never commit generated data.

### Feature schema (136)
- `mic_*` (19), `knock_*` (19), `accel_x_*` (19), `accel_y_*` (19), `accel_z_*` (19): time (std, rms, peak, crest_factor, kurtosis, zero_crossing_rate), FFT (dominant_frequency_hz, spectral_centroid_hz), band energy + ratio for `low_0_1k` / `mid_1k_5k` / `knock_5k_10k`, and 5-10 kHz Hilbert envelope (env_mean/std/peak/peak_to_mean/repetition_hz).
  - `accel_x` carries the 7 kHz knock ring (5-10 kHz band up for knock); `accel_y` carries the 4 kHz valve-train click (1-5 kHz band up for valve wear); `accel_z` is crank-rate amplitude-modulated under misfire. Per-run `amp_scale` keeps raw loudness non-discriminative.
- `crank_*` (3): crank_rpm, crank_rpm_std, crank_interval_cv.
- `tel_<signal>_{mean,std,min,max,slope}` (35) for engine_rpm, coolant_temp_c, throttle_pct, engine_load, stft_pct, ltft_pct, **ignition_timing** (deg BTDC; retards under knock, scatters under misfire).
- Crank-gated per-cycle knock (3): knock_cycle_intensity_mean, knock_cycle_intensity_max, knock_cycle_pct_knocking.

Figures for the new sensors: `python -m feature_extraction.plot_sensors {accel|ignition|signature|all}`.

### For Person C (models) — READ THIS
- Drop the 5 meta columns (`run_id, window_id, label, t_start, t_end`) before training.
- **Split train/test by `run_id`, never randomly.** Windows overlap 50 %, so a random split leaks: amplitude-only honest accuracy is 0.49 under GroupKFold vs 0.83 under a random split. Use `GroupKFold(groups=run_id)`.
- Honest baseline: RandomForest on all features, GroupKFold (physics generalizes to held-out runs). Loudness alone (rms/peak, including the accel axes) is **not** discriminative by design.
- `spectrograms.npy` aligns row-for-row with `features.csv` for a CNN.

### For Person D (dashboard) — live path
`feature_extraction.pipeline.features_for_window(mic, knock, crank, telemetry_rows, accel_x, accel_y, accel_z, fs=25600)` returns `(feature_row, spectrogram)` using the exact same code as training — use it for real-time inference so live features match what the model was trained on.

> Note: `simulator/engine_simulator.py` was upgraded to 25.6 kHz + richer fault physics on this branch — **coordinate with Person A** (simulator owner) when merging.
