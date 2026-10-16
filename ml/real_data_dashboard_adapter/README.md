# Real-data dashboard adapter

This ML-owned package is a read-only adapter for the teammate-produced
`features_real/real_outputs/` export. It never joins audio and CSV data,
infers timestamps, fills missing values, renames fault codes, or invokes a
classifier.

The adapter accepts any directory containing the original export filenames:

```python
from ml.real_data_dashboard_adapter import RealDataAdapter

adapter = RealDataAdapter(r"C:\path\to\real_outputs")
meta = adapter.metadata()
recordings = adapter.list_audio_recordings()
rows = adapter.audio_windows(recordings[0]["recording_key"])
spec = adapter.audio_spectrogram(rows[0]["window_id"])["spectrogram"]
codes = adapter.list_csv_fault_codes()
splits = adapter.list_csv_splits()
csv_rows = adapter.csv_windows(split="test")
```

Audio retains fuel labels and split assignments; its engine-fault label is
always documented as `unknown`. CSV rows retain numeric fault codes `0`, `1`,
`2`, and `3`; their meanings remain `unknown`. CSV plotting must use
`row_start`/`row_end` because the export has no physical timestamps.

Dependencies are `numpy` and `pandas` (both already in `ml/requirements.txt`).
The adapter is not a model interface and does not accept the synthetic V2
167-feature contract.
