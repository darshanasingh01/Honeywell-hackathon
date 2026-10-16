# Real carOBD telemetry import

This isolated ML-owned dataset contains selected original CSV records from
[`eron93br/carOBD`](https://github.com/eron93br/carOBD), a Toyota Etios 2014
OBD-II acquisition. The upstream README describes 27 PIDs acquired with a
Carloop logger at 1 Hz. The import contains `idle1.csv` and `drive1.csv`; the
dashboard export uses the first 120 records of each and retains every source
value without resampling, interpolation, or imputation.

Files:

- `raw/idle1.csv` and `raw/drive1.csv`: downloaded source records, unchanged.
- `manifest.json`: hashes, provenance, source columns, mapping, sampling, and limitations.
- `dashboard_timeseries.json`: compact timestamped telemetry for dashboard use.

Mapped fields are RPM, engine load, intake MAP, throttle, coolant temperature,
short/long fuel trims, and timing advance. The source misspells its timestamp
column as `ENGINE_RUN_TINE`; this spelling is preserved and exported as
`timestamp_s`. Blank cells remain `null`. Source headers do not specify units;
mapped units are standard OBD interpretations and are marked inferred in the
manifest.

The data is real recorded telemetry but has no verified fault labels. It must
not be merged with synthetic training data, real audio, or treated as a
diagnostic evaluation set. The upstream README requests citation and no formal
LICENSE file was found; review upstream terms before redistribution.
