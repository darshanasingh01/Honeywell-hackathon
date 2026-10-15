from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from simulator.engine_simulator import CONDITIONS, simulate_engine
from simulator.signal_analysis import frequency_spectrum, time_frequency_spectrogram
from simulator.signal_quality import inspect_signal, check_timestamp_spacing

st.set_page_config(
    page_title="EngineLab | Fault Simulator",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ EngineLab")
st.caption(
    "Synthetic engine fault simulation • Multisensor analysis • "
    "Signal-quality checks"
)

st.warning(
    "Research prototype: signals are synthetic and are not calibrated "
    "or validated against a real engine. Indicators are not confirmed diagnoses."
)

with st.sidebar:
    st.header("Simulation controls")
    condition = st.selectbox(
        "Operating condition",
        CONDITIONS,
        format_func=lambda x: x.replace("_", " ").title(),
    )
    duration = st.slider("Duration (seconds)", 1, 10, 2)
    telemetry_hz = st.select_slider(
        "Telemetry sampling rate (Hz)",
        options=[5, 10, 20, 50, 100],
        value=10,
    )
    signal_hz = st.select_slider(
        "Waveform sampling rate (Hz)",
        options=[2000, 4000, 8000, 12000],
        value=4000,
    )
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    run = st.button("▶ Run simulation", type="primary", width="stretch")

if "simulation" not in st.session_state:
    st.session_state.simulation = None

if run or st.session_state.simulation is None:
    with st.spinner("Generating synthetic engine signals..."):
        telemetry, waveform = simulate_engine(
            condition=condition,
            duration_s=float(duration),
            telemetry_hz=int(telemetry_hz),
            signal_hz=int(signal_hz),
            seed=int(seed),
        )
        st.session_state.simulation = {
            "condition": condition,
            "telemetry": telemetry,
            "waveform": waveform,
            "telemetry_hz": int(telemetry_hz),
            "signal_hz": int(signal_hz),
        }

sim = st.session_state.simulation
telemetry = sim["telemetry"]
waveform = sim["waveform"]

if sim["condition"] != condition:
    st.info("Settings changed. Select **Run simulation** to generate the new condition.")

if sim["condition"] == "normal":
    st.success("Selected scenario: Normal operating baseline")
else:
    st.info(f"Selected scenario: {sim['condition'].replace('_', ' ').title()}")

rpm_mean = telemetry["engine_rpm"].mean()
temp_max = telemetry["coolant_temp_c"].max()
audio_rms = float(np.sqrt(np.mean(waveform["microphone_audio"].to_numpy() ** 2)))
knock_rms = float(np.sqrt(np.mean(waveform["knock_sensor"].to_numpy() ** 2)))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Mean engine speed", f"{rpm_mean:,.0f} RPM")
m2.metric("Peak coolant temperature", f"{temp_max:.1f} °C")
m3.metric("Microphone RMS", f"{audio_rms:.4f}")
m4.metric("Knock sensor RMS", f"{knock_rms:.4f}")

tab_signals, tab_frequency, tab_quality, tab_data = st.tabs(
    ["📈 Signals", "🌈 Frequency analysis", "🧪 Signal quality", "📦 Data & export"]
)

with tab_signals:
    st.subheader("Operating conditions")

    st.caption("Engine speed")
    st.line_chart(
        telemetry.set_index("timestamp_s")[["engine_rpm"]],
        height=220,
    )

    st.caption("Coolant temperature")
    st.line_chart(
        telemetry.set_index("timestamp_s")[["coolant_temp_c"]],
        height=200,
    )

    st.caption("Driver demand and engine load")
    st.line_chart(
        telemetry.set_index("timestamp_s")[["throttle_pct", "engine_load"]],
        height=220,
    )

    st.caption("Fuel-trim response")
    st.line_chart(
        telemetry.set_index("timestamp_s")[["stft_pct", "ltft_pct"]],
        height=220,
    )

    st.subheader("High-frequency waveform")
    st.caption(
        f"Showing a maximum of 2,000 points for readability. "
        f"Original sampling rate: {sim['signal_hz']:,} Hz."
    )
    display_n = min(2000, len(waveform))
    indices = np.linspace(0, len(waveform) - 1, display_n, dtype=int)
    view = waveform.iloc[indices].set_index("timestamp_s")
    st.line_chart(
        view[["microphone_audio", "knock_sensor"]],
        height=280,
    )
    st.line_chart(
        view[["accel_x", "accel_y", "accel_z"]],
        height=240,
    )
    st.line_chart(view[["crank_pulse"]], height=150)

with tab_frequency:
    st.subheader("Frequency spectrum and spectrogram")
    st.caption(
        "These plots reveal signal patterns for investigation; they do not "
        "prove a particular fault."
    )

    signal_choice = st.selectbox(
        "Signal to analyse",
        ["microphone_audio", "knock_sensor", "accel_x", "accel_y", "accel_z"],
    )
    max_seconds = len(waveform) / sim["signal_hz"]
    window_seconds = st.slider(
        "Analysis window (seconds)",
        min_value=0.25,
        max_value=float(max(0.25, min(2.0, max_seconds))),
        value=float(min(1.0, max_seconds)),
        step=0.25,
    )

    sample_rate = sim["signal_hz"]
    sample_start = min(
        int(len(waveform) * 0.5),
        max(0, int((max_seconds - window_seconds) * sample_rate / 2)),
    )
    sample_count = min(
        len(waveform) - sample_start,
        max(16, int(window_seconds * sample_rate)),
    )
    signal_values = waveform[signal_choice].to_numpy()[
        sample_start:sample_start + sample_count
    ]

    if len(signal_values) < 16:
        st.warning("Not enough samples for frequency analysis.")
    else:
        spectrum_hz, amplitude = frequency_spectrum(signal_values, sample_rate)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(spectrum_hz, amplitude, linewidth=0.8)
        ax.set_title(f"One-sided amplitude spectrum — {signal_choice}")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude (synthetic units)")
        ax.set_xlim(0, sample_rate / 2)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            f"Nyquist limit: {sample_rate / 2:.0f} Hz. "
            f"Approximate FFT-bin spacing: {sample_rate / len(signal_values):.2f} Hz."
        )

        times, frequencies, power = time_frequency_spectrogram(
            signal_values,
            sample_rate,
            nperseg=min(512, len(signal_values)),
        )
        power_db = 10 * np.log10(np.maximum(power, 1e-12))

        fig, ax = plt.subplots(figsize=(10, 4))
        mesh = ax.pcolormesh(
            times, frequencies, power_db,
            shading="auto", cmap="magma",
        )
        ax.set_title(f"STFT spectrogram — {signal_choice}")
        ax.set_xlabel("Time within selected window (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(mesh, ax=ax, label="Power spectral density (dB/Hz)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "Interpretation caution: spectral peaks and bright patches may "
            "arise from engine speed, load, background noise, sensor placement "
            "or sampling artefacts. Compare them with CKP timing, vibration, "
            "RPM and temperature before raising a fault hypothesis."
        )

with tab_quality:
    st.subheader("Automated signal-quality checks")
    st.caption(
        "Quality checks identify data limitations. They do not diagnose "
        "engine faults by themselves."
    )

    signal_columns = [
        "microphone_audio", "knock_sensor",
        "accel_x", "accel_y", "accel_z",
    ]
    quality_rows = []

    for column in signal_columns:
        values = waveform[column].to_numpy()
        result = inspect_signal(values, sim["signal_hz"])
        quality_rows.append({
            "Signal": column,
            "Finite fraction": result.get("finite_fraction", np.nan),
            "Quality OK": result.get("quality_ok", False),
            "Warnings": ", ".join(result.get("warnings", [])) or "None",
        })

    st.dataframe(
        pd.DataFrame(quality_rows),
        width="stretch",
        hide_index=True,
    )

    spacing = check_timestamp_spacing(
        waveform["timestamp_s"].to_numpy(), sim["signal_hz"]
    )
    st.write("**Waveform timestamp validation**")
    st.json(spacing)

    nyquist_hz = sim["signal_hz"] / 2
    st.metric("Nyquist frequency", f"{nyquist_hz:,.0f} Hz")
    st.write(
        "Frequencies above the Nyquist limit cannot be uniquely represented "
        "by this sampling rate. A higher sampling rate alone does not prove "
        "that a signal is alias-free; the acquisition chain also needs "
        "appropriate anti-alias filtering."
    )

    st.subheader("Cross-sensor interpretation")
    st.write(
        f"- Scenario label: **{sim['condition'].replace('_', ' ').title()}** "
        "(known simulation setting, not a model prediction)."
    )
    st.write(
        f"- Microphone RMS: **{audio_rms:.4f}**; "
        f"knock sensor RMS: **{knock_rms:.4f}**."
    )
    st.write(
        f"- Mean RPM: **{rpm_mean:.0f}**; maximum coolant temperature: "
        f"**{temp_max:.1f} °C**."
    )
    st.info(
        "A real fault detector should compare features across aligned time "
        "windows and operating conditions, then report confidence and "
        "supporting evidence. A spectrogram patch alone is not proof of "
        "knock or valve-train wear."
    )

with tab_data:
    st.subheader("Dataset overview")
    c1, c2 = st.columns(2)
    c1.metric("Telemetry rows", f"{len(telemetry):,}")
    c2.metric("Waveform rows", f"{len(waveform):,}")

    st.write("**Telemetry preview**")
    st.dataframe(telemetry.head(20), width="stretch")
    st.write("**Waveform preview**")
    st.dataframe(waveform.head(20), width="stretch")

    st.download_button(
        "⬇ Download telemetry CSV",
        telemetry.to_csv(index=False).encode("utf-8"),
        file_name=f"{sim['condition']}_telemetry.csv",
        mime="text/csv",
        width="stretch",
    )
    st.download_button(
        "⬇ Download waveform CSV",
        waveform.to_csv(index=False).encode("utf-8"),
        file_name=f"{sim['condition']}_waveform.csv",
        mime="text/csv",
        width="stretch",
    )

st.divider()
st.caption(
    "EngineLab • Synthetic demonstration only • "
    "Validate against real, labelled engine data before making diagnostic claims."
)
