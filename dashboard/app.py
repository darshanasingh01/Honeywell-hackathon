"""Engine Health Monitor — Honeywell HackHive PS2 dashboard.

Reads Person B's real outputs (data/raw waveforms + telemetry, data/processed
features) and renders an interactive fault-diagnosis dashboard with a Light/Dark
theme toggle. The classifier is a nearest-centroid PREVIEW standing in for Person
C's model — swap `preview_predict()` for their output when it lands.

Run from the repo root:
    streamlit run dashboard/app.py
Needs: streamlit plotly scipy pandas numpy. Generate data first:
    python -m feature_extraction.generate_dataset
    python -m feature_extraction.pipeline
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import spectrogram

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FS = 25600

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
NICE = {"normal": "Normal", "knock": "Knock", "misfire": "Misfire",
        "valve_train_wear": "Valve-train wear", "overheating": "Overheating"}
CONDITIONS = ["normal", "knock", "misfire", "valve_train_wear", "overheating"]

# ---- two validated theme instances ----
THEMES = {
    "Dark": dict(
        plane="#0b0b0b", surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#8b8a84",
        grid="#2a2a2c", card_border="rgba(255,255,255,.08)",
        card_grad="linear-gradient(160deg, rgba(255,255,255,.04), rgba(255,255,255,0))",
        app_bg="radial-gradient(1200px 500px at 15% -5%, #1b1420 0%, transparent 55%),"
               "radial-gradient(1000px 480px at 95% 0%, #101a24 0%, transparent 50%), #0b0b0b",
        hero="linear-gradient(115deg, rgba(40,26,20,.9) 0%, rgba(26,20,34,.85) 50%, rgba(16,26,36,.9) 100%)",
        classes={"normal": "#3987e5", "knock": "#d95926", "misfire": "#199e70",
                 "valve_train_wear": "#c98500", "overheating": "#d55181"},
        plotly="plotly_dark", polar_bg="rgba(255,255,255,.02)", panel_bg="rgba(255,255,255,.03)",
        spec_scale=[[0.0, "#0a0a0a"], [0.25, "#241017"], [0.5, "#7a2f20"],
                    [0.72, "#d95926"], [0.88, "#f5a524"], [1.0, "#ffe9a8"]],
    ),
    "Light": dict(
        plane="#f1f1ee", surface="#ffffff", ink="#121211", ink2="#52514e", muted="#8a8984",
        grid="#e5e4de", card_border="rgba(0,0,0,.10)",
        card_grad="linear-gradient(160deg, rgba(0,0,0,.025), rgba(0,0,0,0))",
        app_bg="radial-gradient(1200px 520px at 12% -8%, #fdeee6 0%, transparent 55%),"
               "radial-gradient(1000px 480px at 96% -4%, #e9f1fb 0%, transparent 52%), #f1f1ee",
        hero="linear-gradient(115deg, #fff3ea 0%, #f3eefb 52%, #e9f2fc 100%)",
        classes={"normal": "#2a78d6", "knock": "#eb6834", "misfire": "#1baf7a",
                 "valve_train_wear": "#eda100", "overheating": "#e87ba4"},
        plotly="plotly_white", polar_bg="rgba(0,0,0,.015)", panel_bg="rgba(0,0,0,.015)",
        spec_scale=[[0.0, "#fcfcfb"], [0.28, "#f7cf9a"], [0.55, "#eb6834"],
                    [0.78, "#b5371b"], [1.0, "#5e1e12"]],
    ),
}

st.set_page_config(page_title="Engine Health Monitor", page_icon="🔧", layout="wide")

# ---------------- sidebar / state ----------------
qp = st.query_params
qp_cond = qp.get("cond", "knock"); qp_cond = qp_cond if qp_cond in CONDITIONS else "knock"
try:
    qp_seed = int(qp.get("seed", 0))
except ValueError:
    qp_seed = 0
qp_seed = qp_seed if qp_seed in range(6) else 0
qp_theme = qp.get("theme", "Dark"); qp_theme = qp_theme if qp_theme in THEMES else "Dark"

with st.sidebar:
    st.markdown("<div class='sect'>🔧 Controls</div>", unsafe_allow_html=True)
    theme_name = st.radio("Theme", list(THEMES), index=list(THEMES).index(qp_theme), horizontal=True)
    cond = st.selectbox("Engine condition (simulated run)", CONDITIONS,
                        index=CONDITIONS.index(qp_cond), format_func=lambda c: NICE[c])
    seed = st.select_slider("Run / seed", options=[0, 1, 2, 3, 4, 5], value=qp_seed)
    channel = st.radio("Spectrogram channel", ["knock_sensor", "microphone_audio"],
                       format_func=lambda c: "Knock sensor" if c == "knock_sensor" else "Microphone",
                       horizontal=True)
    t0 = st.slider("Inspection window start (s)", 0.0, 9.5, 1.0, 0.25)
    st.query_params.update({"cond": cond, "seed": str(seed), "theme": theme_name})

T = THEMES[theme_name]
INK, INK2, MUTED, GRID = T["ink"], T["ink2"], T["muted"], T["grid"]
CLASS_COLORS = T["classes"]
accent_pulse = CLASS_COLORS["knock"]

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] {{ font-family: 'Inter', system-ui, sans-serif; }}
      .stApp {{ background: {T['app_bg']}; }}
      #MainMenu, footer, header {{ visibility: hidden; }}
      .block-container {{ padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1560px; }}
      .hero {{ background: {T['hero']}; border: 1px solid {T['card_border']}; border-radius: 18px;
        padding: 22px 28px; margin-bottom: 18px; backdrop-filter: blur(8px);
        box-shadow: 0 10px 40px rgba(0,0,0,.18); }}
      .hero h1 {{ color:{INK}; font-size: 1.8rem; margin:0; font-weight:800; letter-spacing:-.025em; }}
      .hero p {{ color:{INK2}; margin:.35rem 0 0; font-size:.95rem; }}
      .pill {{ display:inline-flex; align-items:center; gap:7px; padding:4px 12px; border-radius:999px;
        font-size:.72rem; font-weight:700; letter-spacing:.07em; vertical-align:middle; margin-left:10px; }}
      .dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
      .live .dot {{ animation: pulse 1.8s infinite; }}
      @keyframes pulse {{ 0%{{box-shadow:0 0 0 0 {accent_pulse}99}} 70%{{box-shadow:0 0 0 9px {accent_pulse}00}} 100%{{box-shadow:0 0 0 0 {accent_pulse}00}} }}
      .kpi {{ background:{T['card_grad']}; border:1px solid {T['card_border']}; border-radius:16px;
        padding:15px 17px; position:relative; overflow:hidden; transition:transform .18s ease, border-color .18s ease; }}
      .kpi:hover {{ transform:translateY(-3px); border-color:var(--accent); }}
      .kpi::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px;
        background:var(--accent); box-shadow:0 0 16px var(--accent); }}
      .kpi .lbl {{ color:{MUTED}; font-size:.70rem; text-transform:uppercase; letter-spacing:.08em; font-weight:600; }}
      .kpi .val {{ color:{INK}; font-size:1.6rem; font-weight:800; margin-top:4px; line-height:1.05; letter-spacing:-.02em; }}
      .kpi .sub {{ color:{INK2}; font-size:.77rem; margin-top:3px; }}
      .sect {{ color:{INK}; font-size:1.0rem; font-weight:700; margin:.2rem 0 .5rem; letter-spacing:-.01em; }}
      .note {{ color:{MUTED}; font-size:.75rem; font-style:italic; }}
      [data-testid="stSidebar"] {{ background:{T['surface']}; border-right:1px solid {T['card_border']}; }}
      [data-testid="stSidebar"] * {{ color:{INK} !important; }}
      [data-testid="stSidebar"] .sect {{ font-size:1.05rem; }}
    </style>
    """, unsafe_allow_html=True)


if theme_name == "Light":
    st.markdown(
        """<style>
          [data-testid="stSidebar"] div[data-baseweb="select"] > div,
          [data-testid="stSidebar"] div[data-baseweb="input"] > div {
            background:#ffffff !important; border-color:#d9d8d2 !important; }
          [data-testid="stSidebar"] div[data-baseweb="select"] * { color:#121211 !important; }
          div[data-baseweb="popover"] li { background:#ffffff !important; color:#121211 !important; }
          [data-testid="stSidebar"] [role="radiogroup"] * { color:#121211 !important; }
        </style>""", unsafe_allow_html=True)


@st.cache_data
def load_features():
    return pd.read_csv(PROC / "features.csv")


@st.cache_data
def load_telemetry(cond, seed):
    return pd.read_csv(RAW / f"{cond}_telemetry_s{seed}.csv")


@st.cache_data
def load_waveform(cond, seed):
    return pd.read_csv(RAW / f"{cond}_waveform_s{seed}.csv")


SIG_FEATS = [("knock_knock_5k_10k_ratio", "5–10 kHz ratio"), ("knock_cycle_pct_knocking", "% cycles knocking"),
             ("knock_mid_1k_5k_ratio", "1–5 kHz ratio"), ("tel_coolant_temp_c_slope", "coolant slope"),
             ("tel_stft_pct_mean", "fuel trim")]
SIG_NAMES = [n for _, n in SIG_FEATS]
SIG_KEYS = [k for k, _ in SIG_FEATS]


@st.cache_data
def class_centroids():
    return load_features().groupby("label")[SIG_KEYS].mean()


def preview_predict(run_row):
    """PLACEHOLDER nearest-centroid classifier → pseudo-probabilities."""
    cents = class_centroids()
    x = run_row[SIG_KEYS].to_numpy(dtype=float)
    scale = cents.std(axis=0).replace(0, 1).to_numpy()
    d = np.sqrt((((cents.to_numpy() - x) / scale) ** 2).sum(axis=1))
    p = np.exp(-d - (-d).max()); p /= p.sum()
    return pd.Series(p, index=cents.index).sort_values(ascending=False)


def health_from(run_row):
    knock = float(run_row["knock_cycle_pct_knocking"]); cool = float(run_row["tel_coolant_temp_c_slope"])
    fuel = abs(float(run_row["tel_stft_pct_mean"])); valve = max(0.0, float(run_row["knock_mid_1k_5k_ratio"]) - 0.33)
    penalty = min(100, knock * 260 + max(0, cool) * 22 + fuel * 7 + valve * 180)
    score = max(0, 100 - penalty)
    tier, icon = (("good", "🟢") if score >= 80 else ("warning", "🟡") if score >= 55
                  else ("serious", "🟠") if score >= 30 else ("critical", "🔴"))
    return round(score), tier, icon


def style_fig(fig, h=240, title=None, legend=False):
    fig.update_layout(
        template=T["plotly"], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=30 if title else 8, b=8), height=h,
        font=dict(color=INK2, size=11, family="Inter, system-ui, sans-serif"),
        title=dict(text=title, font=dict(color=INK, size=12.5), x=0.01, y=0.97) if title else None,
        showlegend=legend)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


WIN = 0.3
feats = load_features()
run_id = f"{cond}_s{seed}"
run_row = feats[feats.run_id == run_id].mean(numeric_only=True)
tel = load_telemetry(cond, seed)
wav = load_waveform(cond, seed)
accent = CLASS_COLORS[cond]
pred = preview_predict(run_row)
top_pred = pred.index[0]
score, tier, tier_icon = health_from(run_row)

# ---------------- hero ----------------
st.markdown(
    f"""
    <div class='hero'>
      <h1>⚙️ Engine Health Monitor
        <span class='pill live' style='background:{accent}1f; color:{accent}; border:1px solid {accent}66;'>
          <span class='dot' style='background:{accent}'></span>LIVE</span>
        <span class='pill' style='background:{INK}12; color:{INK2};'>HackHive · PS2</span>
      </h1>
      <p>Multivariable engine signature extraction &amp; real-time fault diagnosis —
         microphone · knock · crankshaft @ 25.6 kHz + ECU / CAN telemetry</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- KPI row ----------------
rpm = tel.engine_rpm.mean(); cool_now = tel.coolant_temp_c.iloc[-1]
pct_knock = float(run_row["knock_cycle_pct_knocking"]) * 100
kpis = [
    ("Detected fault", NICE[top_pred], f"{pred.iloc[0]*100:.0f}% confidence", CLASS_COLORS[top_pred]),
    ("Health score", f"{tier_icon} {score}", tier.capitalize(), STATUS[tier]),
    ("Engine speed", f"{rpm:,.0f}", "rpm (mean)", CLASS_COLORS["normal"]),
    ("Coolant", f"{cool_now:.0f}°C", f"slope {run_row['tel_coolant_temp_c_slope']:+.2f} °C/s", CLASS_COLORS["overheating"]),
    ("Cycles knocking", f"{pct_knock:.0f}%", "crank-gated", CLASS_COLORS["knock"]),
]
for c, (lbl, val, sub, acc) in zip(st.columns(len(kpis)), kpis):
    c.markdown(f"<div class='kpi' style='--accent:{acc}'><div class='lbl'>{lbl}</div>"
               f"<div class='val'>{val}</div><div class='sub'>{sub}</div></div>", unsafe_allow_html=True)

st.write("")

# ---------------- diagnosis strip ----------------
d1, d2, d3 = st.columns([0.9, 1.1, 1.1])
with d1:
    st.markdown("<div class='sect'>Health score</div>", unsafe_allow_html=True)
    steps_dark = [[0, 30, "#3a1414"], [30, 55, "#3a2a12"], [55, 80, "#30340f"], [80, 100, "#123020"]]
    steps_light = [[0, 30, "#f6dcdc"], [30, 55, "#f7ead1"], [55, 80, "#eef0d6"], [80, 100, "#d8efe0"]]
    zones = steps_dark if theme_name == "Dark" else steps_light
    g = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"font": {"size": 44, "color": INK, "family": "Inter"}},
        gauge={"axis": {"range": [0, 100], "tickcolor": MUTED, "tickwidth": 1,
                        "tickvals": [0, 25, 50, 75, 100], "tickfont": {"size": 10, "color": MUTED}},
               "bar": {"color": STATUS[tier], "thickness": 0.28}, "bgcolor": T["panel_bg"], "borderwidth": 0,
               "steps": [{"range": [a, b], "color": c} for a, b, c in zones],
               "threshold": {"line": {"color": INK, "width": 3}, "thickness": 0.8, "value": score}}))
    style_fig(g, h=240); g.update_layout(margin=dict(l=36, r=44, t=14, b=0))
    st.plotly_chart(g, use_container_width=True, theme=None, config={"displayModeBar": False})
    st.markdown(f"<div class='note' style='text-align:center'>{tier_icon} {tier.capitalize()} — "
                f"placeholder score (Person C owns the real health model)</div>", unsafe_allow_html=True)

with d2:
    st.markdown("<div class='sect'>Fault fingerprint</div>", unsafe_allow_html=True)
    allc = class_centroids(); norm = allc.abs().max().replace(0, 1)
    cur = (run_row[SIG_KEYS] / norm).clip(0, 1).to_numpy()
    base = (allc.loc["normal"] / norm).clip(0, 1).to_numpy()
    r = go.Figure()
    r.add_trace(go.Scatterpolar(r=np.append(base, base[0]), theta=SIG_NAMES + [SIG_NAMES[0]], fill="toself",
                name="Healthy", line=dict(color=CLASS_COLORS["normal"], width=1.5),
                fillcolor=CLASS_COLORS["normal"] + "2e"))
    r.add_trace(go.Scatterpolar(r=np.append(cur, cur[0]), theta=SIG_NAMES + [SIG_NAMES[0]], fill="toself",
                name=NICE[cond], line=dict(color=accent, width=2.4), fillcolor=accent + "40"))
    r.update_layout(template=T["plotly"], paper_bgcolor="rgba(0,0,0,0)", height=256,
                    margin=dict(l=40, r=40, t=24, b=24), font=dict(color=INK2, size=10, family="Inter"),
                    polar=dict(bgcolor=T["polar_bg"],
                               radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor=GRID),
                               angularaxis=dict(gridcolor=GRID, color=INK2)),
                    legend=dict(orientation="h", y=1.18, x=0.5, xanchor="center", font=dict(size=10)))
    st.plotly_chart(r, use_container_width=True, theme=None, config={"displayModeBar": False})

with d3:
    st.markdown("<div class='sect'>Fault classification</div>", unsafe_allow_html=True)
    p = pred.sort_values()
    bar = go.Figure(go.Bar(y=[NICE[i] for i in p.index], x=p.values * 100, orientation="h",
        marker=dict(color=[CLASS_COLORS[i] for i in p.index], line=dict(color=T["card_border"], width=1)),
        text=[f"{v*100:.0f}%" for v in p.values], textposition="outside",
        textfont=dict(color=INK2, size=11), hovertemplate="%{x:.1f}%<extra></extra>"))
    style_fig(bar, h=256); bar.update_xaxes(range=[0, 112], showgrid=False)
    st.plotly_chart(bar, use_container_width=True, theme=None, config={"displayModeBar": False})
    st.markdown("<div class='note'>⚠ Preview classifier — placeholder for Person C's SVM / ANN / CNN.</div>",
                unsafe_allow_html=True)

st.write("")

# ---------------- spectrogram (whitened) ----------------
st.markdown(f"<div class='sect'>Signature spectrogram — "
            f"{'knock sensor' if channel=='knock_sensor' else 'microphone'} "
            f"@ window {t0:.2f}–{t0+WIN:.2f}s</div>", unsafe_allow_html=True)
t = wav["timestamp_s"].to_numpy(); m = (t >= t0) & (t < t0 + WIN)
sig = wav[channel].to_numpy()[m]
f, tt, Sxx = spectrogram(sig, fs=FS, nperseg=256, noverlap=224, window="hann", scaling="spectrum")
fkhz = f / 1000.0; band = (fkhz >= 2) & (fkhz <= 12)
D = 10 * np.log10(Sxx[band] + 1e-12)
D_w = np.clip(D - np.median(D, axis=1, keepdims=True), 0, None)
band_line = "rgba(90,150,210,.75)" if theme_name == "Light" else "rgba(140,220,255,.55)"
# light surface shows the noise speckle, so lift the floor to keep the background clean
spec_lo = np.percentile(D_w, 72) if theme_name == "Light" else 0
sp = go.Figure(go.Heatmap(z=D_w, x=tt, y=fkhz[band], colorscale=T["spec_scale"],
    zmin=spec_lo, zmax=max(3, np.percentile(D_w, 99.3)),
    colorbar=dict(title="dB↑", thickness=11, len=0.9, outlinewidth=0),
    hovertemplate="t %{x:.3f}s · %{y:.1f} kHz<extra></extra>"))
for yb in (5, 10):
    sp.add_hline(y=yb, line=dict(color=band_line, width=1, dash="dot"))
sp.add_annotation(x=tt[1], y=10.5, text="5–10 kHz knock band", showarrow=False,
                  font=dict(color=band_line, size=10), xanchor="left")
style_fig(sp, h=330); sp.update_yaxes(title="frequency (kHz)"); sp.update_xaxes(title="time (s)")
st.plotly_chart(sp, use_container_width=True, theme=None, config={"displayModeBar": False})
st.markdown("<div class='note'>Noise floor removed per frequency (spectral whitening) — only transient "
            "fault energy shows. Knock → bursts in the 5–10 kHz band; valve wear → 3–5 kHz.</div>",
            unsafe_allow_html=True)

st.write("")

# ---------------- telemetry ----------------
st.markdown("<div class='sect'>ECU / CAN telemetry</div>", unsafe_allow_html=True)
series = [("engine_rpm", "Engine RPM", CLASS_COLORS["normal"]), ("coolant_temp_c", "Coolant °C", CLASS_COLORS["overheating"]),
          ("throttle_pct", "Throttle %", CLASS_COLORS["misfire"]), ("stft_pct", "Short-term fuel trim %", CLASS_COLORS["valve_train_wear"])]
win_fill = "rgba(0,0,0,.08)" if theme_name == "Light" else "rgba(255,255,255,.10)"
tcols = st.columns(4)
for i, (col, name, clr) in enumerate(series):
    fig = go.Figure(go.Scatter(x=tel.timestamp_s, y=tel[col], mode="lines",
                    line=dict(color=clr, width=2, shape="spline"), fill="tozeroy",
                    fillcolor=clr + "14", hovertemplate="t %{x:.1f}s · %{y:.1f}<extra></extra>"))
    fig.add_vrect(x0=t0, x1=t0 + WIN, fillcolor=win_fill, line_width=0, layer="below")
    style_fig(fig, h=170, title=name)
    tcols[i].plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
