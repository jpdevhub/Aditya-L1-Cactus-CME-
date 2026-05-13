import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌞 ISRO CME Prediction System",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 50%, #FFD700 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,107,53,0.25);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.6rem 0;
        color: #fff;
    }
    .card-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #FF6B35;
        margin-bottom: 0.4rem;
    }
    .card-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #fff;
    }
    .card-sub {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 0.2rem;
    }
    .badge-1 { background:#2e7d32; color:#fff; border-radius:8px; padding:3px 10px; font-size:0.8rem; }
    .badge-2 { background:#f57f17; color:#fff; border-radius:8px; padding:3px 10px; font-size:0.8rem; }
    .badge-3 { background:#e65100; color:#fff; border-radius:8px; padding:3px 10px; font-size:0.8rem; }
    .badge-4 { background:#b71c1c; color:#fff; border-radius:8px; padding:3px 10px; font-size:0.8rem; }
    .insight-box {
        background: rgba(255,107,53,0.08);
        border-left: 4px solid #FF6B35;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
        color: #ddd;
        font-size: 0.92rem;
    }
    .stSlider > div[data-baseweb="slider"] { padding-top: 4px; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Feature names expected by the models ────────────────────────────────────
FEATURE_COLS = [
    "cme_width", "pa",
    "pre_cme_12h_proton_density_mean", "pre_cme_12h_proton_density_std",
    "pre_cme_12h_proton_density_max", "pre_cme_12h_proton_density_min",
    "pre_cme_12h_proton_density_trend",
    "pre_cme_12h_proton_bulk_speed_mean", "pre_cme_12h_proton_bulk_speed_std",
    "pre_cme_12h_proton_bulk_speed_max", "pre_cme_12h_proton_bulk_speed_min",
    "pre_cme_12h_proton_bulk_speed_trend",
    "pre_cme_12h_proton_temperature_mean", "pre_cme_12h_proton_temperature_std",
    "pre_cme_12h_proton_temperature_max", "pre_cme_12h_proton_temperature_min",
    "pre_cme_12h_proton_temperature_trend",
    "pre_cme_12h_alpha_density_mean", "pre_cme_12h_alpha_density_std",
    "pre_cme_12h_alpha_density_max", "pre_cme_12h_alpha_density_min",
    "pre_cme_12h_alpha_density_trend",
    "pre_cme_12h_alpha_proton_ratio_mean", "pre_cme_12h_alpha_proton_ratio_std",
    "pre_cme_12h_alpha_proton_ratio_max", "pre_cme_12h_alpha_proton_ratio_min",
    "pre_cme_12h_alpha_proton_ratio_trend",
    "post_cme_12h_proton_density_mean", "post_cme_12h_proton_density_std",
    "post_cme_12h_proton_density_max", "post_cme_12h_proton_density_min",
    "post_cme_12h_proton_density_trend",
    "post_cme_12h_proton_bulk_speed_mean", "post_cme_12h_proton_bulk_speed_std",
    "post_cme_12h_proton_bulk_speed_max", "post_cme_12h_proton_bulk_speed_min",
    "post_cme_12h_proton_bulk_speed_trend",
    "post_cme_12h_proton_temperature_mean", "post_cme_12h_proton_temperature_std",
    "post_cme_12h_proton_temperature_max", "post_cme_12h_proton_temperature_min",
    "post_cme_12h_proton_temperature_trend",
    "post_cme_12h_alpha_density_mean", "post_cme_12h_alpha_density_std",
    "post_cme_12h_alpha_density_max", "post_cme_12h_alpha_density_min",
    "post_cme_12h_alpha_density_trend",
    "post_cme_12h_alpha_proton_ratio_mean", "post_cme_12h_alpha_proton_ratio_std",
    "post_cme_12h_alpha_proton_ratio_max", "post_cme_12h_alpha_proton_ratio_min",
    "post_cme_12h_alpha_proton_ratio_trend",
    "cme_kinetic_energy", "cme_momentum",
    "cme_hour", "cme_day_of_year", "cme_month",
]

# 30 features for Velocity Regressor (no circular cme_kinetic_energy / cme_momentum)
VELOCITY_FEATURE_COLS = [
    "cme_width", "pa",
    "pre_cme_12h_proton_density_mean", "pre_cme_12h_proton_density_std",
    "pre_cme_12h_proton_density_max", "pre_cme_12h_proton_density_min",
    "pre_cme_12h_proton_density_trend",
    "pre_cme_12h_proton_bulk_speed_mean", "pre_cme_12h_proton_bulk_speed_std",
    "pre_cme_12h_proton_bulk_speed_max", "pre_cme_12h_proton_bulk_speed_min",
    "pre_cme_12h_proton_bulk_speed_trend",
    "pre_cme_12h_proton_temperature_mean", "pre_cme_12h_proton_temperature_std",
    "pre_cme_12h_proton_temperature_max", "pre_cme_12h_proton_temperature_min",
    "pre_cme_12h_proton_temperature_trend",
    "pre_cme_12h_alpha_density_mean", "pre_cme_12h_alpha_density_std",
    "pre_cme_12h_alpha_density_max", "pre_cme_12h_alpha_density_min",
    "pre_cme_12h_alpha_density_trend",
    "pre_cme_12h_alpha_proton_ratio_mean", "pre_cme_12h_alpha_proton_ratio_std",
    "pre_cme_12h_alpha_proton_ratio_max", "pre_cme_12h_alpha_proton_ratio_min",
    "pre_cme_12h_alpha_proton_ratio_trend",
    "cme_hour", "cme_day_of_year", "cme_month",
]


def build_feature_vectors(inputs: dict):
    """
    Map UI slider inputs to feature arrays for all three models.
    Returns:
      fv_full  — 57-feature array for Halo Class + Earth-Directed models
      fv_vel   — 30-feature array for Velocity Regressor (no circular features)
    """
    density   = inputs["proton_density"]        # n/cc
    speed     = inputs["proton_velocity"]       # km/s
    temp      = inputs["proton_temperature"]    # K
    cme_width = inputs["cme_width"]             # degrees
    pa        = inputs["pa"]                    # position angle
    now       = inputs["now"]                   # datetime

    alpha_density      = density * 0.04   # canonical solar wind alpha fraction
    alpha_proton_ratio = 0.04

    # Kinetic energy & momentum (for halo/earth models — NOT for velocity model)
    m_p        = 1.67e-27  # kg
    kin_energy = 0.5 * density * 1e6 * m_p * (speed * 1e3) ** 2
    momentum   = density * 1e6 * m_p * (speed * 1e3)

    def stats(val, frac=0.12):
        """Return (mean, std, max, min, trend) with ±frac variation."""
        return val, val * frac, val * (1 + frac), val * (1 - frac), 0.0

    d  = stats(density);            s = stats(speed);           t = stats(temp)
    a  = stats(alpha_density, 0.15); r = stats(alpha_proton_ratio, 0.10)
    pd = stats(density * 0.90);     ps = stats(speed * 1.08);  pt = stats(temp * 1.05)
    pa_d = stats(alpha_density * 0.90, 0.15)
    pr   = stats(alpha_proton_ratio * 1.02, 0.10)

    hour = float(now.hour)
    doy  = float(now.timetuple().tm_yday)
    mon  = float(now.month)

    # Full 57-feature vector (halo class + earth-directed)
    fv_full = np.array([
        cme_width, pa,
        *d, *s, *t, *a, *r,
        *pd, *ps, *pt, *pa_d, *pr,
        kin_energy, momentum,
        hour, doy, mon,
    ], dtype=float).reshape(1, -1)

    # 30-feature vector (velocity regressor — no kinetic energy / momentum)
    fv_vel = np.array([
        cme_width, pa,
        *d, *s, *t, *a, *r,
        hour, doy, mon,
    ], dtype=float).reshape(1, -1)

    return fv_full, fv_vel


# ── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    paths = {
        "halo":  "models/halo_class_model.pkl",
        "vel":   "models/velocity_regressor.pkl",
        "earth": "models/earth_directed_model.pkl",
    }
    all_ok = True
    for key, path in paths.items():
        try:
            models[key] = joblib.load(path)
        except Exception as e:
            st.error(f"❌ Could not load `{path}`: {e}")
            all_ok = False

    if all_ok:
        halo_acc  = models["halo"]["test_accuracy"]
        earth_acc = models["earth"]["test_accuracy"]
        vel_mae   = models["vel"]["mae_kms"]
        vel_r2    = models["vel"]["r2"]
        st.success("✅ All 3 models loaded successfully")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Halo Class Accuracy", f"{halo_acc:.1%}")
        col_b.metric("Velocity MAE", f"{vel_mae:.1f} km/s", help=f"R² = {vel_r2:.4f}")
        col_c.metric("Earth-Directed Accuracy", f"{earth_acc:.1%}")

    return models if all_ok else None


def predict(models, fv_full, fv_vel):
    """Run all three models and return results dict."""
    results = {}

    # --- Halo Class (57 features) ---
    pkg = models["halo"]
    X = pkg["scaler"].transform(fv_full)
    results["halo_class"] = int(pkg["model"].predict(X)[0])
    results["halo_proba"] = pkg["model"].predict_proba(X)[0].tolist()
    results["halo_classes"] = pkg["classes"]

    # --- CME Velocity (30 features — no circular kinetic energy) ---
    pkg = models["vel"]
    X = pkg["scaler"].transform(fv_vel)
    results["cme_velocity"] = float(pkg["model"].predict(X)[0])

    # --- Earth Directed (57 features) ---
    pkg = models["earth"]
    X = pkg["scaler"].transform(fv_full)
    results["earth_directed"] = int(pkg["model"].predict(X)[0])
    results["earth_proba"] = pkg["model"].predict_proba(X)[0].tolist()

    return results


# ── Main app ─────────────────────────────────────────────────────────────────
def main():
    st.markdown('<h1 class="main-header">🌞 ISRO CME Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Aditya-L1 · APEX SWISS · CACTUS | Real-time Space Weather ML Forecasting</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Load all three models
    models = load_models()
    if models is None:
        st.error("⚠️ Please run `python3 train_models_v2.py` first to generate the models.")
        return

    # ── Sidebar inputs ────────────────────────────────────────────────────────
    st.sidebar.header("🛰️ Solar Wind Parameters")
    st.sidebar.markdown("*Enter current L1 solar wind measurements*")

    with st.sidebar:
        st.markdown("### 🌬️ Plasma Parameters")
        proton_density     = st.slider("Proton Density (n/cc)", 0.1, 50.0, 8.0, 0.1,
                                       help="Solar wind proton number density")
        proton_velocity    = st.slider("Proton Velocity (km/s)", 200, 1600, 450, 10,
                                       help="Solar wind bulk velocity at L1")
        proton_temperature = st.slider("Proton Temperature (K)", 10_000, 500_000, 80_000, 1_000,
                                       help="Solar wind proton temperature")

        st.markdown("### 🧲 Magnetic Field")
        bx = st.slider("Bx (nT)", -30.0, 30.0, 0.0, 0.1)
        by = st.slider("By (nT)", -30.0, 30.0, 0.0, 0.1)
        bz = st.slider("Bz (nT)", -30.0, 30.0, 0.0, 0.1)
        b_total = np.sqrt(bx**2 + by**2 + bz**2)

        st.markdown("### ☀️ CME Geometry")
        cme_width = st.slider("CME Angular Width (°)", 5, 360, 60, 5,
                              help="Angular width of the CME (CACTUS observation)")
        pa = st.slider("Position Angle (°)", 0.0, 360.0, 180.0, 1.0,
                       help="Central position angle of the CME")

    # ── Build feature vectors ────────────────────────────────────────────────
    now = datetime.now()
    inputs = {
        "proton_density":     proton_density,
        "proton_velocity":    proton_velocity,
        "proton_temperature": proton_temperature,
        "cme_width":          cme_width,
        "pa":                 pa,
        "now":                now,
    }
    fv_full, fv_vel = build_feature_vectors(inputs)

    # ── Live conditions panel ─────────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 📡 Current L1 Solar Wind Conditions")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Proton Density", f"{proton_density:.1f} n/cc")
        m2.metric("Velocity", f"{proton_velocity} km/s")
        m3.metric("Temperature", f"{proton_temperature/1000:.0f} kK")
        m4.metric("|B| Total", f"{b_total:.1f} nT")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Bx", f"{bx:.1f} nT")
        m6.metric("By", f"{by:.1f} nT")
        m7.metric("Bz", f"{bz:.1f} nT", delta=f"{bz:.1f} nT")
        m8.metric("CME Width", f"{cme_width}°")

        st.markdown("---")

        # ── Prediction button ──────────────────────────────────────────────
        if st.button("🚀 Predict CME Properties", type="primary", width="stretch"):
            with st.spinner("Running ML models…"):
                res = predict(models, fv_full, fv_vel)

            halo_class = res["halo_class"]
            cme_vel    = res["cme_velocity"]
            earth_dir  = res["earth_directed"]
            halo_proba = res["halo_proba"]
            earth_proba= res["earth_proba"]

            # ── Result cards ───────────────────────────────────────────────
            st.markdown("## 🎯 Prediction Results")

            r1, r2, r3 = st.columns(3)

            # Card 1 — Halo CME Class
            class_info = {
                1: ("Class 1 — Narrow CME", "width < 120°",   "badge-1", "🟢"),
                2: ("Class 2 — Partial Halo", "120°–300°",    "badge-2", "🟡"),
                3: ("Class 3 — Wide Halo",  "300°–360°",      "badge-3", "🟠"),
                4: ("Class 4 — Full Halo",  "width ≥ 360°",   "badge-4", "🔴"),
            }
            label, desc, badge, emoji = class_info.get(halo_class, (f"Class {halo_class}", "", "badge-1", "⭕"))
            conf = halo_proba[res["halo_classes"].index(halo_class)] if halo_class in res["halo_classes"] else 0.0

            with r1:
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">🏷️ Halo CME Class</div>
                    <div class="card-value">{emoji} {halo_class}</div>
                    <div class="card-sub">{label}</div>
                    <div class="card-sub">{desc}</div>
                    <div style="margin-top:0.5rem">
                        <span class="{badge}">Confidence {conf:.1%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Card 2 — Predicted CME Velocity
            with r2:
                vel_cat = "Slow" if cme_vel < 400 else "Medium" if cme_vel < 700 else "Fast" if cme_vel < 1000 else "Very Fast"
                vel_color = "#2e7d32" if cme_vel < 400 else "#f57f17" if cme_vel < 700 else "#e65100" if cme_vel < 1000 else "#b71c1c"
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">🚀 Predicted CME Velocity</div>
                    <div class="card-value">{cme_vel:.0f} km/s</div>
                    <div class="card-sub">Category: <b style="color:{vel_color}">{vel_cat}</b></div>
                    <div class="card-sub">Model: GradientBoosting Regressor</div>
                </div>
                """, unsafe_allow_html=True)

            # Card 3 — Earth-Directed Risk
            with r3:
                if earth_dir == 1:
                    risk_label = "🔴 Earth-Directed"
                    risk_desc  = "CME likely impacts Earth's magnetosphere"
                    risk_color = "#b71c1c"
                    risk_badge = "badge-4"
                else:
                    risk_label = "🟢 Not Earth-Directed"
                    risk_desc  = "CME will miss or graze Earth"
                    risk_color = "#2e7d32"
                    risk_badge = "badge-1"
                earth_conf = earth_proba[earth_dir]

                st.markdown(f"""
                <div class="card">
                    <div class="card-title">🌍 Earth Impact Risk</div>
                    <div class="card-value" style="font-size:1.5rem">{risk_label}</div>
                    <div class="card-sub">{risk_desc}</div>
                    <div style="margin-top:0.5rem">
                        <span class="{risk_badge}">Confidence {earth_conf:.1%}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Probability bar chart ──────────────────────────────────────
            st.markdown("#### 📊 Halo Class Probability Distribution")
            class_labels = [f"Class {c}" for c in res["halo_classes"]]
            fig_bar = go.Figure(go.Bar(
                x=class_labels,
                y=[p * 100 for p in halo_proba],
                marker_color=["#2e7d32", "#f57f17", "#e65100", "#b71c1c"][:len(halo_proba)],
                text=[f"{p:.1%}" for p in halo_proba],
                textposition="outside",
            ))
            fig_bar.update_layout(
                yaxis_title="Probability (%)",
                yaxis_range=[0, 110],
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ccc"),
                height=280,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_bar, width="stretch")

            # ── Physics insights ───────────────────────────────────────────
            st.markdown("#### 🔬 Physical Parameter Insights")
            insights = []
            if bz < -10:
                insights.append("⚠️ Strong southward Bz detected — enhanced magnetospheric coupling expected")
            if bz < -5:
                insights.append("🧲 Southward Bz component — geomagnetic storm conditions possible")
            if proton_velocity > 800:
                insights.append("🚀 Very fast solar wind — CME transit time will be short (<24h likely)")
            elif proton_velocity > 600:
                insights.append("⚡ Elevated solar wind speed — watch for shock arrival")
            if proton_density > 20:
                insights.append("📊 High proton density — enhanced dynamic pressure on magnetopause")
            if b_total > 15:
                insights.append("🧲 Strong total magnetic field — elevated risk of geomagnetic storm")
            if cme_width >= 300:
                insights.append("🌐 Wide angular CME — Earth-directed probability is high")
            if halo_class >= 3 and earth_dir == 1:
                insights.append("🚨 Wide halo + Earth-directed: significant space weather event expected")
            if not insights:
                insights.append("✅ Current conditions appear within typical solar wind parameters")

            for ins in insights:
                st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

    # ── Radar chart panel ────────────────────────────────────────────────────
    with col2:
        st.markdown("### 📊 Parameter Radar")
        params = ["Density", "Velocity", "Temp", "B Total", "CME Width", "|Bz|"]
        values = [
            proton_density / 50 * 100,
            proton_velocity / 1600 * 100,
            proton_temperature / 500_000 * 100,
            b_total / 50 * 100,
            cme_width / 360 * 100,
            abs(bz) / 30 * 100,
        ]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=params + [params[0]],
            fill="toself",
            name="Current Conditions",
            line_color="#FF6B35",
            fillcolor="rgba(255,107,53,0.25)",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9))),
            showlegend=False,
            title="Solar Wind Profile",
            height=380,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig_radar, width="stretch")

        # ── Model info panel ───────────────────────────────────────────────
        st.markdown("### 🤖 Model Details")
        with st.expander("Click to expand"):
            halo_pkg  = models["halo"]
            vel_pkg   = models["vel"]
            earth_pkg = models["earth"]
            st.markdown(f"""
| | Model | Accuracy / Error |
|---|---|---|
| 🏷️ | Halo Class | {halo_pkg['test_accuracy']:.2%} |
| 🚀 | Velocity   | MAE {vel_pkg['mae_kms']:.1f} km/s · R² {vel_pkg['r2']:.4f} |
| 🌍 | Earth-Directed | {earth_pkg['test_accuracy']:.2%} |

**Training Dataset**: `halo_classification_dataset.csv`  
**Samples**: 1,062 (SMOTE-balanced for Class model)  
**Features**: {len(FEATURE_COLS)} physics-derived features  
**Algorithms**: GradientBoosting + RandomForest  
            """)

        st.markdown("### 📚 Citation")
        st.markdown("""
<small>
K. Singh, K. K. Jha, B. K. Das, and B. D. Biswas,<br>
"A machine learning framework for CME prediction from L1 solar wind observations,"<br>
<i>Proc. CICPS 2026</i>, Springer Nature LNNS (Scopus)
</small>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()