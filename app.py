# =============================================================================
# CARDIQ™  —  AI Customer Intelligence Platform
# Enterprise Banking Analytics — K-Means Customer Segmentation Engine
# =============================================================================

import io
import time
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# PAGE CONFIG  (must be first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title="CARDIQ™ | AI Customer Intelligence Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# CONSTANTS
# =============================================================================
MODEL_DIR = "."

CLUSTER_LABELS = {
    0: "Low Activity Customer",
    1: "Frequent Shopper",
    2: "Premium Customer",
    3: "Balanced Customer",
    4: "Cash Advance User",
}

CLUSTER_ICONS = {
    0: "🌙",
    1: "🛍️",
    2: "💎",
    3: "⚖️",
    4: "💵",
}

CLUSTER_COLORS = {
    0: "#7c8aa5",
    1: "#10b981",
    2: "#d4af37",
    3: "#3b82f6",
    4: "#e2593b",
}

CLUSTER_DESCRIPTIONS = {
    0: "Minimal engagement across purchasing and credit activity. Strong candidate for re-engagement campaigns.",
    1: "High purchase frequency with consistent spending patterns. Ideal for loyalty and rewards programs.",
    2: "Elite spending power, high credit limits, and strong payment discipline. Top-tier wealth management target.",
    3: "Healthy, well-distributed financial activity with moderate risk. Reliable, steady revenue contributor.",
    4: "Heavy reliance on cash advances relative to purchases. Requires proactive risk monitoring.",
}

# Feature order exactly as produced by the notebook:
# joblib.dump(list(df.drop("Cluster", axis=1).columns), "feature_names.pkl")
FALLBACK_FEATURE_ORDER = [
    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES", "CASH_ADVANCE", "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX", "PURCHASES_TRX",
    "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE",
]

# =============================================================================
# RESOURCE LOADING  (cached — models are never re-trained, logic untouched)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_pipeline():
    kmeans = joblib.load(f"{MODEL_DIR}/kmeans_model.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    pca = joblib.load(f"{MODEL_DIR}/pca_model.pkl")
    try:
        feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
    except Exception:
        feature_names = FALLBACK_FEATURE_ORDER
    return kmeans, scaler, pca, feature_names


@st.cache_resource(show_spinner=False)
def load_dataset_artifacts():
    """Optional pre-computed dataset artifacts used purely for dashboard context
    (KPIs, distributions, PCA backdrop). Falls back gracefully if absent."""
    artifacts = {}
    for name in ["pca_df", "cluster_summary", "full_df"]:
        try:
            artifacts[name] = joblib.load(f"{MODEL_DIR}/{name}.pkl")
        except Exception:
            artifacts[name] = None
    return artifacts


try:
    kmeans_model, scaler_model, pca_model, feature_names = load_pipeline()
    MODELS_LOADED = True
except Exception as e:
    MODELS_LOADED = False
    MODEL_LOAD_ERROR = str(e)

dataset_artifacts = load_dataset_artifacts()
PCA_DF = dataset_artifacts.get("pca_df")
CLUSTER_SUMMARY = dataset_artifacts.get("cluster_summary")
FULL_DF = dataset_artifacts.get("full_df")

SILHOUETTE_SCORE = 0.1931  # reproduced from notebook: KMeans(n_clusters=5, random_state=42, n_init=10)

# =============================================================================
# SESSION STATE
# =============================================================================
defaults = {
    "page": "home",
    "theme": "royal",
    "prediction_history": [],
    "last_result": None,
    "reset_flag": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =============================================================================
# CSS  —  LUXURY BANKING INTELLIGENCE THEME
# =============================================================================
THEMES = {
    "royal": {
        "bg0": "#040610",
        "bg1": "#060a18",
        "bg2": "#0a1128",
        "panel": "rgba(13, 20, 40, 0.62)",
        "panel_solid": "#0c1226",
        "border": "rgba(212, 175, 55, 0.18)",
        "border_strong": "rgba(212, 175, 55, 0.38)",
        "navy": "#0b1330",
        "blue": "#2563eb",
        "blue2": "#1d4ed8",
        "emerald": "#10b981",
        "gold": "#d4af37",
        "gold2": "#f4d675",
        "text": "#f4f6fb",
        "text_dim": "#93a0bd",
        "text_faint": "#5c6786",
        "glow": "rgba(37, 99, 235, 0.45)",
        "glow_gold": "rgba(212, 175, 55, 0.35)",
    },
    "obsidian": {
        "bg0": "#000000",
        "bg1": "#050505",
        "bg2": "#0d0d10",
        "panel": "rgba(18, 18, 22, 0.65)",
        "panel_solid": "#0e0e12",
        "border": "rgba(16, 185, 129, 0.18)",
        "border_strong": "rgba(16, 185, 129, 0.38)",
        "navy": "#0a0a0d",
        "blue": "#3b82f6",
        "blue2": "#2563eb",
        "emerald": "#10b981",
        "gold": "#d4af37",
        "gold2": "#f4d675",
        "text": "#f5f5f7",
        "text_dim": "#9a9aa5",
        "text_faint": "#5a5a66",
        "glow": "rgba(16, 185, 129, 0.4)",
        "glow_gold": "rgba(212, 175, 55, 0.3)",
    },
}

T = THEMES[st.session_state.theme]

CSS_TEMPLATE = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,600;0,700;0,800;1,500&family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ---------------------------------------------------------------------- */
/* 0. HIDE ALL STREAMLIT CHROME                                           */
/* ---------------------------------------------------------------------- */
#MainMenu, footer, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stHeader"],
[data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
#stDecoration, .stAppToolbar, [data-testid="collapsedControl"] {
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
    display: none !important;
}
.block-container { padding-top: 2rem !important; }

/* ---------------------------------------------------------------------- */
/* 1. ROOT VARIABLES                                                      */
/* ---------------------------------------------------------------------- */
:root {
    --bg0: __BG0__;
    --bg1: __BG1__;
    --bg2: __BG2__;
    --panel: __PANEL__;
    --panel-solid: __PANEL_SOLID__;
    --border: __BORDER__;
    --border-strong: __BORDER_STRONG__;
    --navy: __NAVY__;
    --blue: __BLUE__;
    --blue2: __BLUE2__;
    --emerald: __EMERALD__;
    --gold: __GOLD__;
    --gold2: __GOLD2__;
    --text: __TEXT__;
    --text-dim: __TEXT_DIM__;
    --text-faint: __TEXT_FAINT__;
    --glow: __GLOW__;
    --glow-gold: __GLOW_GOLD__;
}

/* ---------------------------------------------------------------------- */
/* 2. GLOBAL BASE                                                         */
/* ---------------------------------------------------------------------- */
html, body, .stApp {
    background: var(--bg0) !important;
    color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(37,99,235,0.18), transparent 60%),
        radial-gradient(ellipse 70% 50% at 100% 0%, rgba(212,175,55,0.10), transparent 55%),
        radial-gradient(ellipse 60% 60% at 50% 110%, rgba(16,185,129,0.10), transparent 60%),
        linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 45%, var(--bg2) 100%);
    background-attachment: fixed;
    background-size: 200% 200%, 200% 200%, 200% 200%, 100% 100%;
    animation: bgDrift 24s ease-in-out infinite alternate;
    position: relative;
}

@keyframes bgDrift {
    0%   { background-position: 0% 0%, 100% 0%, 50% 100%, 0 0; }
    100% { background-position: 15% 10%, 85% 15%, 45% 90%, 0 0; }
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background-image:
        linear-gradient(rgba(255,255,255,0.012) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.012) 1px, transparent 1px);
    background-size: 42px 42px;
    z-index: -1;
}

[data-testid="stAppViewContainer"], .main, .block-container {
    position: relative;
    z-index: 1;
}

* { box-sizing: border-box; }

::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: var(--bg1); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--gold), var(--blue));
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover { background: var(--gold2); }

h1, h2, h3 { font-family: 'Inter', sans-serif; letter-spacing: -0.02em; }

a { color: var(--gold2); }

/* ---------------------------------------------------------------------- */
/* 3. TYPOGRAPHY UTILITIES                                                */
/* ---------------------------------------------------------------------- */
.brand-serif {
    font-family: 'Playfair Display', serif;
}

.gradient-text {
    background: linear-gradient(100deg, var(--gold2) 0%, var(--gold) 25%, #fff 50%, var(--blue) 75%, var(--emerald) 100%);
    background-size: 300% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shineText 6s linear infinite;
}
@keyframes shineText {
    0% { background-position: 0% 50%; }
    100% { background-position: 300% 50%; }
}

.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    font-size: 0.72rem;
    color: var(--gold2);
    opacity: 0.85;
}

.text-dim { color: var(--text-dim); }
.text-faint { color: var(--text-faint); }

/* ---------------------------------------------------------------------- */
/* 4. HERO / LANDING PAGE                                                 */
/* ---------------------------------------------------------------------- */
.hero-wrap {
    min-height: 88vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    position: relative;
    padding: 2rem 1rem 3rem 1rem;
}

.hero-orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(70px);
    opacity: 0.55;
    z-index: 0;
    animation: floatOrb 10s ease-in-out infinite;
}
.orb1 { width: 420px; height: 420px; background: radial-gradient(circle, var(--blue), transparent 70%); top: -8%; left: 8%; animation-delay: 0s; }
.orb2 { width: 360px; height: 360px; background: radial-gradient(circle, var(--gold), transparent 70%); bottom: -6%; right: 6%; animation-delay: 2.4s; }
.orb3 { width: 260px; height: 260px; background: radial-gradient(circle, var(--emerald), transparent 70%); top: 40%; right: 24%; animation-delay: 4.8s; }

@keyframes floatOrb {
    0%, 100% { transform: translateY(0px) translateX(0px); }
    50% { transform: translateY(-26px) translateX(14px); }
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 1.1rem;
    border-radius: 100px;
    border: 1px solid var(--border-strong);
    background: var(--panel);
    backdrop-filter: blur(18px);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gold2);
    margin-bottom: 2rem;
    z-index: 1;
    animation: fadeInUp 0.9s ease both;
}

.hero-card {
    z-index: 1;
    max-width: 900px;
    padding: 3.4rem 3rem;
    border-radius: 32px;
    background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
    border: 1px solid var(--border-strong);
    backdrop-filter: blur(28px);
    -webkit-backdrop-filter: blur(28px);
    box-shadow:
        0 40px 120px -30px rgba(0,0,0,0.75),
        0 0 0 1px rgba(255,255,255,0.02) inset,
        0 0 90px -20px var(--glow);
    animation: fadeInUp 1.1s ease both, cardFloat 7s ease-in-out infinite 1.1s;
    position: relative;
    overflow: hidden;
}

.hero-card::before {
    content: "";
    position: absolute;
    top: -2px; left: -2px; right: -2px; bottom: -2px;
    background: linear-gradient(120deg, var(--gold), transparent 30%, var(--blue) 60%, transparent 90%);
    opacity: 0.25;
    z-index: -1;
    border-radius: 32px;
    filter: blur(18px);
}

@keyframes cardFloat {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: clamp(2.6rem, 6vw, 4.4rem);
    line-height: 1.05;
    margin: 0 0 0.6rem 0;
    letter-spacing: -0.01em;
}

.hero-sub-title {
    font-family: 'Inter', sans-serif;
    font-weight: 300;
    font-size: clamp(1rem, 1.6vw, 1.35rem);
    color: var(--text-dim);
    margin-bottom: 1.6rem;
    letter-spacing: 0.01em;
}

.hero-desc {
    max-width: 620px;
    margin: 0 auto 2.4rem auto;
    color: var(--text-dim);
    font-size: 1.02rem;
    line-height: 1.75;
    font-weight: 300;
}

.hero-stats-row {
    display: flex;
    justify-content: center;
    gap: 2.6rem;
    margin-top: 2.6rem;
    flex-wrap: wrap;
    z-index: 1;
    animation: fadeInUp 1.4s ease both;
}
.hero-stat { text-align: center; }
.hero-stat .num {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--gold2), var(--gold));
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-stat .lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-faint);
    margin-top: 0.2rem;
}

@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(28px); }
    100% { opacity: 1; transform: translateY(0); }
}

/* ---------------------------------------------------------------------- */
/* 5. BUTTONS                                                             */
/* ---------------------------------------------------------------------- */
div.stButton { display: flex; justify-content: center; }

div.stButton > button {
    position: relative;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 1.02rem;
    letter-spacing: 0.02em;
    color: #05070f;
    background: linear-gradient(100deg, var(--gold2), var(--gold) 45%, var(--blue) 130%);
    background-size: 220% auto;
    border: none;
    border-radius: 14px;
    padding: 0.85rem 2.6rem;
    box-shadow: 0 12px 34px -8px var(--glow-gold), 0 0 0 1px rgba(255,255,255,0.08) inset;
    transition: all 0.35s cubic-bezier(.2,.8,.2,1);
    overflow: hidden;
}
div.stButton > button:hover {
    background-position: 100% center;
    transform: translateY(-3px) scale(1.015);
    box-shadow: 0 18px 46px -6px var(--glow-gold), 0 0 40px var(--glow);
    color: #05070f;
}
div.stButton > button:active { transform: translateY(0px) scale(0.99); }
div.stButton > button:focus { color: #05070f; }
div.stButton > button p { color: #05070f !important; font-weight: 700 !important; }

div.stButton > button::after {
    content: "";
    position: absolute;
    top: 50%; left: 50%;
    width: 0; height: 0;
    background: rgba(255,255,255,0.5);
    border-radius: 50%;
    transform: translate(-50%,-50%);
    transition: width 0.5s ease, height 0.5s ease, opacity 0.6s ease;
    opacity: 0;
}
div.stButton > button:active::after { width: 240px; height: 240px; opacity: 1; transition: 0s; }

/* secondary / ghost button variant applied via key wrapper class */
.ghost-btn-wrap div.stButton > button {
    background: var(--panel);
    color: var(--text) !important;
    border: 1px solid var(--border-strong);
    box-shadow: none;
}
.ghost-btn-wrap div.stButton > button p { color: var(--text) !important; }
.ghost-btn-wrap div.stButton > button:hover {
    border-color: var(--gold);
    box-shadow: 0 0 24px -6px var(--glow-gold);
    transform: translateY(-2px);
}

/* download buttons */
.stDownloadButton > button {
    background: linear-gradient(100deg, var(--emerald), var(--blue)) !important;
    color: #05070f !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    box-shadow: 0 10px 26px -8px var(--glow) !important;
}
.stDownloadButton > button p { color: #05070f !important; }

/* ---------------------------------------------------------------------- */
/* 6. GLASS CARDS / PANELS                                                */
/* ---------------------------------------------------------------------- */
.glass-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1.6rem 1.7rem;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 24px 60px -24px rgba(0,0,0,0.65);
    position: relative;
    animation: fadeInUp 0.7s ease both;
    transition: border-color 0.3s ease, transform 0.3s ease;
}
.glass-card:hover { border-color: var(--border-strong); }

.glass-card-tight { padding: 1.1rem 1.3rem; }

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
    display: flex;
    align-items: center;
    gap: 0.55rem;
}
.section-sub { color: var(--text-dim); font-size: 0.9rem; margin-bottom: 1.4rem; font-weight: 300; }

.divider-gold {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    opacity: 0.5;
    margin: 1.4rem 0;
    border: none;
}

/* ---------------------------------------------------------------------- */
/* 7. TOP NAV BAR                                                         */
/* ---------------------------------------------------------------------- */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.9rem 1.6rem;
    border-radius: 20px;
    background: var(--panel);
    border: 1px solid var(--border);
    backdrop-filter: blur(24px);
    margin-bottom: 1.6rem;
    box-shadow: 0 20px 50px -28px rgba(0,0,0,0.7);
    animation: fadeInUp 0.6s ease both;
}
.nav-logo {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: 1.4rem;
    letter-spacing: 0.01em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.nav-logo .tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    color: var(--text-faint);
    text-transform: uppercase;
    margin-left: 0.4rem;
    border-left: 1px solid var(--border-strong);
    padding-left: 0.6rem;
}
.nav-live {
    display: flex; align-items: center; gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: var(--emerald);
    letter-spacing: 0.08em;
}
.pulse-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--emerald);
    box-shadow: 0 0 0 0 var(--emerald);
    animation: pulseDot 1.8s infinite;
}
@keyframes pulseDot {
    0% { box-shadow: 0 0 0 0 rgba(16,185,129,0.55); }
    70% { box-shadow: 0 0 0 9px rgba(16,185,129,0); }
    100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
}

/* ---------------------------------------------------------------------- */
/* 8. KPI CARDS                                                           */
/* ---------------------------------------------------------------------- */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 1rem;
    margin-bottom: 1.8rem;
}
@media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 620px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }

.kpi-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.15rem 1.1rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(18px);
    transition: transform 0.35s ease, border-color 0.35s ease, box-shadow 0.35s ease;
    animation: fadeInUp 0.8s ease both;
}
.kpi-card:hover {
    transform: translateY(-6px);
    border-color: var(--border-strong);
    box-shadow: 0 20px 46px -18px var(--glow);
}
.kpi-card::after {
    content: "";
    position: absolute;
    top: -40%; left: -20%;
    width: 60%; height: 180%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,0.05), transparent);
    transform: rotate(20deg);
    animation: shimmerSweep 5s ease-in-out infinite;
}
@keyframes shimmerSweep {
    0% { left: -40%; }
    50% { left: 120%; }
    100% { left: 120%; }
}
.kpi-icon { font-size: 1.3rem; opacity: 0.9; margin-bottom: 0.4rem; }
.kpi-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
}
.kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-faint);
    margin-top: 0.35rem;
}
.kpi-delta { font-size: 0.72rem; margin-top: 0.35rem; font-weight: 600; }
.kpi-delta.up { color: var(--emerald); }
.kpi-delta.down { color: #e2593b; }

/* ---------------------------------------------------------------------- */
/* 9. INPUT PANEL — sliders, number inputs, selects                       */
/* ---------------------------------------------------------------------- */
.input-group-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gold2);
    margin: 1.1rem 0 0.15rem 0;
    display: flex; align-items: center; gap: 0.4rem;
}

.stSlider label, .stNumberInput label, .stSelectbox label, .stTextInput label {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.86rem !important;
    color: var(--text-dim) !important;
}

div[data-baseweb="slider"] > div > div { background: rgba(255,255,255,0.08) !important; }
div[data-baseweb="slider"] div[role="slider"] {
    background: linear-gradient(135deg, var(--gold2), var(--blue)) !important;
    border: 3px solid var(--bg0) !important;
    box-shadow: 0 0 14px var(--glow-gold) !important;
}
div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"] {
    color: var(--text-faint) !important; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem !important;
}
.stSlider [data-baseweb="slider"] > div:nth-child(2) > div {
    background: linear-gradient(90deg, var(--blue), var(--gold)) !important;
}

.stNumberInput input, .stTextInput input {
    background: rgba(255,255,255,0.035) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px var(--glow-gold) !important;
}
.stNumberInput button { background: rgba(255,255,255,0.04) !important; border-color: var(--border) !important; }

div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.035) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* checkbox / toggle */
.stCheckbox label span { color: var(--text-dim) !important; }

/* ---------------------------------------------------------------------- */
/* 10. TABS                                                               */
/* ---------------------------------------------------------------------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background: var(--panel);
    padding: 0.4rem;
    border-radius: 16px;
    border: 1px solid var(--border);
    backdrop-filter: blur(18px);
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    border-radius: 11px;
    color: var(--text-dim);
    font-weight: 500;
    font-size: 0.9rem;
    background: transparent;
    transition: all 0.25s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text); background: rgba(255,255,255,0.03); }
.stTabs [aria-selected="true"] {
    background: linear-gradient(100deg, var(--gold2), var(--gold)) !important;
    color: #05070f !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 24px -8px var(--glow-gold);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------------------------------------------------------------------- */
/* 11. EXPANDER                                                           */
/* ---------------------------------------------------------------------- */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-weight: 500 !important;
}
[data-testid="stExpander"] {
    border: none !important;
    background: transparent !important;
}
[data-testid="stExpanderDetails"] {
    background: rgba(255,255,255,0.015);
    border-radius: 0 0 12px 12px;
    border: 1px solid var(--border);
    border-top: none;
    padding: 0.6rem 0.9rem 0.9rem 0.9rem !important;
}

/* ---------------------------------------------------------------------- */
/* 12. DATAFRAME / TABLE                                                  */
/* ---------------------------------------------------------------------- */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    overflow: hidden;
}

/* ---------------------------------------------------------------------- */
/* 13. PROGRESS BAR                                                       */
/* ---------------------------------------------------------------------- */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--blue), var(--gold), var(--emerald)) !important;
    background-size: 200% auto;
    animation: shineText 2.4s linear infinite;
}
.stProgress > div > div { background: rgba(255,255,255,0.06) !important; border-radius: 10px; }

/* ---------------------------------------------------------------------- */
/* 14. BADGES / PILLS                                                     */
/* ---------------------------------------------------------------------- */
.segment-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.55rem 1.4rem;
    border-radius: 100px;
    font-weight: 700;
    font-size: 1rem;
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 0 30px -6px var(--glow-gold);
    animation: badgeGlow 2.6s ease-in-out infinite;
}
@keyframes badgeGlow {
    0%, 100% { box-shadow: 0 0 22px -8px var(--glow-gold); }
    50% { box-shadow: 0 0 36px -6px var(--glow-gold); }
}

.rec-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1rem;
    border-radius: 14px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    font-size: 0.87rem;
    margin: 0.28rem 0.35rem 0.28rem 0;
    transition: all 0.25s ease;
}
.rec-chip:hover {
    border-color: var(--gold);
    background: rgba(212,175,55,0.06);
    transform: translateX(4px);
}

/* ---------------------------------------------------------------------- */
/* 15. LOADING SEQUENCE                                                   */
/* ---------------------------------------------------------------------- */
.load-step {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.85rem 1.2rem;
    border-radius: 14px;
    background: var(--panel);
    border: 1px solid var(--border-strong);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.92rem;
    margin-bottom: 0.6rem;
    animation: fadeInUp 0.4s ease both, shimmerBg 1.4s ease-in-out infinite;
    background-image: linear-gradient(100deg, transparent 30%, rgba(212,175,55,0.10) 50%, transparent 70%);
    background-size: 200% 100%;
}
@keyframes shimmerBg {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ---------------------------------------------------------------------- */
/* 16. FOOTER                                                             */
/* ---------------------------------------------------------------------- */
.cardiq-footer {
    text-align: center;
    padding: 2.2rem 1rem 1.4rem 1rem;
    color: var(--text-faint);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    border-top: 1px solid var(--border);
    margin-top: 2.4rem;
}

/* ---------------------------------------------------------------------- */
/* 17. METRIC OVERRIDE (native st.metric, used sparingly)                 */
/* ---------------------------------------------------------------------- */
[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0.9rem 1.1rem;
    backdrop-filter: blur(16px);
}
[data-testid="stMetricLabel"] { color: var(--text-faint) !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Playfair Display', serif !important; }

/* ---------------------------------------------------------------------- */
/* 18. MISC                                                               */
/* ---------------------------------------------------------------------- */
.block-container { max-width: 1360px; }
hr { border-color: var(--border) !important; }

.small-caps {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-faint);
}

.stAlert { border-radius: 14px !important; }

.floaty { animation: floatSlow 6s ease-in-out infinite; }
@keyframes floatSlow {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
}
"""


def inject_css():
    css = CSS_TEMPLATE
    for key, val in [
        ("BG0", T["bg0"]), ("BG1", T["bg1"]), ("BG2", T["bg2"]),
        ("PANEL", T["panel"]), ("PANEL_SOLID", T["panel_solid"]),
        ("BORDER", T["border"]), ("BORDER_STRONG", T["border_strong"]),
        ("NAVY", T["navy"]), ("BLUE", T["blue"]), ("BLUE2", T["blue2"]),
        ("EMERALD", T["emerald"]), ("GOLD", T["gold"]), ("GOLD2", T["gold2"]),
        ("TEXT", T["text"]), ("TEXT_DIM", T["text_dim"]), ("TEXT_FAINT", T["text_faint"]),
        ("GLOW", T["glow"]), ("GLOW_GOLD", T["glow_gold"]),
    ]:
        css = css.replace(f"__{key}__", val)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


inject_css()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def pct(v, lo, hi):
    if hi == lo:
        return 50.0
    return float(np.clip((v - lo) / (hi - lo) * 100, 0, 100))


def compute_scores(inputs: dict) -> dict:
    """Presentation-layer scoring heuristics (not part of the ML pipeline)."""
    balance = inputs["BALANCE"]
    credit_limit = max(inputs["CREDIT_LIMIT"], 1)
    cash_adv_freq = inputs["CASH_ADVANCE_FREQUENCY"]
    full_payment = inputs["PRC_FULL_PAYMENT"]
    purchases = inputs["PURCHASES"]
    payments = inputs["PAYMENTS"]
    tenure = inputs["TENURE"]
    balance_freq = inputs["BALANCE_FREQUENCY"]
    purchase_freq = inputs["PURCHASES_FREQUENCY"]
    min_payments = inputs["MINIMUM_PAYMENTS"]

    credit_utilization = float(np.clip((balance / credit_limit) * 100, 0, 100))

    risk_score = float(np.clip(
        cash_adv_freq * 45 +
        (1 - full_payment) * 30 +
        credit_utilization * 0.25, 0, 100
    ))

    value_score = float(np.clip(
        pct(purchases, 0, 20000) * 0.35 +
        pct(payments, 0, 20000) * 0.30 +
        pct(credit_limit, 50, 30000) * 0.20 +
        pct(tenure, 6, 12) * 0.15, 0, 100
    ))

    loyalty_index = float(np.clip(
        pct(tenure, 6, 12) * 0.6 + balance_freq * 40, 0, 100
    ))

    spending_level = float(np.clip(pct(purchases, 0, 15000), 0, 100))

    financial_health = float(np.clip(
        100 - risk_score * 0.6 + full_payment * 25 - (min_payments / max(payments, 1)) * 10,
        0, 100
    ))

    return {
        "credit_utilization": credit_utilization,
        "risk_score": risk_score,
        "value_score": value_score,
        "loyalty_index": loyalty_index,
        "spending_level": spending_level,
        "financial_health": financial_health,
    }


def generate_recommendations(cluster_id: int, scores: dict, inputs: dict) -> list:
    recs = []
    if cluster_id == 2:
        recs += ["💎 Offer Platinum / Signature Credit Card", "📈 Priority Wealth Management Outreach", "🏆 Eligible for Premium Rewards Tier"]
    if cluster_id == 1:
        recs += ["🎯 Target for Cashback Campaign", "🛍️ Enroll in Merchant Partner Offers", "🔁 Recurring Rewards Multiplier"]
    if cluster_id == 4:
        recs += ["⚠️ Flag for Risk Monitoring", "💳 Recommend Structured Repayment Plan", "🚫 Cap Further Cash Advance Limit Increases"]
    if cluster_id == 3:
        recs += ["📊 Balanced Growth Nurture Track", "🏦 Cross-sell Insurance Products", "💰 Present Fixed Deposit Opportunity"]
    if cluster_id == 0:
        recs += ["📩 Re-engagement Email Sequence", "🎁 Activation Bonus for First Purchase", "📞 Proactive Relationship Manager Call"]

    if scores["risk_score"] < 30:
        recs.append("✅ Low Risk Customer — Fast-track Approvals")
    if scores["value_score"] > 70:
        recs.append("📈 Investment Opportunity — Present Advisory Services")
    if scores["credit_utilization"] > 75:
        recs.append("🔺 Consider Increasing Credit Limit")
    if inputs["CASH_ADVANCE"] < 1 and inputs["PURCHASES"] > 3000:
        recs.append("🌟 Strong Organic Spender — No Cash Dependency")
    if scores["loyalty_index"] > 70:
        recs.append("🤝 Potential Loan Customer — High Retention Confidence")

    seen, unique = set(), []
    for r in recs:
        if r not in seen:
            unique.append(r)
            seen.add(r)
    return unique[:8]


def gauge_fig(value, title, color, suffix="", ref=None):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"size": 30, "color": T["text"], "family": "Playfair Display"}},
        title={"text": title, "font": {"size": 13, "color": T["text_dim"], "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": T["text_faint"], "tickfont": {"size": 9, "color": T["text_faint"]}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 1,
            "bordercolor": T["border"],
            "steps": [
                {"range": [0, 33], "color": "rgba(255,255,255,0.02)"},
                {"range": [33, 66], "color": "rgba(255,255,255,0.045)"},
                {"range": [66, 100], "color": "rgba(255,255,255,0.07)"},
            ],
            "threshold": {"line": {"color": T["gold2"], "width": 3}, "thickness": 0.8, "value": ref if ref else value},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=24, r=24, t=48, b=12),
        font={"color": T["text"]},
    )
    return fig


def plotly_dark_layout(fig, height=420, legend=True):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["text_dim"], family="Inter"),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)") if legend else None,
        showlegend=legend,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)")
    return fig


def kpi_cards_html(items):
    cards = ""
    for icon, value, label, color in items:
        cards += f"""
        <div class="kpi-card" style="border-top:2px solid {color};">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>"""
    return f'<div class="kpi-grid">{cards}</div>'


# =============================================================================
# DATA-DRIVEN CONSTANTS FOR OVERVIEW (fallback to static values if artifacts absent)
# =============================================================================
if FULL_DF is not None:
    TOTAL_CUSTOMERS = int(len(FULL_DF))
    AVG_CREDIT_LIMIT = float(FULL_DF["CREDIT_LIMIT"].mean())
    AVG_BALANCE = float(FULL_DF["BALANCE"].mean())
    AVG_PURCHASES = float(FULL_DF["PURCHASES"].mean())
else:
    TOTAL_CUSTOMERS, AVG_CREDIT_LIMIT, AVG_BALANCE, AVG_PURCHASES = 8950, 4494.0, 1564.0, 1003.0

N_CLUSTERS = 5

# =============================================================================
# PAGE: HOME
# =============================================================================

def render_home():
    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-orb orb1"></div>
            <div class="hero-orb orb2"></div>
            <div class="hero-orb orb3"></div>
            <div class="hero-badge">⚡ Enterprise Banking Intelligence Suite</div>
            <div class="hero-card">
                <div class="eyebrow">K-MEANS CUSTOMER SEGMENTATION ENGINE</div>
                <h1 class="hero-title gradient-text">💳 CARDIQ™</h1>
                <div class="hero-sub-title brand-serif">AI Customer Intelligence Platform</div>
                <p class="hero-desc">
                    Leverage artificial intelligence and K-Means clustering to identify customer
                    segments, spending behavior, and premium banking opportunities — with the
                    precision institutions like JPMorgan, Goldman Sachs and American Express demand.
                </p>
                <div class="hero-stats-row">
                    <div class="hero-stat"><div class="num">{TOTAL_CUSTOMERS:,}</div><div class="lbl">Customers Profiled</div></div>
                    <div class="hero-stat"><div class="num">{N_CLUSTERS}</div><div class="lbl">AI Segments</div></div>
                    <div class="hero-stat"><div class="num">{SILHOUETTE_SCORE:.3f}</div><div class="lbl">Silhouette Score</div></div>
                    <div class="hero-stat"><div class="num">17</div><div class="lbl">Behavioral Signals</div></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("✨  Start Customer Analysis", key="start_btn", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()

    st.markdown(
        f"""
        <div class="cardiq-footer">
            CARDIQ™ © {datetime.now().year} &nbsp;·&nbsp; AI Customer Intelligence Platform
            &nbsp;·&nbsp; Powered by K-Means Clustering &amp; PCA &nbsp;·&nbsp; For Institutional Use
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE: DASHBOARD
# =============================================================================

def render_navbar():
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            """
            <div class="nav-logo">💳 CARDIQ™ <span class="tag">Banking Analytics Platform</span></div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.markdown('<div class="ghost-btn-wrap">', unsafe_allow_html=True)
            if st.button("🏠 Home", key="nav_home"):
                st.session_state.page = "home"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="ghost-btn-wrap">', unsafe_allow_html=True)
            theme_label = "🌓 Obsidian" if st.session_state.theme == "royal" else "🌓 Royal Navy"
            if st.button(theme_label, key="nav_theme"):
                st.session_state.theme = "obsidian" if st.session_state.theme == "royal" else "royal"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="ghost-btn-wrap">', unsafe_allow_html=True)
            if st.button("♻️ Reset", key="nav_reset"):
                st.session_state.reset_flag += 1
                st.session_state.last_result = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


def render_kpis():
    items = [
        ("👥", f"{TOTAL_CUSTOMERS:,}", "Total Customers", T["blue"]),
        ("🧩", f"{N_CLUSTERS}", "Clusters Created", T["gold"]),
        ("📐", f"{SILHOUETTE_SCORE:.3f}", "Silhouette Score", T["emerald"]),
        ("💳", f"${AVG_CREDIT_LIMIT:,.0f}", "Avg Credit Limit", T["blue"]),
        ("💰", f"${AVG_BALANCE:,.0f}", "Avg Balance", T["gold"]),
        ("🛒", f"${AVG_PURCHASES:,.0f}", "Avg Purchases", T["emerald"]),
    ]
    st.markdown(kpi_cards_html(items), unsafe_allow_html=True)


FEATURE_FIELDS_PRIMARY = [
    ("BALANCE", "Balance", 0.0, 20000.0, 1500.0, 50.0, "$", "Current outstanding balance"),
    ("BALANCE_FREQUENCY", "Balance Frequency", 0.0, 1.0, 0.88, 0.01, "", "How frequently the balance is updated (0–1)"),
    ("PURCHASES", "Purchases", 0.0, 20000.0, 1000.0, 50.0, "$", "Total purchase amount"),
    ("ONEOFF_PURCHASES", "One-off Purchases", 0.0, 15000.0, 600.0, 50.0, "$", "Maximum purchase amount in one go"),
    ("INSTALLMENTS_PURCHASES", "Installment Purchases", 0.0, 8000.0, 400.0, 50.0, "$", "Purchases made in installments"),
    ("CASH_ADVANCE", "Cash Advance", 0.0, 15000.0, 0.0, 50.0, "$", "Cash advanced by the customer"),
    ("PURCHASES_FREQUENCY", "Purchase Frequency", 0.0, 1.0, 0.50, 0.01, "", "How frequently purchases are made (0–1)"),
    ("CASH_ADVANCE_FREQUENCY", "Cash Advance Frequency", 0.0, 1.0, 0.13, 0.01, "", "How frequently cash advances are taken (0–1)"),
    ("CREDIT_LIMIT", "Credit Limit", 50.0, 30000.0, 4500.0, 50.0, "$", "Credit limit of the customer"),
    ("PAYMENTS", "Payments", 0.0, 20000.0, 1700.0, 50.0, "$", "Total payments made by customer"),
    ("MINIMUM_PAYMENTS", "Minimum Payments", 0.0, 10000.0, 850.0, 50.0, "$", "Minimum payments made by customer"),
    ("PRC_FULL_PAYMENT", "Full Payment %", 0.0, 1.0, 0.15, 0.01, "", "Percent of full payment paid"),
    ("TENURE", "Tenure (months)", 6.0, 12.0, 12.0, 1.0, "", "Tenure of credit card service"),
]

FEATURE_FIELDS_ADVANCED = [
    ("ONEOFF_PURCHASES_FREQUENCY", "One-off Purchase Frequency", 0.0, 1.0, 0.20, 0.01, "", "Frequency of one-off purchases (0–1)"),
    ("PURCHASES_INSTALLMENTS_FREQUENCY", "Installment Purchase Frequency", 0.0, 1.0, 0.36, 0.01, "", "Frequency of installment purchases (0–1)"),
    ("CASH_ADVANCE_TRX", "Cash Advance Transactions", 0, 50, 0, 1, "", "Number of cash advance transactions"),
    ("PURCHASES_TRX", "Purchase Transactions", 0, 150, 15, 1, "", "Number of purchase transactions"),
]


def render_input_panel():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧾 Luxury Customer Input Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter the customer\'s financial profile to run AI segmentation</div>', unsafe_allow_html=True)

    values = {}
    st.markdown('<div class="input-group-label">💰 Core Financial Profile</div>', unsafe_allow_html=True)
    for key, label, lo, hi, default, step, prefix, help_txt in FEATURE_FIELDS_PRIMARY:
        wkey = f"inp_{key}_{st.session_state.reset_flag}"
        values[key] = st.slider(
            f"{label}", min_value=float(lo), max_value=float(hi),
            value=float(default), step=float(step), key=wkey, help=help_txt,
        )

    with st.expander("⚙️ Advanced Transaction Details"):
        for key, label, lo, hi, default, step, prefix, help_txt in FEATURE_FIELDS_ADVANCED:
            wkey = f"inp_{key}_{st.session_state.reset_flag}"
            if isinstance(default, int):
                values[key] = st.number_input(label, min_value=int(lo), max_value=int(hi), value=int(default), step=int(step), key=wkey, help=help_txt)
            else:
                values[key] = st.slider(label, min_value=float(lo), max_value=float(hi), value=float(default), step=float(step), key=wkey, help=help_txt)

    st.markdown('<div class="divider-gold"></div>', unsafe_allow_html=True)
    predict_clicked = st.button("🚀  Analyze Customer", key="predict_btn", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    return values, predict_clicked


def run_loading_sequence():
    steps = [
        "🔍 Reading Financial Profile...",
        "📈 Scaling Customer Features...",
        "🧠 Running K-Means Clustering...",
        "📊 Comparing with Similar Customers...",
        "💎 Identifying Customer Segment...",
        "✨ Generating Business Insights...",
    ]
    ph = st.empty()
    bar = st.progress(0)
    for i, s in enumerate(steps):
        ph.markdown(f'<div class="load-step">{s}</div>', unsafe_allow_html=True)
        bar.progress(int((i + 1) / len(steps) * 100))
        time.sleep(0.35)
    time.sleep(0.15)
    ph.empty()
    bar.empty()


def predict_customer(values: dict):
    ordered = [values[f] for f in feature_names]
    sample = pd.DataFrame([ordered], columns=feature_names)
    scaled = scaler_model.transform(sample)
    cluster_id = int(kmeans_model.predict(scaled)[0])
    pca_coords = pca_model.transform(scaled)[0]
    return cluster_id, pca_coords, scaled


def render_result(cluster_id, pca_coords, values):
    label = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
    icon = CLUSTER_ICONS.get(cluster_id, "🔷")
    color = CLUSTER_COLORS.get(cluster_id, T["gold"])
    desc = CLUSTER_DESCRIPTIONS.get(cluster_id, "")
    scores = compute_scores(values)

    st.markdown('<div class="glass-card floaty">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="section-title">🎯 Segmentation Result</div>
        <div style="margin: 0.6rem 0 1rem 0;">
            <span class="segment-badge" style="background:{color}22; color:{color}; border-color:{color}66;">
                {icon} &nbsp; {label}
            </span>
        </div>
        <p class="text-dim" style="max-width:760px; line-height:1.7;">{desc}</p>
        <div class="small-caps" style="margin-top:0.6rem;">Cluster #{cluster_id} &nbsp;·&nbsp; PCA Coordinates: ({pca_coords[0]:.2f}, {pca_coords[1]:.2f})</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(gauge_fig(scores["risk_score"], "RISK SCORE", "#e2593b"), use_container_width=True, config={"displayModeBar": False})
    with g2:
        st.plotly_chart(gauge_fig(scores["value_score"], "CUSTOMER VALUE", T["gold"]), use_container_width=True, config={"displayModeBar": False})
    with g3:
        st.plotly_chart(gauge_fig(scores["credit_utilization"], "CREDIT UTILIZATION", T["blue"], suffix="%"), use_container_width=True, config={"displayModeBar": False})

    g4, g5, g6 = st.columns(3)
    with g4:
        st.plotly_chart(gauge_fig(scores["loyalty_index"], "LOYALTY INDEX", T["emerald"]), use_container_width=True, config={"displayModeBar": False})
    with g5:
        st.plotly_chart(gauge_fig(scores["spending_level"], "SPENDING LEVEL", T["gold2"]), use_container_width=True, config={"displayModeBar": False})
    with g6:
        st.plotly_chart(gauge_fig(scores["financial_health"], "FINANCIAL HEALTH", T["emerald"]), use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 AI Banking Recommendations</div>', unsafe_allow_html=True)
    recs = generate_recommendations(cluster_id, scores, values)
    chips_html = "".join([f'<span class="rec-chip">{r}</span>' for r in recs])
    st.markdown(chips_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    return scores, label


def render_pca_scatter(pca_coords=None):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧭 PCA Cluster Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">2D projection of all customer segments</div>', unsafe_allow_html=True)

    if PCA_DF is not None:
        fig = go.Figure()
        for cid in sorted(PCA_DF["Cluster"].unique()):
            sub = PCA_DF[PCA_DF["Cluster"] == cid]
            fig.add_trace(go.Scattergl(
                x=sub["PCA1"], y=sub["PCA2"], mode="markers",
                name=f"{CLUSTER_ICONS.get(cid,'')} {CLUSTER_LABELS.get(cid, cid)}",
                marker=dict(size=5, color=CLUSTER_COLORS.get(cid, "#888"), opacity=0.55, line=dict(width=0)),
            ))
        if pca_coords is not None:
            fig.add_trace(go.Scatter(
                x=[pca_coords[0]], y=[pca_coords[1]], mode="markers",
                name="🎯 This Customer",
                marker=dict(size=20, color=T["text"], symbol="star", line=dict(width=2, color=T["gold2"])),
            ))
        fig.update_layout(xaxis_title="PCA 1", yaxis_title="PCA 2")
        fig = plotly_dark_layout(fig, height=460)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("PCA dataset backdrop unavailable — showing prediction point only.")
    st.markdown('</div>', unsafe_allow_html=True)


def render_analytics_tab():
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🍩 Cluster Distribution</div>', unsafe_allow_html=True)
        if FULL_DF is not None:
            counts = FULL_DF["Cluster"].value_counts().sort_index()
            fig = go.Figure(go.Pie(
                labels=[f"{CLUSTER_ICONS.get(i,'')} {CLUSTER_LABELS.get(i,i)}" for i in counts.index],
                values=counts.values, hole=0.62,
                marker=dict(colors=[CLUSTER_COLORS.get(i, "#888") for i in counts.index], line=dict(color=T["bg0"], width=2)),
                textfont=dict(color=T["text"]),
            ))
            fig = plotly_dark_layout(fig, height=380)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Dataset artifacts unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Customers per Cluster</div>', unsafe_allow_html=True)
        if FULL_DF is not None:
            counts = FULL_DF["Cluster"].value_counts().sort_index()
            fig = go.Figure(go.Bar(
                x=[CLUSTER_LABELS.get(i, i) for i in counts.index], y=counts.values,
                marker=dict(color=[CLUSTER_COLORS.get(i, "#888") for i in counts.index]),
                text=counts.values, textposition="outside",
            ))
            fig = plotly_dark_layout(fig, height=380, legend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Dataset artifacts unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🕸️ Cluster Radar Profile</div>', unsafe_allow_html=True)
        if CLUSTER_SUMMARY is not None:
            radar_features = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "PRC_FULL_PAYMENT"]
            norm = CLUSTER_SUMMARY[radar_features].copy()
            norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9) * 100
            fig = go.Figure()
            for cid in norm.index:
                fig.add_trace(go.Scatterpolar(
                    r=norm.loc[cid].values.tolist() + [norm.loc[cid].values[0]],
                    theta=radar_features + [radar_features[0]],
                    fill="toself", name=f"{CLUSTER_LABELS.get(cid,cid)}",
                    line=dict(color=CLUSTER_COLORS.get(cid, "#888")), opacity=0.55,
                ))
            fig.update_layout(polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.08)", color=T["text_dim"]),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", color=T["text_dim"]),
            ))
            fig = plotly_dark_layout(fig, height=420)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Cluster summary unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔥 Feature Heatmap by Cluster</div>', unsafe_allow_html=True)
        if CLUSTER_SUMMARY is not None:
            heat_features = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "TENURE"]
            heat_df = CLUSTER_SUMMARY[heat_features]
            heat_norm = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min() + 1e-9)
            fig = go.Figure(go.Heatmap(
                z=heat_norm.values, x=heat_features,
                y=[CLUSTER_LABELS.get(i, i) for i in heat_df.index],
                colorscale=[[0, T["navy"]], [0.5, T["blue"]], [1, T["gold2"]]],
                showscale=True,
            ))
            fig = plotly_dark_layout(fig, height=420, legend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Cluster summary unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    col5, col6 = st.columns(2)

    with col5:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎼 Parallel Coordinates</div>', unsafe_allow_html=True)
        if FULL_DF is not None:
            sample = FULL_DF.sample(min(600, len(FULL_DF)), random_state=42)
            pc_features = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "TENURE"]
            fig = go.Figure(go.Parcoords(
                line=dict(color=sample["Cluster"], colorscale=[[0, "#7c8aa5"], [0.25, "#10b981"], [0.5, "#d4af37"], [0.75, "#3b82f6"], [1, "#e2593b"]]),
                dimensions=[dict(label=f, values=sample[f]) for f in pc_features],
            ))
            fig = plotly_dark_layout(fig, height=400, legend=False)
            fig.update_layout(font=dict(color=T["text_dim"], size=10))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Dataset artifacts unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col6:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧮 PCA Feature Importance</div>', unsafe_allow_html=True)
        try:
            loadings = pd.Series(np.abs(pca_model.components_[0]), index=feature_names).sort_values(ascending=True)
            fig = go.Figure(go.Bar(
                x=loadings.values, y=loadings.index, orientation="h",
                marker=dict(color=loadings.values, colorscale=[[0, T["blue"]], [1, T["gold2"]]]),
            ))
            fig = plotly_dark_layout(fig, height=420, legend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception:
            st.info("PCA model unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)

    render_pca_scatter(st.session_state.last_result["pca_coords"] if st.session_state.last_result else None)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏦 Bank Performance Metrics</div>', unsafe_allow_html=True)
    if FULL_DF is not None:
        total_portfolio = FULL_DF["PAYMENTS"].sum()
        avg_util = (FULL_DF["BALANCE"] / FULL_DF["CREDIT_LIMIT"].replace(0, np.nan)).mean() * 100
        high_risk_pct = (FULL_DF["CASH_ADVANCE_FREQUENCY"] > 0.3).mean() * 100
        avg_tenure = FULL_DF["TENURE"].mean()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("💼 Total Portfolio Value", f"${total_portfolio:,.0f}")
        m2.metric("📉 Avg Credit Utilization", f"{avg_util:.1f}%")
        m3.metric("⚠️ High-Risk Customers", f"{high_risk_pct:.1f}%")
        m4.metric("📅 Avg Customer Tenure", f"{avg_tenure:.1f} mo")
    st.markdown('</div>', unsafe_allow_html=True)


def render_history_tab():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🕓 Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">All customer analyses run during this session</div>', unsafe_allow_html=True)

    history = st.session_state.prediction_history
    if not history:
        st.info("No predictions yet. Run a customer analysis in the Prediction tab to populate history.")
    else:
        hist_df = pd.DataFrame(history)
        st.dataframe(hist_df, use_container_width=True, height=320)

        c1, c2, c3 = st.columns(3)
        with c1:
            csv_buf = io.StringIO()
            hist_df.to_csv(csv_buf, index=False)
            st.download_button("⬇️ Export CSV", data=csv_buf.getvalue(), file_name="cardiq_prediction_history.csv", mime="text/csv", use_container_width=True)
        with c2:
            last = history[-1]
            report = (
                f"CARDIQ (TM) — AI CUSTOMER INTELLIGENCE PLATFORM\n"
                f"CUSTOMER SEGMENTATION REPORT\n"
                f"{'='*52}\n"
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"Segment: {last.get('Segment')}\n"
                f"Cluster ID: {last.get('Cluster')}\n"
                f"Risk Score: {last.get('Risk Score')}\n"
                f"Customer Value Score: {last.get('Value Score')}\n"
                f"Credit Utilization: {last.get('Credit Utilization')}\n"
                f"Financial Health: {last.get('Financial Health')}\n\n"
                f"{'-'*52}\n"
                f"This report is generated by an AI segmentation model\n"
                f"(K-Means, n_clusters=5) trained on customer transaction\n"
                f"behavior. For internal analytics use only.\n"
            )
            st.download_button("📄 Download Report", data=report, file_name="cardiq_report.txt", mime="text/plain", use_container_width=True)
        with c3:
            st.markdown('<div class="ghost-btn-wrap">', unsafe_allow_html=True)
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_dashboard():
    render_navbar()

    if not MODELS_LOADED:
        st.error(
            "⚠️ Model artifacts not found. Please ensure `kmeans_model.pkl`, `scaler.pkl`, "
            "`pca_model.pkl`, and `feature_names.pkl` are present in the same folder as `app.py`."
        )
        st.code(globals().get("MODEL_LOAD_ERROR", "Unknown error"))
        return

    render_kpis()

    tab1, tab2, tab3 = st.tabs(["🧠 Prediction", "📊 Analytics", "🕓 History"])

    with tab1:
        left, right = st.columns([1, 1.35], gap="large")
        with left:
            values, predict_clicked = render_input_panel()

        with right:
            if predict_clicked:
                with st.container():
                    run_loading_sequence()
                cluster_id, pca_coords, _ = predict_customer(values)
                scores, label = render_result(cluster_id, pca_coords, values)

                st.session_state.last_result = {
                    "cluster_id": cluster_id, "pca_coords": pca_coords,
                    "values": values, "scores": scores, "label": label,
                }
                st.session_state.prediction_history.append({
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Segment": label,
                    "Cluster": cluster_id,
                    "Balance": round(values["BALANCE"], 2),
                    "Purchases": round(values["PURCHASES"], 2),
                    "Credit Limit": round(values["CREDIT_LIMIT"], 2),
                    "Risk Score": round(scores["risk_score"], 1),
                    "Value Score": round(scores["value_score"], 1),
                    "Credit Utilization": round(scores["credit_utilization"], 1),
                    "Financial Health": round(scores["financial_health"], 1),
                })
            elif st.session_state.last_result:
                r = st.session_state.last_result
                render_result(r["cluster_id"], r["pca_coords"], r["values"])
            else:
                st.markdown(
                    """
                    <div class="glass-card" style="text-align:center; padding:4rem 2rem;">
                        <div style="font-size:2.6rem; margin-bottom:0.8rem;">🧠</div>
                        <div class="section-title" style="justify-content:center;">Awaiting Customer Data</div>
                        <p class="text-dim">Configure the customer profile on the left and click
                        <b>Analyze Customer</b> to run the AI segmentation engine.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with tab2:
        render_analytics_tab()

    with tab3:
        render_history_tab()

    st.markdown(
        f"""
        <div class="cardiq-footer">
            CARDIQ™ © {datetime.now().year} &nbsp;·&nbsp; AI Customer Intelligence Platform
            &nbsp;·&nbsp; Silhouette Score {SILHOUETTE_SCORE:.3f} &nbsp;·&nbsp; Model: K-Means (k=5) + PCA
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# ROUTER
# =============================================================================
try:
    if st.session_state.page == "home":
        render_home()
    else:
        render_dashboard()
except Exception as e:
    st.error("⚠️ CARDIQ hit an unexpected error while rendering this page.")
    st.exception(e)
    st.markdown('<div class="ghost-btn-wrap">', unsafe_allow_html=True)
    if st.button("🏠 Return Home"):
        st.session_state.page = "home"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)