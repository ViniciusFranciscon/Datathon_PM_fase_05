# ---> app datathon passos mágicos - diagnóstico preditivo <---

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import textwrap
import math
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go
import streamlit.components.v1 as st_components


st.set_page_config(
    page_title="Passos Mágicos | Diagnóstico Preditivo",
    layout="wide",
    page_icon="🎓",
    initial_sidebar_state="expanded",
)

def html(s: str) -> None:
    s = textwrap.dedent(s).strip()
    if hasattr(st, "html"):
        st.html(s)
    else:
        st.markdown(s, unsafe_allow_html=True)

@st.cache_resource
def load_artefatos():
    modelo = joblib.load("pkl/modelo.pkl")
    scaler_model = joblib.load("pkl/scaler.pkl")
    cluster_model = joblib.load("pkl/cluster_model.pkl")
    cluster_scaler = joblib.load("pkl/cluster_scaler.pkl")
    return modelo, scaler_model, cluster_model, cluster_scaler

@st.cache_data
def load_base():
    return pd.read_csv("data/df_model.csv")

@st.cache_resource
def get_shap_explainer(_modelo):
    try:
        import shap
        return shap.TreeExplainer(_modelo), True
    except Exception:
        return None, False

modelo, scaler_model, cluster_model, cluster_scaler = load_artefatos()
df_model = load_base()
shap_explainer, SHAP_OK = get_shap_explainer(modelo)

PERFIL_CLUSTER = {
    0: "Deterioração gradual",
    1: "Risco estrutural",
    2: "Falha de aprendizado",
    3: "Queda recente de aprendizado",
}

PERFIL_DESC = {
    0: "Alunos com queda progressiva e contínua de indicadores ao longo do ciclo. Tendem a vir de uma base inicial razoável e perder terreno gradualmente.",
    1: "Alunos com defasagem profunda e múltipla em vários eixos simultaneamente. Demandam intervenção estruturada e multidisciplinar.",
    2: "Alunos com falha pontual em aprendizado, geralmente concentrada em um ou dois indicadores. Boa janela para correção dirigida.",
    3: "Alunos que apresentaram queda recente, após período de estabilidade. Sinal de alerta para investigar mudanças contextuais.",
}

DIAGNOSTICO_KPI = {
    "IDA":     "Baixo desempenho acadêmico",
    "IEG":     "Baixo engajamento",
    "IPS":     "Baixa participação psicossocial",
    "IPP":     "Baixa persistência psicopedagógica",
    "IAN_LAG": "Histórico de defasagem de nível",
    "IPV":     "Baixa velocidade de aprendizado",
}

ACAO_KPI = {
    "IDA":     "Direcionar conteúdo e adaptar exercícios às lacunas observadas, com avaliação formativa quinzenal.",
    "IEG":     "Acompanhamento ativo do engajamento, com metas de participação semanais nas atividades.",
    "IPS":     "Plano de presença e participação, com atividades semanais e suporte psicossocial.",
    "IPP":     "Acompanhamento individual focado em continuidade e metas de persistência.",
    "IAN_LAG": "Plano de recuperação progressivo, baseado no histórico de defasagem identificado.",
    "IPV":     "Ajuste de ritmo de ensino, com reforço incremental e revisão contínua.",
}

FEATURES_MODELO  = ["IAN_LAG", "IEG", "IDA", "IPS", "IPP", "IPV"]
FEATURES_CLUSTER = ["IEG", "IDA", "IPV", "IPS", "IPP"]

SIM_KEYS = ["IDA", "IEG", "IPV", "IPS", "IPP", "IAN", "IAA"]

def safe_image(path):
    p = Path(path)
    return Image.open(p) if p.exists() else None

logo_pm   = safe_image("img/passos_magicos.png")
logo_fiap = safe_image("img/fiap.png")

html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"], .stMarkdown, .stText {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-feature-settings: "cv02","cv03","cv04","cv11";
}

/* ===== BODY ===== */
.main { background: #f4f6fb; }
.block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1400px; }

h1, h2, h3, h4, h5 { color: #0a0e1a; letter-spacing: -0.015em; font-weight: 700; }

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: #0a0e1a !important;
    border-right: 1px solid #1a1f33;
}
[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}
[data-testid="stSidebar"] a {
    color: #cbd5e1 !important;
    text-decoration: none !important;
    display: block;
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    margin: 2px 0;
    border-left: 2px solid transparent;
    transition: all 0.15s ease;
}
[data-testid="stSidebar"] a:hover {
    background: #1a1f33 !important;
    color: #ffffff !important;
    border-left-color: #ff005c !important;
}
.sidebar-brand {
    font-size: 11px;
    color: #9ca3af !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 6px;
}
.sidebar-title {
    font-size: 18px;
    color: #ffffff !important;
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.01em;
}
.sidebar-divider {
    border-top: 1px solid #1a1f33;
    margin: 18px -10px;
}
.sidebar-eyebrow {
    font-size: 10px;
    color: #9ca3af !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    padding: 0 14px;
    margin: 12px 0 4px;
}
.sidebar-num {
    color: #6b7280 !important;
    font-variant-numeric: tabular-nums;
    margin-right: 8px;
    font-size: 11px;
    font-weight: 700;
}

/* ===== SUPERFÍCIES ===== */
.surface {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 22px 24px;
}
.surface-tight { padding: 16px 18px; }
.surface-dark {
    background: linear-gradient(135deg, #0a0e1a 0%, #1a1f33 100%);
    color: #ffffff;
    border-radius: 12px;
    padding: 28px 32px;
    border: 1px solid #1a1f33;
    box-shadow: 0 4px 24px rgba(10,14,26,0.12);
}
.surface-pink {
    background: linear-gradient(180deg, #fff8fa 0%, #ffeaf0 100%);
    border: 1px solid #ffccdd;
    border-radius: 10px;
    padding: 22px 24px;
}
.surface-tint {
    background: #f9fafd;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 20px 22px;
}

/* ===== TIPOGRAFIA ===== */
.eyebrow {
    font-size: 10px;
    color: #6b7280;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 6px;
}
.eyebrow-light {
    font-size: 10px;
    color: rgba(255,255,255,0.55);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 8px;
}
.figure-xxl {
    font-size: 88px;
    font-weight: 900;
    line-height: 0.95;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.035em;
}
.figure-xl {
    font-size: 56px;
    font-weight: 800;
    line-height: 1;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.025em;
}
.figure {
    font-size: 38px;
    font-weight: 800;
    line-height: 1;
    color: #0a0e1a;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
}
.figure-sm {
    font-size: 22px;
    font-weight: 800;
    line-height: 1.1;
    color: #0a0e1a;
    font-variant-numeric: tabular-nums;
}
.meta { font-size: 12px; color: #6b7280; font-weight: 500; }
.meta-light { font-size: 12px; color: rgba(255,255,255,0.65); font-weight: 500; }

/* ===== SECTION HEADERS ===== */
.section-head {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin: 26px 0 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #e5e7eb;
}
.section-num {
    font-size: 11px;
    color: #ff005c;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
    letter-spacing: 0.1em;
}
.section-title {
    font-size: 18px;
    color: #0a0e1a;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.section-meta {
    font-size: 12px;
    color: #6b7280;
    margin-left: auto;
    font-weight: 500;
}

/* ===== BADGES ===== */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }

/* ===== BARRA DE PROGRESSO ===== */
.bar-track {
    background: #f3f4f6;
    height: 6px;
    border-radius: 3px;
    overflow: hidden;
    width: 100%;
}
.bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
}

/* ===== BOTÕES ===== */
.stButton > button {
    background: #ff005c;
    color: #ffffff;
    border: 1px solid #ff005c;
    padding: 11px 24px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.01em;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: #e6004f;
    border-color: #e6004f;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(255,0,92,0.25);
}
.stButton > button:focus { box-shadow: 0 0 0 3px rgba(255,0,92,0.25); }

/* botões secundários do what-if */
.what-if-btn .stButton > button {
    background: #ffffff;
    color: #0a0e1a;
    border: 1px solid #e5e7eb;
}
.what-if-btn .stButton > button:hover {
    background: #f9fafb;
    border-color: #ff005c;
    color: #ff005c;
    box-shadow: none;
    transform: none;
}

/* ===== INPUTS ===== */
.stNumberInput input, .stTextInput input {
    border-radius: 8px !important;
    border: 1px solid #e5e7eb !important;
    font-family: 'Inter', sans-serif !important;
    font-variant-numeric: tabular-nums;
}
.stSelectbox > div > div { border-radius: 8px !important; }

/* ===== TABELA DE INDICADORES ===== */
.ind-table {
    width: 100%;
    border-collapse: collapse;
    font-variant-numeric: tabular-nums;
}
.ind-table th {
    text-align: left;
    font-size: 10px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 700;
    padding: 10px 14px;
    border-bottom: 1px solid #e5e7eb;
}
.ind-table td {
    padding: 14px;
    border-bottom: 1px solid #f3f4f6;
    font-size: 13px;
    color: #0a0e1a;
    vertical-align: middle;
}
.ind-table tr:last-child td { border-bottom: none; }
.ind-table .ind-driver { background: #fff5f8; }
.ind-table .ind-val {
    font-size: 20px;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
}

hr { border: none; border-top: 1px solid #e5e7eb; margin: 1.4rem 0; }
.modebar { display: none !important; }

/* esconde labels redundantes do nav padrão */
[data-testid="collapsedControl"] { color: #ffffff !important; }
</style>
""")


def predizer_risco(IAN, IEG, IDA, IPS, IPP, IPV):
    entrada = pd.DataFrame([{
        "IAN_LAG": IAN, "IEG": IEG, "IDA": IDA,
        "IPS": IPS, "IPP": IPP, "IPV": IPV,
    }])[FEATURES_MODELO]
    entrada_scaled = scaler_model.transform(entrada)
    prob = float(modelo.predict_proba(entrada_scaled)[0][1])
    return prob, entrada, entrada_scaled

def predizer_cluster(IEG, IDA, IPV, IPS, IPP):
    entrada = pd.DataFrame([{
        "IEG": IEG, "IDA": IDA, "IPV": IPV, "IPS": IPS, "IPP": IPP,
    }])[FEATURES_CLUSTER]
    cluster_scaled = cluster_scaler.transform(entrada)
    return int(cluster_model.predict(cluster_scaled)[0])

def risco_composto(prob_modelo, valores_ind, inde_aluno, inde_base):

    s_modelo = prob_modelo * 100
    crit_count = sum(1 for v in valores_ind.values() if v < 4.0)
    media_ind  = np.mean(list(valores_ind.values()))
    s_patamar  = max(0.0, min(100.0, (8.0 - media_ind) / 8.0 * 100))
    s_patamar  = min(100.0, s_patamar + crit_count * 8)
    percentil  = (inde_base < inde_aluno).mean() * 100
    s_distrib  = 100.0 - percentil
    composto = 0.45 * s_modelo + 0.35 * s_patamar + 0.20 * s_distrib
    return composto, s_modelo, s_patamar, s_distrib

def classificar(score):
    if   score >= 70: return "Crítico",    "#991b1b", "Intervenção imediata"
    elif score >= 50: return "Alto risco", "#dc2626", "Probabilidade elevada de defasagem"
    elif score >= 35: return "Atenção",    "#d97706", "Risco moderado"
    elif score >= 20: return "Estável",    "#0284c7", "Risco baixo"
    else:             return "Excelente",  "#059669", "Risco mínimo"

def calcular_inde(IAN, IDA, IEG, IAA, IPS, IPP, IPV):
    return IAN*0.1 + IDA*0.2 + IEG*0.2 + IAA*0.1 + IPS*0.1 + IPP*0.1 + IPV*0.2

def estimar_pedra(inde):
    if   inde >= 7.5: return "Topázio"
    elif inde >= 6.5: return "Ametista"
    elif inde >= 5.5: return "Ágata"
    else:             return "Quartzo"

def get_shap_values(entrada_scaled):
    if not SHAP_OK or shap_explainer is None:
        return None
    try:
        vals = shap_explainer.shap_values(entrada_scaled)
        if isinstance(vals, list):
            vals = vals[1]
        vals = np.array(vals).flatten()
        return dict(zip(FEATURES_MODELO, vals))
    except Exception:
        return None

def svg_gauge(value_pct, color, size=240):
    value_pct = max(0.0, min(100.0, value_pct))
    angle_end = math.radians(180 - 1.8 * value_pct)
    cx, cy, r = 120, 130, 100
    end_x = cx + r * math.cos(angle_end)
    end_y = cy - r * math.sin(angle_end)
    large_arc = 0 

    ticks = ""
    for pct in [0, 25, 50, 75, 100]:
        a = math.radians(180 - 1.8 * pct)
        x1 = cx + (r + 8) * math.cos(a)
        y1 = cy - (r + 8) * math.sin(a)
        x2 = cx + (r + 14) * math.cos(a)
        y2 = cy - (r + 14) * math.sin(a)
        tx = cx + (r + 26) * math.cos(a)
        ty = cy - (r + 26) * math.sin(a)
        ticks += (
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#9ca3af" stroke-width="1.5"/>'
            f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="10" fill="#9ca3af" '
            f'text-anchor="middle" font-family="Inter" font-weight="600">{pct}</text>'
        )

    return f"""
    <svg viewBox="0 0 240 180" style="width:100%;max-width:{size}px;display:block;margin:0 auto;">
      <!-- background arc -->
      <path d="M 20 130 A 100 100 0 0 1 220 130"
            stroke="#e5e7eb" stroke-width="16" fill="none" stroke-linecap="round"/>
      <!-- value arc -->
      <path d="M 20 130 A 100 100 0 {large_arc} 1 {end_x:.1f} {end_y:.1f}"
            stroke="{color}" stroke-width="16" fill="none" stroke-linecap="round"/>
      <!-- ticks -->
      {ticks}
      <!-- center value -->
      <text x="120" y="115" text-anchor="middle"
            font-size="42" font-weight="900" fill="#0a0e1a"
            font-family="Inter" letter-spacing="-1.5">
        {value_pct:.0f}<tspan font-size="18" font-weight="700" fill="#9ca3af">%</tspan>
      </text>
      <text x="120" y="138" text-anchor="middle"
            font-size="10" fill="#6b7280" font-family="Inter"
            font-weight="700" letter-spacing="1.2">
        PROBABILIDADE
      </text>
    </svg>
    """

def render_dual_gauge_card(prob_pct, inde_val, risk_color):

    if inde_val >= 7.5:
        inde_color, inde_pedra = "#059669", "Topázio"
    elif inde_val >= 6.5:
        inde_color, inde_pedra = "#0284c7", "Ametista"
    elif inde_val >= 5.5:
        inde_color, inde_pedra = "#d97706", "Ágata"
    else:
        inde_color, inde_pedra = "#dc2626", "Quartzo"

    def _gauge_svg(pct, color, big_num_str, suffix_str, mini_label):
        pct = max(0.0, min(100.0, pct))
        angle_end = math.radians(180 - 1.8 * pct)
        cx, cy, r = 120, 130, 100
        end_x = cx + r * math.cos(angle_end)
        end_y = cy - r * math.sin(angle_end)
        ticks = ""
        for tp in [0, 25, 50, 75, 100]:
            a = math.radians(180 - 1.8 * tp)
            x1 = cx + (r + 6) * math.cos(a)
            y1 = cy - (r + 6) * math.sin(a)
            x2 = cx + (r + 12) * math.cos(a)
            y2 = cy - (r + 12) * math.sin(a)
            ticks += (
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="#cbd5e1" stroke-width="1.2"/>'
            )
        return (
            f'<svg viewBox="0 0 240 160" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;max-width:220px;height:auto;display:block;margin:0 auto;">'
            f'<path d="M 20 130 A 100 100 0 0 1 220 130" '
            f'stroke="#e5e7eb" stroke-width="14" fill="none" stroke-linecap="round"/>'
            f'<path d="M 20 130 A 100 100 0 0 1 {end_x:.1f} {end_y:.1f}" '
            f'stroke="{color}" stroke-width="14" fill="none" stroke-linecap="round"/>'
            f'{ticks}'
            f'<text x="120" y="115" text-anchor="middle" '
            f'font-size="40" font-weight="900" fill="#0a0e1a" '
            f'font-family="Inter, sans-serif" letter-spacing="-1.2">'
            f'{big_num_str}<tspan font-size="16" font-weight="700" fill="#9ca3af">{suffix_str}</tspan>'
            f'</text>'
            f'<text x="120" y="140" text-anchor="middle" font-size="9" fill="#6b7280" '
            f'font-family="Inter, sans-serif" font-weight="700" letter-spacing="1.5">'
            f'{mini_label}'
            f'</text>'
            f'</svg>'
        )

    prob_svg = _gauge_svg(prob_pct, risk_color, f"{prob_pct:.0f}", "%", "PROBABILIDADE")
    inde_pct_for_arc = (inde_val / 10.0) * 100.0
    inde_svg = _gauge_svg(inde_pct_for_arc, inde_color, f"{inde_val:.1f}", "", "INDE  0–10")

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 0; font-family: 'Inter', -apple-system, sans-serif; background: transparent; }}
.card {{
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 22px 24px;
  height: 100%;
}}
.eyebrow {{
  font-size: 10px; color: #6b7280; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.12em;
}}
.section-title {{
  font-size: 13px; color: #0a0e1a; font-weight: 700; margin-top: 4px;
}}
.gauges-row {{
  display: flex; gap: 10px; margin: 14px 0 6px;
  justify-content: space-around; align-items: flex-start;
}}
.gauge-block {{ flex: 1; text-align: center; min-width: 0; }}
.gauge-title {{
  font-size: 10px; color: #0a0e1a; font-weight: 800;
  text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;
}}
.gauge-sub {{
  font-size: 11px; font-weight: 700; margin-top: 4px;
  text-transform: uppercase; letter-spacing: 0.06em;
}}
.divider {{ width: 1px; background: #f3f4f6; align-self: stretch; margin: 4px 0; }}
.meta {{
  font-size: 11px; color: #6b7280; font-weight: 500;
  text-align: center; line-height: 1.6;
  margin-top: 14px; padding-top: 14px; border-top: 1px solid #f3f4f6;
}}
</style></head>
<body>
<div class="card">
  <div class="eyebrow">Indicadores Sintéticos</div>
  <div class="section-title">Probabilidade de Defasagem &amp; INDE</div>
  <div class="gauges-row">
    <div class="gauge-block">
      <div class="gauge-title">Risco · Modelo</div>
      {prob_svg}
      <div class="gauge-sub" style="color:{risk_color};">Saída XGBoost</div>
    </div>
    <div class="divider"></div>
    <div class="gauge-block">
      <div class="gauge-title">Desenvolvimento</div>
      {inde_svg}
      <div class="gauge-sub" style="color:{inde_color};">Pedra · {inde_pedra}</div>
    </div>
  </div>
  <div class="meta">
    Probabilidade · padrões históricos identificados pelo modelo &middot;
    INDE · índice composto institucional (PEDE 2022–2024)
  </div>
</div>
</body></html>"""


if "diagnosed" not in st.session_state:
    st.session_state.diagnosed = False
    st.session_state.inputs = None

def submit_diagnose():
    st.session_state.diagnosed = True
    st.session_state.inputs = {
        "nome": st.session_state.f_nome,
        "ra":   st.session_state.f_ra,
        "idade":st.session_state.f_idade,
        "sexo": st.session_state.f_sexo,
        "ano":  st.session_state.f_ano,
        "IAN":  st.session_state.f_IAN,
        "IEG":  st.session_state.f_IEG,
        "IDA":  st.session_state.f_IDA,
        "IPS":  st.session_state.f_IPS,
        "IPP":  st.session_state.f_IPP,
        "IPV":  st.session_state.f_IPV,
        "IAA":  st.session_state.f_IAA,
    }

    for k in SIM_KEYS:
        st.session_state[f"sim_{k}"] = float(st.session_state[f"f_{k}"])


with st.sidebar:
    if logo_pm is not None:
        st.image(logo_pm, width=70)
    html("""
    <div style="padding:6px 0 0;">
      <div class="sidebar-brand">Associação</div>
      <div class="sidebar-title">Passos Mágicos</div>
      <div style="font-size:11px;color:#9ca3af;margin-top:6px;">
        Diagnóstico Preditivo
      </div>
    </div>
    <div class="sidebar-divider"></div>
    <div class="sidebar-eyebrow">Navegação</div>
    <a href="#sobre"><span class="sidebar-num">00</span>Sobre o Projeto</a>
    <a href="#dados"><span class="sidebar-num">01</span>Dados do Aluno</a>
    <a href="#resumo"><span class="sidebar-num">02</span>Resumo Executivo</a>
    <a href="#kpis"><span class="sidebar-num">03</span>Indicadores-Chave</a>
    <a href="#graficos"><span class="sidebar-num">04</span>Análise Gráfica</a>
    <a href="#shap"><span class="sidebar-num">05</span>SHAP · Drivers</a>
    <a href="#diag"><span class="sidebar-num">06</span>Diagnóstico &amp; Ação</a>
    <a href="#painel"><span class="sidebar-num">07</span>Painel Detalhado</a>
    <a href="#whatif"><span class="sidebar-num">08</span>Simulador What-If</a>
    <div class="sidebar-divider"></div>
    <div style="font-size:10px;color:#6b7280;padding:0 14px;line-height:1.55;">
      Modelo XGBoost · ciclo 2022–2024<br/>
      Datathon FIAP · POS Tech
    </div>
    """)


html("""
<div style="background:linear-gradient(135deg,#0a0e1a 0%,#1a1f33 100%);
            padding:24px 32px;border-radius:12px;margin-bottom:24px;
            position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;height:3px;
              background:linear-gradient(90deg,#ff005c 0%,#ff6b9d 100%);"></div>
  <div style="display:flex;justify-content:space-between;align-items:center;gap:24px;flex-wrap:wrap;">
    <div>
      <div style="font-size:11px;color:rgba(255,255,255,0.6);font-weight:700;
                  text-transform:uppercase;letter-spacing:0.14em;">
        Associação Passos Mágicos · Datathon FIAP
      </div>
      <div style="font-size:28px;font-weight:800;color:#ffffff;line-height:1.1;
                  margin-top:6px;letter-spacing:-0.02em;">
        Diagnóstico Preditivo de Risco Educacional
      </div>
      <div style="font-size:13px;color:rgba(255,255,255,0.7);margin-top:6px;font-weight:500;">
        Avaliação individual com modelo XGBoost · ciclo 2022–2024
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:10px;color:rgba(255,255,255,0.5);
                  text-transform:uppercase;letter-spacing:0.12em;font-weight:700;">
        Modelo
      </div>
      <div style="font-size:14px;color:#ffffff;font-weight:700;margin-top:4px;">
        XGBoost + KMeans + SHAP
      </div>
    </div>
  </div>
</div>
""")

html('<div id="sobre"></div>')
html("""
<div class="section-head">
  <span class="section-num">00</span>
  <span class="section-title">Sobre o Projeto</span>
  <span class="section-meta">Contexto institucional · objetivo da ferramenta</span>
</div>
""")

scol1, scol2 = st.columns([1.6, 1])
with scol1:
    html("""
    <div class="surface">
      <div style="font-size:11px;color:#ff005c;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;">
        Missão
      </div>
      <div style="font-size:18px;color:#0a0e1a;font-weight:700;margin-top:8px;line-height:1.4;letter-spacing:-0.01em;">
        Antecipar quedas de desempenho de alunos em vulnerabilidade social,
        permitindo intervenção pedagógica direcionada antes da defasagem se consolidar.
      </div>
      <div style="font-size:13px;color:#4b5563;margin-top:14px;line-height:1.65;">
        A Associação Passos Mágicos atua há 35 anos transformando a vida de crianças e jovens
        de baixa renda em Embu-Guaçu por meio da educação. Este sistema integra três frentes
        analíticas — modelo preditivo, segmentação por perfil e explicabilidade SHAP — para
        gerar um diagnóstico executivo individualizado, com plano de ação acionável.
      </div>
    </div>
    """)
with scol2:
    html(f"""
    <div class="surface-dark" style="height:100%;">
      <div class="eyebrow-light">A Base de Dados</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:10px;">
        <div>
          <div style="font-size:32px;font-weight:800;color:#ffffff;font-variant-numeric:tabular-nums;letter-spacing:-0.02em;">
            {len(df_model):,}
          </div>
          <div class="meta-light">avaliações</div>
        </div>
        <div>
          <div style="font-size:32px;font-weight:800;color:#ff6b9d;font-variant-numeric:tabular-nums;letter-spacing:-0.02em;">
            3
          </div>
          <div class="meta-light">ciclos · 2022–2024</div>
        </div>
        <div>
          <div style="font-size:32px;font-weight:800;color:#ffffff;font-variant-numeric:tabular-nums;letter-spacing:-0.02em;">
            4
          </div>
          <div class="meta-light">perfis de cluster</div>
        </div>
        <div>
          <div style="font-size:32px;font-weight:800;color:#ff6b9d;font-variant-numeric:tabular-nums;letter-spacing:-0.02em;">
            7
          </div>
          <div class="meta-light">indicadores PEDE</div>
        </div>
      </div>
    </div>
    """)

html('<div id="dados"></div>')
html("""
<div class="section-head">
  <span class="section-num">01</span>
  <span class="section-title">Dados do Aluno</span>
  <span class="section-meta">Identificação e indicadores educacionais</span>
</div>
""")

with st.container():
    html('<div class="surface">')

    html("<div class='eyebrow' style='margin-bottom:10px;'>Identificação</div>")
    i1, i2, i3, i4, i5 = st.columns([2, 1, 1, 1, 1])
    with i1: st.text_input("Nome do aluno", key="f_nome", placeholder="Ex.: Maria Silva")
    with i2: st.text_input("RA / ID", key="f_ra")
    with i3: st.number_input("Idade", 6, 25, 12, key="f_idade")
    with i4: st.selectbox("Sexo", ["—", "Feminino", "Masculino"], key="f_sexo")
    with i5: st.selectbox("Ano referência", [2024, 2023, 2022], index=0, key="f_ano")

    html("<div class='eyebrow' style='margin:20px 0 10px;'>Indicadores Educacionais · escala 0 a 10</div>")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("**Acadêmico**")
        st.number_input("IDA · Desempenho",       0.0, 10.0, 6.0, 0.1, key="f_IDA")
        st.number_input("IEG · Engajamento",      0.0, 10.0, 6.0, 0.1, key="f_IEG")
        st.number_input("IPV · Ponto de Virada",  0.0, 10.0, 6.0, 0.1, key="f_IPV")
    with g2:
        st.markdown("**Psicossocial / Pedagógico**")
        st.number_input("IPS · Psicossocial",     0.0, 10.0, 6.0, 0.1, key="f_IPS")
        st.number_input("IPP · Psicopedagógico",  0.0, 10.0, 6.0, 0.1, key="f_IPP")
    with g3:
        st.markdown("**Histórico / Percepção**")
        st.number_input("IAN · Adequação de Nível", 0.0, 10.0, 6.0, 0.1, key="f_IAN")
        st.number_input("IAA · Autoavaliação",    0.0, 10.0, 6.0, 0.1, key="f_IAA")

    st.write("")
    st.button("Gerar diagnóstico preditivo", type="primary", on_click=submit_diagnose)
    html('</div>')


if not st.session_state.diagnosed:
    st.write("")
    html("""
    <div class="surface" style="text-align:center;padding:50px 30px;border-style:dashed;">
      <div class="eyebrow">Aguardando dados</div>
      <div style="font-size:18px;font-weight:700;color:#0a0e1a;margin-top:10px;">
        Preencha os indicadores do aluno e clique em <span style="color:#ff005c;">Gerar diagnóstico preditivo</span>
      </div>
      <div class="meta" style="margin-top:8px;line-height:1.55;">
        O sistema combinará modelo preditivo, patamares absolutos e distribuição comparativa
        para entregar um diagnóstico executivo com plano de ação.
      </div>
    </div>
    """)
    st.stop()


inp = st.session_state.inputs
IAN, IEG, IDA, IPS, IPP, IPV, IAA = (
    inp["IAN"], inp["IEG"], inp["IDA"], inp["IPS"], inp["IPP"], inp["IPV"], inp["IAA"]
)
valores_ind = {"IAN_LAG": IAN, "IEG": IEG, "IDA": IDA, "IPS": IPS, "IPP": IPP, "IPV": IPV}

prob_modelo, _, entrada_scaled = predizer_risco(IAN, IEG, IDA, IPS, IPP, IPV)
cluster = predizer_cluster(IEG, IDA, IPV, IPS, IPP)
perfil  = PERFIL_CLUSTER.get(cluster, "—")
perfil_desc = PERFIL_DESC.get(cluster, "")

inde_aluno = calcular_inde(IAN, IDA, IEG, IAA, IPS, IPP, IPV)
inde_base  = (
    df_model["IAN_LAG"]*0.1 + df_model["IDA"]*0.2 + df_model["IEG"]*0.2
    + df_model["IPS"]*0.1   + df_model["IPP"]*0.1 + df_model["IPV"]*0.2
)
pedra = estimar_pedra(inde_aluno)
media_inde = float(inde_base.mean())
percentil_aluno = (inde_base < inde_aluno).mean() * 100

composto, s_modelo, s_patamar, s_distrib = risco_composto(
    prob_modelo, valores_ind, inde_aluno, inde_base
)
classif_label, cor, classif_desc = classificar(composto)

cluster_size = int((df_model["CLUSTER"] == cluster).sum())
cluster_share = cluster_size / len(df_model) * 100


shap_vals = get_shap_values(entrada_scaled)
if shap_vals is not None:
    driver = max(shap_vals, key=shap_vals.get)
    driver_origem = "SHAP TreeExplainer"
else:
    medias_geral = df_model[FEATURES_MODELO].mean()
    gaps = {k: valores_ind[k] - medias_geral[k] for k in FEATURES_MODELO}
    driver = min(gaps, key=gaps.get)
    driver_origem = "gap à média (SHAP indisponível)"

driver_label = driver.replace("_LAG", "")
diagnostico_txt = DIAGNOSTICO_KPI[driver]
acao_txt        = ACAO_KPI[driver]
medias_geral = df_model[FEATURES_MODELO].mean()
gap_driver = valores_ind[driver] - medias_geral[driver]



nome_exibido = inp["nome"].strip() if inp["nome"].strip() else "Aluno"
ra_exibido   = f" · RA {inp['ra']}" if inp['ra'].strip() else ""
sexo_exibido = f" · {inp['sexo']}" if inp["sexo"] != "—" else ""

html('<div id="resumo"></div>')
html("""
<div class="section-head">
  <span class="section-num">02</span>
  <span class="section-title">Resumo Executivo</span>
  <span class="section-meta">Veredicto consolidado · perfil · ações imediatas</span>
</div>
""")

html(f"""
<div class="surface-dark">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:32px;flex-wrap:wrap;">
    <div style="flex:1.3;min-width:260px;">
      <div class="eyebrow-light">Aluno avaliado · ano {inp['ano']}</div>
      <div style="font-size:36px;font-weight:800;color:#ffffff;line-height:1.05;letter-spacing:-0.025em;margin-top:6px;">
        {nome_exibido}<span style="color:rgba(255,255,255,0.5);font-weight:500;">{ra_exibido}</span>
      </div>
      <div class="meta-light" style="margin-top:10px;font-size:14px;">
        {inp['idade']} anos{sexo_exibido}
      </div>
      <div style="margin-top:22px;display:flex;gap:18px;flex-wrap:wrap;">
        <div>
          <div class="eyebrow-light">Pedra estimada</div>
          <div style="font-size:22px;font-weight:800;color:#ff6b9d;margin-top:4px;">{pedra}</div>
        </div>
        <div>
          <div class="eyebrow-light">INDE</div>
          <div style="font-size:22px;font-weight:800;color:#ffffff;margin-top:4px;font-variant-numeric:tabular-nums;">{inde_aluno:.2f}</div>
        </div>
        <div>
          <div class="eyebrow-light">Percentil</div>
          <div style="font-size:22px;font-weight:800;color:#ffffff;margin-top:4px;font-variant-numeric:tabular-nums;">{percentil_aluno:.0f}º</div>
        </div>
      </div>
    </div>
    <div style="text-align:right;padding-left:24px;border-left:1px solid rgba(255,255,255,0.12);">
      <div class="eyebrow-light">Risco Composto</div>
      <div class="figure-xxl" style="color:{cor};">{composto:.0f}<span style="font-size:26px;color:rgba(255,255,255,0.4);font-weight:600;"> /100</span></div>
      <div style="margin-top:12px;">
        <span class="badge" style="background:{cor}30;color:#ffffff;border:1px solid {cor};">
          <span class="dot" style="background:{cor};"></span>{classif_label} · {classif_desc}
        </span>
      </div>
    </div>
  </div>
</div>
""")

st.write("")

tc1, tc2, tc3 = st.columns(3)
with tc1:
    html(f"""
    <div class="surface" style="height:100%;border-left:3px solid #6366f1;">
      <div class="eyebrow">Perfil de Cluster</div>
      <div style="font-size:11px;color:#6366f1;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-top:6px;">
        Cluster {cluster}
      </div>
      <div style="font-size:20px;font-weight:800;color:#0a0e1a;margin-top:4px;line-height:1.25;letter-spacing:-0.01em;">
        {perfil}
      </div>
      <div style="font-size:12px;color:#4b5563;margin-top:10px;line-height:1.55;">
        {perfil_desc}
      </div>
      <div style="margin-top:14px;padding-top:12px;border-top:1px solid #f3f4f6;display:flex;gap:18px;">
        <div>
          <div class="eyebrow" style="margin-bottom:2px;">Alunos similares</div>
          <div style="font-size:18px;font-weight:800;color:#0a0e1a;font-variant-numeric:tabular-nums;">{cluster_size}</div>
        </div>
        <div>
          <div class="eyebrow" style="margin-bottom:2px;">% da base</div>
          <div style="font-size:18px;font-weight:800;color:#0a0e1a;font-variant-numeric:tabular-nums;">{cluster_share:.0f}%</div>
        </div>
      </div>
    </div>
    """)
with tc2:
    html(f"""
    <div class="surface-pink" style="height:100%;">
      <div class="eyebrow" style="color:#9f1239;">Diagnóstico Principal</div>
      <div style="font-size:11px;color:#9f1239;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-top:6px;">
        Driver crítico · {driver_label}
      </div>
      <div style="font-size:22px;font-weight:800;color:#0a0e1a;margin-top:6px;line-height:1.2;letter-spacing:-0.01em;">
        {diagnostico_txt}
      </div>
      <div style="font-size:12px;color:#4b5563;margin-top:10px;line-height:1.55;">
        Indicador identificado com maior contribuição para o risco previsto. Detalhamento SHAP na seção 05.
      </div>
    </div>
    """)
with tc3:
    html(f"""
    <div class="surface" style="height:100%;border-left:3px solid #059669;">
      <div class="eyebrow">Ação Recomendada</div>
      <div style="font-size:11px;color:#059669;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;margin-top:6px;">
        Intervenção dirigida
      </div>
      <div style="font-size:14px;font-weight:600;color:#0a0e1a;margin-top:8px;line-height:1.55;">
        {acao_txt}
      </div>
    </div>
    """)

html('<div id="kpis"></div>')
html("""
<div class="section-head">
  <span class="section-num">03</span>
  <span class="section-title">Indicadores-Chave</span>
  <span class="section-meta">Decomposição do risco · 3 sinais independentes</span>
</div>
""")

def sig_color(v):
    if v >= 65: return "#dc2626"
    if v >= 40: return "#d97706"
    if v >= 20: return "#0284c7"
    return "#059669"

km1, km2, km3, km4 = st.columns(4)
for col, (titulo, val, sub) in zip(
    [km1, km2, km3, km4],
    [
        ("Sinal do Modelo", s_modelo, "XGBoost · padrões históricos"),
        ("Sinal de Patamar", s_patamar, "Indicadores abaixo do crítico"),
        ("Sinal de Distribuição", s_distrib, "Posição vs. base institucional"),
        ("Composto · Resultado", composto, "0.45 modelo + 0.35 patamar + 0.20 dist."),
    ],
):
    c = cor if titulo.startswith("Composto") else sig_color(val)
    bg_extra = f"background:{c}06;border-color:{c}40;" if titulo.startswith("Composto") else ""
    with col:
        html(f"""
        <div class="surface surface-tight" style="{bg_extra}">
          <div class="eyebrow">{titulo}</div>
          <div class="figure-sm" style="color:{c};">{val:.0f}<span style="font-size:13px;color:#9ca3af;"> /100</span></div>
          <div class="meta" style="margin-top:6px;">{sub}</div>
          <div class="bar-track" style="margin-top:12px;">
            <div class="bar-fill" style="width:{val:.0f}%;background:{c};"></div>
          </div>
        </div>
        """)


html('<div id="graficos"></div>')
html("""
<div class="section-head">
  <span class="section-num">04</span>
  <span class="section-title">Análise Gráfica</span>
  <span class="section-meta">Probabilidade do modelo · comparativo de indicadores</span>
</div>
""")

gcol1, gcol2 = st.columns([1, 1.3])

with gcol1:

    st_components.html(
        render_dual_gauge_card(prob_modelo * 100, inde_aluno, cor),
        height=340,
        scrolling=False,
    )

with gcol2:
    html('<div class="surface" style="padding-bottom:6px;">')
    html("<div class='eyebrow'>Aluno vs. Cluster vs. Média Institucional</div>")

    feats_radar = ["IEG", "IDA", "IPV", "IPS", "IPP"]
    v_aluno   = [valores_ind[f] for f in feats_radar]
    v_cluster = df_model[df_model["CLUSTER"] == cluster][feats_radar].mean().tolist()
    v_geral   = df_model[feats_radar].mean().tolist()

    def closed(lst): return lst + [lst[0]]
    cats = feats_radar + [feats_radar[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=closed(v_geral), theta=cats, name="Média Geral",
        line=dict(color="#cbd5e1", width=2), fill="none",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=closed(v_cluster), theta=cats, name=f"Cluster {cluster}",
        line=dict(color="#6366f1", width=2, dash="dot"), fill="none",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=closed(v_aluno), theta=cats, name="Aluno",
        line=dict(color="#ff005c", width=3),
        fill="toself", fillcolor="rgba(255,0,92,0.12)",
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#ffffff",
            radialaxis=dict(visible=True, range=[0, 10],
                            tickfont=dict(size=10, color="#9ca3af"),
                            gridcolor="#f1f5f9", linecolor="#e5e7eb",
                            tickvals=[2, 4, 6, 8, 10]),
            angularaxis=dict(tickfont=dict(size=13, color="#0a0e1a", family="Inter"),
                             gridcolor="#e5e7eb"),
        ),
        height=320,
        margin=dict(l=50, r=50, t=10, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                    xanchor="center", x=0.5, font=dict(size=11, color="#0a0e1a")),
    )
    st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
    html('</div>')


html('<div id="shap"></div>')
html("""
<div class="section-head">
  <span class="section-num">05</span>
  <span class="section-title">SHAP · Drivers do Modelo</span>
  <span class="section-meta">Contribuição de cada indicador para a probabilidade</span>
</div>
""")

if shap_vals is not None:
    shap_df = pd.DataFrame({
        "feat": list(shap_vals.keys()),
        "val":  list(shap_vals.values()),
    }).sort_values("val", ascending=True)
    shap_df["label"] = shap_df["feat"].str.replace("_LAG", "")
    shap_df["color"] = shap_df["val"].apply(lambda v: "#dc2626" if v > 0 else "#059669")

    top_pos = shap_df[shap_df["val"] > 0].nlargest(3, "val")
    top_neg = shap_df[shap_df["val"] < 0].nsmallest(3, "val")

    if len(top_pos) > 0:
        nomes_pos = ", ".join(top_pos["label"].tolist())
        texto_pos = f"<b style='color:#dc2626;'>Aumentam o risco:</b> {nomes_pos}."
    else:
        texto_pos = "<b style='color:#059669;'>Nenhum indicador aumenta o risco</b> de forma relevante."

    if len(top_neg) > 0:
        nomes_neg = ", ".join(top_neg["label"].tolist())
        texto_neg = f"<b style='color:#059669;'>Reduzem o risco:</b> {nomes_neg}."
    else:
        texto_neg = ""

    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        x=shap_df["val"],
        y=shap_df["label"],
        orientation="h",
        marker=dict(color=shap_df["color"], line=dict(width=0)),
        text=[f"{v:+.3f}" for v in shap_df["val"]],
        textposition="outside",
        textfont=dict(size=11, color="#0a0e1a", family="Inter"),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.3f}<extra></extra>",
    ))
    fig_shap.add_vline(x=0, line=dict(color="#9ca3af", width=1))
    fig_shap.update_layout(
        height=260,
        margin=dict(l=20, r=80, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter", color="#0a0e1a", size=12),
        xaxis=dict(
            title=dict(text="← reduz risco · aumenta risco →", font=dict(size=11, color="#6b7280")),
            gridcolor="#f3f4f6", zeroline=False, tickfont=dict(size=10, color="#6b7280"),
        ),
        yaxis=dict(gridcolor="#ffffff", tickfont=dict(size=12, color="#0a0e1a")),
        showlegend=False,
    )

    scol1, scol2 = st.columns([2, 1])
    with scol1:
        html('<div class="surface" style="padding-bottom:8px;">')
        st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})
        html('</div>')
    with scol2:
        html(f"""
        <div class="surface-tint" style="height:100%;">
          <div class="eyebrow">Como ler este gráfico</div>
          <div style="font-size:13px;color:#374151;margin-top:10px;line-height:1.65;">
            Cada barra mostra <b>quanto cada indicador empurrou a probabilidade</b> prevista para cima
            (vermelho · aumenta risco) ou para baixo (verde · reduz risco) em relação à expectativa média.
          </div>
          <div style="margin-top:14px;padding-top:12px;border-top:1px solid #e5e7eb;font-size:12px;color:#4b5563;line-height:1.7;">
            {texto_pos}<br/>
            {texto_neg}
          </div>
        </div>
        """)
else:
    html("""
    <div class="surface" style="border-style:dashed;">
      <div class="eyebrow">Análise SHAP indisponível</div>
      <div class="meta" style="margin-top:6px;">
        Adicione <code>shap</code> ao <code>requirements.txt</code> para ativar a análise.
        Driver crítico atual calculado por gap à média institucional.
      </div>
    </div>
    """)


html('<div id="diag"></div>')
html("""
<div class="section-head">
  <span class="section-num">06</span>
  <span class="section-title">Diagnóstico &amp; Plano de Ação</span>
  <span class="section-meta">Síntese acionável · driver crítico identificado</span>
</div>
""")

if shap_vals is not None:
    expl_text = (
        f"O indicador <b>{driver_label}</b> apresenta a maior contribuição SHAP positiva "
        f"(+{shap_vals[driver]:.3f}) para a probabilidade de defasagem prevista, "
        f"sendo o principal vetor de risco identificado pelo modelo."
    )
else:
    expl_text = (
        f"O indicador <b>{driver_label}</b> apresenta a maior distância negativa em relação "
        f"à média institucional ({gap_driver:+.2f} pts)."
    )

html(f"""
<div class="surface-pink">
  <div style="display:flex;gap:32px;flex-wrap:wrap;align-items:stretch;">
    <div style="flex:1.4;min-width:300px;">
      <div class="eyebrow" style="color:#9f1239;">Diagnóstico</div>
      <div style="font-size:30px;font-weight:800;color:#0a0e1a;margin-top:10px;line-height:1.15;letter-spacing:-0.025em;">
        {diagnostico_txt}
      </div>
      <div style="font-size:14px;color:#4b5563;margin-top:18px;line-height:1.7;">
        {expl_text}
      </div>
    </div>
    <div style="flex:1;min-width:300px;background:#ffffff;border:1px solid #ffccdd;
                border-radius:10px;padding:24px 26px;">
      <div class="eyebrow" style="color:#9f1239;">Plano de Ação Recomendado</div>
      <div style="font-size:17px;font-weight:600;color:#0a0e1a;margin-top:10px;
                  line-height:1.55;letter-spacing:-0.005em;">
        {acao_txt}
      </div>
      <div style="margin-top:18px;padding-top:14px;border-top:1px solid #f3f4f6;
                  font-size:11px;color:#6b7280;line-height:1.55;font-weight:500;">
        Acompanhamento sugerido: avaliação de evolução a cada 30 dias.
      </div>
    </div>
  </div>
</div>
""")


html('<div id="painel"></div>')
html("""
<div class="section-head">
  <span class="section-num">07</span>
  <span class="section-title">Painel Detalhado de Indicadores</span>
  <span class="section-meta">Valores · níveis semafóricos · driver crítico</span>
</div>
""")

def semaforo(v):
    if v >= 7.5: return "#059669", "Forte"
    if v >= 5.0: return "#d97706", "Moderado"
    return "#dc2626", "Crítico"

indicadores_lista = [
    ("IDA", "Desempenho Acadêmico", IDA, "IDA"),
    ("IEG", "Engajamento", IEG, "IEG"),
    ("IPV", "Ponto de Virada", IPV, "IPV"),
    ("IPS", "Psicossocial", IPS, "IPS"),
    ("IPP", "Psicopedagógico", IPP, "IPP"),
    ("IAN", "Adequação de Nível", IAN, "IAN_LAG"),
    ("IAA", "Autoavaliação", IAA, "IAA"),
]

rows = []
for label, desc, val, key_modelo in indicadores_lista:
    c, lvl = semaforo(val)
    media_ref = float(df_model[key_modelo].mean()) if key_modelo in df_model.columns else None
    is_driver = key_modelo == driver
    row_class = "ind-driver" if is_driver else ""
    driver_mark = f'<span style="color:#ff005c;font-weight:800;margin-left:4px;">★ DRIVER</span>' if is_driver else ""
    media_html = f"{media_ref:.1f}" if media_ref is not None else "—"
    rows.append(f"""
    <tr class="{row_class}">
      <td style="width:24%;">
        <div style="font-weight:800;font-size:14px;color:#0a0e1a;">{label} {driver_mark}</div>
        <div style="font-size:11px;color:#6b7280;margin-top:2px;">{desc}</div>
      </td>
      <td style="text-align:right;width:10%;">
        <div class="ind-val" style="color:{c};">{val:.1f}</div>
      </td>
      <td style="width:36%;">
        <div class="bar-track" style="height:8px;">
          <div class="bar-fill" style="width:{val*10:.0f}%;background:{c};"></div>
        </div>
      </td>
      <td style="text-align:right;width:14%;font-variant-numeric:tabular-nums;color:#6b7280;font-size:12px;">
        média {media_html}
      </td>
      <td style="text-align:right;width:16%;">
        <span class="badge" style="background:{c}15;color:{c};font-size:10px;padding:4px 10px;">{lvl}</span>
      </td>
    </tr>
    """)

html(f"""
<div class="surface" style="padding:6px 8px;">
  <table class="ind-table">
    <thead>
      <tr>
        <th>Indicador</th>
        <th style="text-align:right;">Valor</th>
        <th>Escala</th>
        <th style="text-align:right;">Referência</th>
        <th style="text-align:right;">Nível</th>
      </tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>
""")


html('<div id="whatif"></div>')
html("""
<div class="section-head">
  <span class="section-num">08</span>
  <span class="section-title">Simulador What-If</span>
  <span class="section-meta">Cenários de intervenção · recálculo em tempo real</span>
</div>
""")


for k in SIM_KEYS:
    sk = f"sim_{k}"
    if sk not in st.session_state:
        st.session_state[sk] = float(inp[k])

def cb_reset():
    for k in SIM_KEYS:
        st.session_state[f"sim_{k}"] = float(inp[k])

def cb_otimista():
    for k in SIM_KEYS:
        st.session_state[f"sim_{k}"] = float(min(10.0, inp[k] + 1.0))

def cb_pessimista():
    for k in SIM_KEYS:
        st.session_state[f"sim_{k}"] = float(max(0.0, inp[k] - 1.0))

def cb_driver_plus():
    for k in SIM_KEYS:
        st.session_state[f"sim_{k}"] = float(inp[k])
    key_sim = driver.replace("_LAG", "")
    if key_sim in SIM_KEYS:
        st.session_state[f"sim_{key_sim}"] = float(min(10.0, inp[key_sim] + 2.0))

with st.container():
    html('<div class="surface">')

    html("""
    <div class="eyebrow">Como usar</div>
    <div style="font-size:13px;color:#4b5563;line-height:1.6;margin:6px 0 18px;">
      Ajuste os indicadores para simular como intervenções pedagógicas impactam o risco previsto.
      O recálculo é instantâneo e considera os três sinais (modelo, patamar e distribuição).
    </div>
    """)

    html('<div class="what-if-btn">')
    sc1, sc2, sc3, sc4, _ = st.columns([1, 1, 1, 1, 3])
    with sc1: st.button("↺ Reset",         key="b_reset",     on_click=cb_reset)
    with sc2: st.button("↑ Otimista +1",   key="b_otimista",  on_click=cb_otimista)
    with sc3: st.button("★ Driver +2",     key="b_driver",    on_click=cb_driver_plus)
    with sc4: st.button("↓ Pessimista −1", key="b_pessimista",on_click=cb_pessimista)
    html('</div>')

    st.write("")

    sg1, sg2, sg3 = st.columns(3)
    with sg1:
        st.slider("IDA · Desempenho",     0.0, 10.0, step=0.1, key="sim_IDA")
        st.slider("IEG · Engajamento",    0.0, 10.0, step=0.1, key="sim_IEG")
        st.slider("IPV · Ponto de Virada",0.0, 10.0, step=0.1, key="sim_IPV")
    with sg2:
        st.slider("IPS · Psicossocial",   0.0, 10.0, step=0.1, key="sim_IPS")
        st.slider("IPP · Psicopedagógico",0.0, 10.0, step=0.1, key="sim_IPP")
    with sg3:
        st.slider("IAN · Adequ. de Nível",0.0, 10.0, step=0.1, key="sim_IAN")
        st.slider("IAA · Autoavaliação",  0.0, 10.0, step=0.1, key="sim_IAA")

   
    sim_IDA = st.session_state["sim_IDA"]
    sim_IEG = st.session_state["sim_IEG"]
    sim_IPV = st.session_state["sim_IPV"]
    sim_IPS = st.session_state["sim_IPS"]
    sim_IPP = st.session_state["sim_IPP"]
    sim_IAN = st.session_state["sim_IAN"]
    sim_IAA = st.session_state["sim_IAA"]

    sim_valores = {"IAN_LAG": sim_IAN, "IEG": sim_IEG, "IDA": sim_IDA,
                   "IPS": sim_IPS, "IPP": sim_IPP, "IPV": sim_IPV}
    sim_prob_modelo, _, _ = predizer_risco(sim_IAN, sim_IEG, sim_IDA, sim_IPS, sim_IPP, sim_IPV)
    sim_inde = calcular_inde(sim_IAN, sim_IDA, sim_IEG, sim_IAA, sim_IPS, sim_IPP, sim_IPV)
    sim_composto, _, _, _ = risco_composto(sim_prob_modelo, sim_valores, sim_inde, inde_base)
    sim_classif, sim_cor, _ = classificar(sim_composto)

    delta_composto = sim_composto - composto
    delta_inde     = sim_inde - inde_aluno
    delta_arrow    = "↓" if delta_composto < -0.5 else ("↑" if delta_composto > 0.5 else "→")
    delta_cor      = "#059669" if delta_composto < -0.5 else ("#dc2626" if delta_composto > 0.5 else "#6b7280")

    st.write("")

    sm1, sm2, sm3 = st.columns(3)
    with sm1:
        html(f"""
        <div class="surface surface-tight" style="border-color:#e5e7eb;background:#f9fafb;">
          <div class="eyebrow">Risco Atual</div>
          <div class="figure-sm" style="color:{cor};">{composto:.0f}<span style="font-size:13px;color:#9ca3af;"> /100</span></div>
          <div class="meta" style="margin-top:4px;">{classif_label}</div>
        </div>
        """)
    with sm2:
        html(f"""
        <div class="surface surface-tight" style="border-color:{sim_cor};background:{sim_cor}10;">
          <div class="eyebrow">Risco Simulado</div>
          <div class="figure-sm" style="color:{sim_cor};">{sim_composto:.0f}<span style="font-size:13px;color:#9ca3af;"> /100</span></div>
          <div class="meta" style="margin-top:4px;">{sim_classif}</div>
        </div>
        """)
    with sm3:
        html(f"""
        <div class="surface surface-tight">
          <div class="eyebrow">Variação</div>
          <div class="figure-sm" style="color:{delta_cor};">{delta_arrow} {abs(delta_composto):.1f}<span style="font-size:13px;color:#9ca3af;"> pts</span></div>
          <div class="meta" style="margin-top:4px;">INDE: {delta_inde:+.2f}</div>
        </div>
        """)

    html('</div>')

st.write("")


html(f"""
<div style="margin-top:24px;padding:16px 20px;background:#0a0e1a;border-radius:10px;
            color:rgba(255,255,255,0.7);font-size:11px;line-height:1.7;">
  <b style="color:#ffffff;">Notas técnicas.</b>
  Modelo XGBoost treinado sobre histórico 2022–2024 da Associação Passos Mágicos.
  Features de risco: <code style="background:#1a1f33;color:#ff6b9d;padding:2px 6px;border-radius:4px;">{', '.join(FEATURES_MODELO)}</code>.
  Segmentação KMeans (k=4) sobre <code style="background:#1a1f33;color:#ff6b9d;padding:2px 6px;border-radius:4px;">{', '.join(FEATURES_CLUSTER)}</code>.
  Risco Composto pondera três sinais independentes para mitigar extrapolação do modelo em regiões raras do espaço de entrada.
  Driver crítico via {driver_origem}.
</div>
""")