import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Axial Flare | AI Drilling Co-Pilot",
    page_icon="\U0001F525",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CUSTOM CSS
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
.stApp { background: linear-gradient(180deg, #0a0e17 0%, #111827 50%, #0f172a 100%); font-family: 'Inter', sans-serif; }
.hero-title { font-size: 2.8rem; font-weight: 800; background: linear-gradient(135deg, #f97316 0%, #ef4444 50%, #f59e0b 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 0; letter-spacing: -1px; }
.hero-sub { color: #94a3b8; text-align: center; font-size: 1.05rem; font-weight: 300; }
.metric-card { background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(30,41,59,0.9)); border: 1px solid rgba(148,163,184,0.2); border-radius: 16px; padding: 20px; text-align: center; backdrop-filter: blur(10px); transition: all 0.3s ease; }
.metric-card:hover { border-color: rgba(249,115,22,0.5); box-shadow: 0 0 30px rgba(249,115,22,0.1); }
.metric-value { font-size: 2.2rem; font-weight: 700; margin: 6px 0; }
.metric-label { color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1.5px; }
.metric-unit { color: #64748b; font-size: 0.85rem; }
.alert-banner { border-radius: 12px; padding: 14px 20px; margin: 12px 0; font-weight: 600; text-align: center; font-size: 1rem; }
.alert-optimal { background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.4); color: #22c55e; }
.alert-caution { background: rgba(234,179,8,0.1); border: 1px solid rgba(234,179,8,0.4); color: #eab308; }
.alert-critical { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.5); color: #ef4444; animation: blink 1s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.6} }
.section-hdr { color: #f97316; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px; margin-top: 28px; margin-bottom: 8px; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
header[data-testid="stHeader"] {display: none !important;}
[data-testid="stToolbar"] {display: none !important;}
[data-testid="stDecoration"] {display: none !important;}
.mobile-tip { display: none; background: rgba(249,115,22,0.12); border: 1px solid rgba(249,115,22,0.4); border-radius: 10px; padding: 10px 16px; color: #f97316; font-size: 0.88rem; text-align: center; margin-bottom: 14px; }
@media (max-width: 768px) { .mobile-tip { display: block !important; } }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CONSTANTS & HELPERS
# ==============================================================================
BIT_DIAMETER = 12.25
BIT_AREA = (np.pi / 4) * BIT_DIAMETER**2
FEATURES = ["WOB_klbs", "RPM", "Torque_ftlbs", "Depth_ft", "Flow_gpm", "GR_API"]
TARGET = "ROP_fthr"
SEED = 42

def calc_mse(wob, rpm, torque, rop):
    rop = max(rop, 0.1)
    return ((wob*1000/BIT_AREA) + (120*np.pi*rpm*torque*12) / (BIT_AREA*rop*12)) / 1000

def mse_zone(mse):
    if mse < 30: return "Optimal", "optimal", "#22c55e"
    elif mse < 60: return "Caution", "caution", "#eab308"
    else: return "CRITICAL", "critical", "#ef4444"

# ==============================================================================
# LOAD DATA & TRAIN MODEL (cached — runs once)
# ==============================================================================
@st.cache_resource
def load_and_train():
    base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base, "data", "volve_f9a_processed.csv"))
    ml = df[FEATURES + [TARGET]].dropna()
    X = ml[FEATURES]
    y = ml[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=SEED)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    return model, df, r2

model, df_drill, r2_val = load_and_train()

# ==============================================================================
# ==============================================================================
# HEADER
# ==============================================================================
st.markdown('<h1 class="hero-title">\U0001F525 AXIAL FLARE</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">AI Drilling Co-Pilot \u2014 Prescriptive Parameter Optimization</p>', unsafe_allow_html=True)
st.markdown(f'<p class="hero-sub" style="font-size:0.8rem;color:#64748b;">Equinor Volve Field (North Sea) | Gradient Boosting R\u00b2 = {r2_val:.3f} | {len(df_drill):,} points</p>', unsafe_allow_html=True)
st.markdown("---")

# ==============================================================================
# CONTROLS — EXPANDER (works on mobile + desktop)
# ==============================================================================
with st.expander("⚙️ Drilling Parameters — Adjust to predict ROP", expanded=True):
    c_a, c_b = st.columns(2)
    with c_a:
        st.markdown('<p class="section-hdr">Surface Controls</p>', unsafe_allow_html=True)
        wob    = st.slider("Weight on Bit (klbs)", 2.0, 40.0, 20.0, 0.5)
        rpm    = st.slider("RPM", 20, 200, 150, 5)
        torque = st.slider("Torque (ft-lbs)", 500, 7500, 4500, 100)
    with c_b:
        st.markdown('<p class="section-hdr">Well Conditions</p>', unsafe_allow_html=True)
        depth = st.slider("Depth (ft)", 1000, 4000, 2500, 50)
        flow  = st.slider("Flow Rate (gpm)", 100, 1000, 550, 25)
        gr    = st.slider("Gamma Ray (API)", 10, 200, 100, 5)

st.markdown("---")

# ==============================================================================
# PREDICTIONS
# ==============================================================================
input_data = pd.DataFrame([{"WOB_klbs":wob,"RPM":rpm,"Torque_ftlbs":torque,"Depth_ft":depth,"Flow_gpm":flow,"GR_API":gr}])
pred_rop = max(model.predict(input_data[FEATURES])[0], 0.1)
mse_val = calc_mse(wob, rpm, torque, pred_rop)
z_label, z_class, z_color = mse_zone(mse_val)
efficiency = min(100, max(0, (1 - (mse_val - 5) / 55) * 100))



# ==============================================================================
# METRICS
# ==============================================================================
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Predicted ROP</div><div class="metric-value" style="color:#22c55e;">{pred_rop:.1f}</div><div class="metric-unit">ft/hr</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">MSE</div><div class="metric-value" style="color:{z_color};">{mse_val:.1f}</div><div class="metric-unit">kpsi</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">MSE Zone</div><div class="metric-value" style="color:{z_color};">{z_label}</div><div class="metric-unit">efficiency status</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Efficiency</div><div class="metric-value" style="color:{z_color};">{efficiency:.0f}%</div><div class="metric-unit">energy utilization</div></div>', unsafe_allow_html=True)

alerts = {"optimal": "\u2705 OPTIMAL \u2014 Drilling efficiently. Energy going into rock destruction.",
          "caution": "\u26a0\ufe0f CAUTION \u2014 MSE elevated. Check for vibrations. Consider adjusting RPM.",
          "critical": "\U0001F6A8 CRITICAL \u2014 Severe energy waste! Likely vibrations or bit balling. REDUCE WOB!"}
st.markdown(f'<div class="alert-banner alert-{z_class}">{alerts[z_class]}</div>', unsafe_allow_html=True)

# ==============================================================================
# CONTOUR MAP
# ==============================================================================
st.markdown("---")
st.markdown("### \U0001F5FA\ufe0f Parameter Optimization Landscape")

@st.cache_data
def gen_contour(_model_id, med_dict):
    wr = np.linspace(5, 35, 50)
    rr = np.linspace(30, 200, 50)
    W, R = np.meshgrid(wr, rr)
    preds, mses = [], []
    for w, r in zip(W.ravel(), R.ravel()):
        row = med_dict.copy()
        row["WOB_klbs"], row["RPM"] = w, r
        p = max(model.predict(pd.DataFrame([row])[FEATURES])[0], 0.1)
        preds.append(p)
        mses.append(calc_mse(w, r, med_dict["Torque_ftlbs"], p))
    return W, R, np.array(preds).reshape(W.shape), np.array(mses).reshape(W.shape)

med = df_drill[FEATURES].median().to_dict()
W, R, ROP_g, MSE_g = gen_contour(id(model), med)

fig = go.Figure()
fig.add_trace(go.Heatmap(z=ROP_g, x=W[0], y=R[:,0], colorscale="Magma",
    zsmooth="best",
    colorbar=dict(title=dict(text="ROP (ft/hr)", side="right"))))
fig.add_trace(go.Contour(z=MSE_g, x=W[0], y=R[:,0],
    contours=dict(start=15, end=80, size=10, showlabels=True,
        labelfont=dict(size=10, color="white")),
    line=dict(color="cyan", width=2, dash="dash"),
    showscale=False, showlegend=True, name="MSE (kpsi)",
    contours_coloring="none"))
fig.add_trace(go.Scatter(x=[wob], y=[rpm], mode="markers+text",
    marker=dict(size=20, color="lime", symbol="star",
        line=dict(width=2, color="black")),
    text=[f"YOU: {pred_rop:.0f} ft/hr"], textposition="top center",
    textfont=dict(color="lime", size=13, family="Inter"),
    name="Current Position"))
fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,1)", xaxis_title="WOB (klbs)", yaxis_title="RPM",
    height=540, font=dict(family="Inter"), margin=dict(l=60,r=80,t=20,b=60),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.6)",
        font=dict(color="white")))
st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# BOTTOM ROW
# ==============================================================================
cl, cr = st.columns(2)
with cl:
    st.markdown("### \U0001F4CA Feature Importance")
    imp = model.feature_importances_
    fdf = pd.DataFrame({"Feature":FEATURES,"Importance":imp}).sort_values("Importance",ascending=True)
    fi = go.Figure(go.Bar(x=fdf["Importance"],y=fdf["Feature"],orientation="h",
        marker=dict(color=fdf["Importance"],colorscale="Viridis"),
        text=[f"{v:.1%}" for v in fdf["Importance"]], textposition="outside"))
    fi.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,1)",height=320,margin=dict(l=100,r=60,t=10,b=30),
        xaxis=dict(showgrid=False),font=dict(family="Inter"))
    st.plotly_chart(fi, use_container_width=True)

with cr:
    st.markdown("### \U0001F52C Real Data \u2014 WOB vs ROP")
    sample = df_drill.sample(min(800, len(df_drill)), random_state=42)
    fs = go.Figure()
    for zn, zc in [("Optimal","#22c55e"),("Caution","#eab308"),("Critical","#ef4444")]:
        zd = sample[sample["MSE_kpsi"].apply(lambda m: mse_zone(m)[0]) == zn]
        if len(zd)>0:
            fs.add_trace(go.Scatter(x=zd["WOB_klbs"],y=zd["ROP_fthr"],mode="markers",
                name=zn,marker=dict(size=5,color=zc,opacity=0.6)))
    fs.add_trace(go.Scatter(x=[wob],y=[pred_rop],mode="markers",name="You",
        marker=dict(size=16,color="lime",symbol="star",line=dict(width=2,color="black"))))
    fs.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,1)",height=320,margin=dict(l=60,r=30,t=10,b=30),
        xaxis_title="WOB (klbs)",yaxis_title="ROP (ft/hr)",font=dict(family="Inter"),
        legend=dict(x=0.01,y=0.99,bgcolor="rgba(0,0,0,0.5)"))
    st.plotly_chart(fs, use_container_width=True)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(f"""
<div style="text-align:center; padding:16px 0;">
    <div style="font-size:1.1rem; font-weight:800; color:#e6edf3; letter-spacing:0.5px; margin-bottom:4px;">
        Romer Villalobos
    </div>
    <div style="font-size:0.85rem; color:#e67e22; font-weight:600; margin-bottom:8px;">
        Senior Drilling & Completion Engineer &nbsp;|&nbsp; ML Engineer &nbsp;|&nbsp; Houston, TX
    </div>
    <div style="font-size:0.75rem; color:#8b949e; margin-bottom:10px;">
        Axial Flare — AI Drilling Co-Pilot &nbsp;·&nbsp;
        Random Forest &nbsp;·&nbsp; R² = {r2_val:.3f} &nbsp;·&nbsp;
        Equinor Volve Field F-9A · North Sea<br/>
        AI & ML Postgraduate Capstone · Built independently as a Senior Drilling Engineer
    </div>
    <div style="display:flex; justify-content:center; gap:16px; flex-wrap:wrap;">
        <a href="https://www.linkedin.com/in/romer-villalobos-2b33a653"
           target="_blank"
           style="color:#e67e22; font-size:0.78rem; text-decoration:none;
                  border:1px solid #e67e22; padding:4px 14px; border-radius:20px;">
            LinkedIn
        </a>
        <a href="https://github.com/romerdaniel/axial-flare"
           target="_blank"
           style="color:#8b949e; font-size:0.78rem; text-decoration:none;
                  border:1px solid #30363d; padding:4px 14px; border-radius:20px;">
            GitHub
        </a>
        <a href="https://drillsense-copilot.streamlit.app/"
           target="_blank"
           style="color:#2ecc71; font-size:0.78rem; text-decoration:none;
                  border:1px solid #2ecc71; padding:4px 14px; border-radius:20px;">
            🛡️ DrillSense
        </a>
    </div>
</div>
""", unsafe_allow_html=True)
