import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import time
from datetime import datetime
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Seizure Prediction", layout="wide")

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0f172a; color: white; }
.card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 15px;
    border: 1px solid #334155;
}
.card h2 { margin: 6px 0 0 0; font-size: 1.8rem; }
.card p  { margin: 0; color: #94a3b8; font-size: 0.85rem; }
.seizure-card {
    background-color: #3d0a0a;
    border: 2px solid #ef4444;
    padding: 20px; border-radius: 15px;
    text-align: center; margin-bottom: 15px;
}
.normal-card {
    background-color: #0a2e1a;
    border: 2px solid #22c55e;
    padding: 20px; border-radius: 15px;
    text-align: center; margin-bottom: 15px;
}
h1 { text-align: center; color: #38bdf8; }
section[data-testid="stSidebar"] { background-color: #1e293b; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;">
    <h1>🧠 Seizure Prediction</h1>
    <p style="color:#94a3b8">Real-Time EEG Seizure Monitoring System</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    # Try saved pickle models first
    if os.path.exists('models/random_forest.pkl') and os.path.exists('models/scaler.pkl'):
        with open('models/random_forest.pkl', 'rb') as f: model  = pickle.load(f)
        with open('models/scaler.pkl',        'rb') as f: scaler = pickle.load(f)
        return model, scaler
    # joblib format
    if os.path.exists('seizure_model.pkl') and os.path.exists('scaler.pkl'):
        return joblib.load('seizure_model.pkl'), joblib.load('scaler.pkl')
    # Fallback: train Random Forest
    df = load_data_raw(r"C:/Users/AB Tech/Desktop/Seizure Prediction/Epileptic Seizure Recognition.csv")
    X  = df.drop(['patient_id', 'y'], axis=1)
    y  = (df['y'] == 1).astype(int)
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xt, _, yt, _ = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    m  = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(Xt, yt)
    return m, sc

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
def load_data_raw(path_or_buffer):
    """
    Reads the CSV, extracts real patient ID from the first column
    (format like X21.V1.791 → patient P021), and assigns group numbers.
    """
    df = pd.read_csv(path_or_buffer)

    # First column = row identifier like X21.V1.791
    id_col = df.columns[0]
    raw_ids = df[id_col].astype(str)

    # Extract patient number from format X21.V1.791 → 21
    
    def extract_patient(raw):
        try:
            # Format: X<num>.V<num>.<num>
            part = raw.split('.')[0]          # e.g. X21
            num  = ''.join(filter(str.isdigit, part))
            return f"P{int(num):03d}" if num else None
        except:
            return None

    extracted = raw_ids.apply(extract_patient)

    if extracted.notna().sum() > len(df) * 0.8:
        # Successfully extracted patient IDs
        df['patient_id'] = extracted
    else:
        # Fallback: group every 23 rows as one patient
        df['patient_id'] = ['P' + str(i // 23 + 1).zfill(3) for i in range(len(df))]

    # Drop original ID column
    df = df.drop(columns=[id_col])

    # Binary label
    df['y'] = df['y'].apply(lambda x: 1 if x == 1 else 0)

    return df

@st.cache_data
def load_data(path_or_buffer):
    return load_data_raw(path_or_buffer)

# ─────────────────────────────────────────────
#  GAUGE CHART
# ─────────────────────────────────────────────
def gauge_chart(prob, threshold):
    color = "#ef4444" if prob > threshold else "#22c55e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob, 4),
        number={'font': {'color': color, 'size': 32}},
        title={'text': "Seizure Risk", 'font': {'color': '#94a3b8', 'size': 13}},
        gauge={
            'axis':  {'range': [0, 1], 'tickcolor': '#94a3b8',
                      'tickfont': {'color': '#94a3b8'}},
            'bar':   {'color': color},
            'bgcolor': '#1e293b',
            'bordercolor': '#334155',
            'steps': [
                {'range': [0,    0.35], 'color': '#0a2e1a'},
                {'range': [0.35, 0.55], 'color': '#1a3a00'},
                {'range': [0.55, 0.75], 'color': '#3e2e00'},
                {'range': [0.75, 0.90], 'color': '#3e1500'},
                {'range': [0.90, 1.0],  'color': '#3d0a0a'},
            ],
            'threshold': {
                'line': {'color': '#ffa726', 'width': 3},
                'thickness': 0.85,
                'value': threshold
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=10),
        height=240
    )
    return fig

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Control Panel")
page = st.sidebar.radio("📌 Navigation", ["Dashboard", "Patient Lookup", "About"])

uploaded_file = st.sidebar.file_uploader("📁 Upload EEG CSV", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.sidebar.success(f"✅ Loaded: {len(df)} rows")
else:
    df = load_data('Epileptic Seizure Recognition.csv')
    st.sidebar.info("Using default dataset")

THRESHOLD = st.sidebar.slider("⚠️ Threshold", 0.0, 1.0, 0.3, 0.01)

start = st.sidebar.button("▶️ Start Monitoring", use_container_width=True)
stop  = st.sidebar.button("⏹️ Stop Monitoring",  use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Random Forest")
st.sidebar.markdown("**Accuracy:** 97.7%")
st.sidebar.markdown("**F1-Score:** 94.2%")

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
with st.spinner("Loading model..."):
    model, scaler = load_artifacts()

# ─────────────────────────────────────────────
#  ABOUT PAGE
# ─────────────────────────────────────────────
if page == "About":
    st.title("📖 About This System")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### What it does
        This system analyzes EEG brain signals to automatically
        detect epileptic seizures in real time.

        **Model:** Random Forest (100 decision trees)
        **Accuracy:** 97.7%
        **F1-Score:** 94.2%
        **Dataset:** University of Bonn, Germany
        """)
    with col2:
        st.markdown("""
        ### Severity Scale
        | Level | Probability |
        |-------|------------|
        | 🔴 Critical | ≥ 0.90 |
        | 🟠 High | ≥ 0.75 |
        | 🟡 Moderate | ≥ 0.55 |
        | 🟢 Low Risk | ≥ 0.35 |
        | ✅ Normal | < 0.35 |
        """)
    st.stop()

# ─────────────────────────────────────────────
#  PATIENT LOOKUP PAGE
# ─────────────────────────────────────────────
if page == "Patient Lookup":
    st.title("🔍 Patient Lookup")

    patient_list = sorted(df['patient_id'].unique())
    selected_id  = st.selectbox("Select Patient ID", patient_list)

    if st.button("🔎 Get Patient Data", use_container_width=True):
        patient_rows = df[df['patient_id'] == selected_id]

        if patient_rows.empty:
            st.error("❌ Patient not found")
        else:
            px = patient_rows.drop(['patient_id', 'y'], axis=1)
            py = patient_rows['y']

            # Predict all rows for this patient
            px_scaled = scaler.transform(px.values)
            probs      = model.predict_proba(px_scaled)[:, 1]
            preds      = (probs > THRESHOLD).astype(int)

            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Records",      len(patient_rows))
            c2.metric("Seizures Detected",  int(preds.sum()))
            c3.metric("Normal Readings",    int((preds == 0).sum()))
            c4.metric("Avg Probability",    f"{probs.mean():.4f}")

            st.markdown("---")

            # Probability plot for all records
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(probs)+1)),
                y=probs,
                mode='lines+markers',
                line=dict(color='#38bdf8', width=2),
                marker=dict(
                    color=['#ef4444' if p == 1 else '#22c55e' for p in preds],
                    size=6
                ),
                fill='tozeroy',
                fillcolor='rgba(56,189,248,0.1)',
                name='Probability'
            ))
            fig.add_hline(y=THRESHOLD, line_dash='dash',
                          line_color='#ffa726',
                          annotation_text=f'Threshold ({THRESHOLD})',
                          annotation_font_color='#ffa726')
            fig.update_layout(
                title=dict(text=f'All EEG Readings — Patient {selected_id}',
                           font=dict(color='white')),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#0f172a',
                xaxis=dict(color='#94a3b8', title='EEG Reading Number',
                           gridcolor='#1e293b'),
                yaxis=dict(color='#94a3b8', title='Seizure Probability',
                           gridcolor='#1e293b', range=[0, 1.05]),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("**Detailed Records**")
            result_df = px.copy()
            result_df.insert(0, 'Reading #',        range(1, len(result_df)+1))
            result_df.insert(1, 'True Label',        ['Seizure' if v == 1 else 'Normal' for v in py])
            result_df.insert(2, 'Predicted',         ['Seizure' if p == 1 else 'Normal' for p in preds])
            result_df.insert(3, 'Probability',       np.round(probs, 4))
            # Show only summary columns to avoid wide table
            st.dataframe(result_df[['Reading #', 'True Label', 'Predicted', 'Probability']],
                         use_container_width=True, hide_index=True)

    st.stop()

# ─────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────

# Patient selector on dashboard
patient_list = sorted(df['patient_id'].unique())
selected_patient = st.selectbox("🆔 Select Patient to Monitor", patient_list)

patient_df = df[df['patient_id'] == selected_patient].reset_index(drop=True)
X = patient_df.drop(['patient_id', 'y'], axis=1)
y = patient_df['y']

# Patient info cards
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"""<div class='card'><p>Patient ID</p><h2 style='color:#38bdf8'>{selected_patient}</h2></div>""", unsafe_allow_html=True)
c2.markdown(f"""<div class='card'><p>Total EEG Records</p><h2>{len(patient_df)}</h2></div>""", unsafe_allow_html=True)
c3.markdown(f"""<div class='card'><p>Known Seizures</p><h2 style='color:#ef4444'>{int(y.sum())}</h2></div>""", unsafe_allow_html=True)
c4.markdown(f"""<div class='card'><p>Normal Records</p><h2 style='color:#22c55e'>{int((y==0).sum())}</h2></div>""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if 'history'      not in st.session_state: st.session_state.history      = []
if 'time_data'    not in st.session_state: st.session_state.time_data    = []
if 'pred_history' not in st.session_state: st.session_state.pred_history = []
if 'stop_stream'  not in st.session_state: st.session_state.stop_stream  = False

# ─────────────────────────────────────────────
#  UI PLACEHOLDERS
# ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
prob_ph   = col1.empty()
status_ph = col2.empty()
thresh_ph = col3.empty()

st.markdown("---")
col4, col5 = st.columns([1, 2])
gauge_ph = col4.empty()
chart_ph = col5.empty()

st.markdown("---")
progress_ph = st.empty()
stats_ph    = st.empty()
caption_ph  = st.empty()

# ─────────────────────────────────────────────
#  STREAM FUNCTION — reads ALL records
# ─────────────────────────────────────────────
def stream_data():
    st.session_state.history      = []
    st.session_state.time_data    = []
    st.session_state.pred_history = []
    st.session_state.stop_stream  = False

    total_records = len(X)   # ALL records for this patient — no limit

    for i in range(total_records):

        # ── Stop check ──
        if st.session_state.stop_stream:
            st.warning(f"⏹️ Monitoring stopped at record {i+1} of {total_records}.")
            break

        # ── Predict ──
        sample        = X.iloc[i].values.astype(float).reshape(1, -1)
        sample_scaled = scaler.transform(sample)
        prob          = model.predict_proba(sample_scaled)[0][1]
        pred          = 1 if prob > THRESHOLD else 0
        now           = datetime.now()

        # ── Update history ──
        st.session_state.history.append(round(prob, 4))
        st.session_state.time_data.append(now)
        st.session_state.pred_history.append(pred)

        # Keep last 50 for chart display (all are stored in history)
        display_history = st.session_state.history[-50:]
        display_times   = st.session_state.time_data[-50:]
        display_preds   = st.session_state.pred_history[-50:]

        # ── Probability card ──
        color = '#ef4444' if pred == 1 else '#22c55e'
        prob_ph.markdown(
            f"<div class='card'><p>Seizure Probability</p>"
            f"<h2 style='color:{color}'>{prob:.4f}</h2></div>",
            unsafe_allow_html=True)

        # ── Status card ──
        if pred == 1:
            status_ph.markdown(
                "<div class='seizure-card'><p style='color:#fca5a5'>Status</p>"
                "<h2 style='color:#ef4444'>🚨 SEIZURE</h2></div>",
                unsafe_allow_html=True)
        else:
            status_ph.markdown(
                "<div class='normal-card'><p style='color:#86efac'>Status</p>"
                "<h2 style='color:#22c55e'>✅ NORMAL</h2></div>",
                unsafe_allow_html=True)

        # ── Threshold card ──
        thresh_ph.markdown(
            f"<div class='card'><p>Threshold</p>"
            f"<h2 style='color:#ffa726'>{THRESHOLD}</h2></div>",
            unsafe_allow_html=True)

        # ── Gauge ──
        gauge_ph.plotly_chart(
    gauge_chart(prob, THRESHOLD),
    use_container_width=True,
    key=f"gauge_{i}"
)

        # ── Line chart 
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(max(1, i-48), i+2)),
            y=display_history,
            mode='lines+markers',
            line=dict(color='#38bdf8', width=2),
            marker=dict(
                color=['#ef4444' if p == 1 else '#22c55e' for p in display_preds],
                size=6
            ),
            fill='tozeroy',
            fillcolor='rgba(56,189,248,0.1)'
        ))
        fig.add_hline(y=THRESHOLD, line_dash='dash',
                      line_color='#ffa726',
                      annotation_text=f'Threshold ({THRESHOLD})',
                      annotation_font_color='#ffa726')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#0f172a',
            xaxis=dict(color='#94a3b8', title='EEG Reading #',
                       gridcolor='#1e293b'),
            yaxis=dict(color='#94a3b8', title='Seizure Probability',
                       gridcolor='#1e293b', range=[0, 1.05]),
            margin=dict(l=10, r=10, t=20, b=10),
            height=240,
            showlegend=False
        )
        chart_ph.plotly_chart(
        fig,
        use_container_width=True,
        key=f"chart_{i}"
        )

        # ── Progress bar ──
        progress_ph.progress(
            (i + 1) / total_records,
            text=f"Reading {i+1} of {total_records} — Patient {selected_patient}"
        )

        # ── Running stats ──
        seizures_so_far = sum(st.session_state.pred_history)
        avg_prob        = round(np.mean(st.session_state.history), 4)
        s1, s2, s3, s4  = stats_ph.columns(4)
        s1.metric("Records Processed", f"{i+1} / {total_records}")
        s2.metric("Seizures Flagged",  seizures_so_far)
        s3.metric("Normal Readings",   (i+1) - seizures_so_far)
        s4.metric("Avg Probability",   avg_prob)

        # ── Caption ──
        caption_ph.caption(
            f"Time: {now.strftime('%H:%M:%S')} | "
            f"True Label: {'Seizure' if y.iloc[i]==1 else 'Normal'} | "
            f"Predicted: {'Seizure' if pred==1 else 'Normal'} | "
            f"Patient: {selected_patient}"
        )

        time.sleep(0.2)

    # ── Finished all records ──
    if not st.session_state.stop_stream:
        st.success(f"✅ Completed all {total_records} EEG records for Patient {selected_patient}!")
        total_s = sum(st.session_state.pred_history)
        total_n = total_records - total_s
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Total Records",     total_records)
        f2.metric("Seizures Detected", total_s)
        f3.metric("Normal Readings",   total_n)
        f4.metric("Final Avg Prob",    round(np.mean(st.session_state.history), 4))

# ─────────────────────────────────────────────
#  START / STOP
# ─────────────────────────────────────────────
if stop:
    st.session_state.stop_stream = True

if start:
    stream_data()