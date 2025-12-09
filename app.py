import io
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# 1. Load data CAvideos.csv + preprocessing time-series
# ======================================================================

@st.cache_data
def load_timeseries(uploaded_bytes: bytes | None):
    """
    Load dataset from uploaded bytes if provided; otherwise from local file.
    """
    if uploaded_bytes is not None:
        df = pd.read_csv(io.BytesIO(uploaded_bytes))
    else:
        try:
            # Cari file relatif terhadap lokasi app.py supaya tidak tergantung cwd
            data_path = Path(__file__).resolve().parent / "CAvideos.csv"
            df = pd.read_csv(data_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "CAvideos.csv tidak ditemukan. Upload file lewat sidebar atau letakkan di direktori yang sama dengan app.py."
            ) from exc

    ts = (
        df.groupby("trending_date")["views"]
        .sum()
        .sort_index()
        .reset_index()
    )

    ts["t"] = range(len(ts))
    ts["cum_views"] = ts["views"].cumsum()

    y = ts["cum_views"].astype(float).values
    t = ts["t"].astype(float).values

    y_max = y.max()
    y_norm = y / y_max

    return t, y_norm, y_max


# ======================================================================
# 2. Logistic ODE + RK4
# ======================================================================

def logistic_rhs(t, C, r):
    return r * C * (1 - C)

def rk4_logistic(t0, C0, h, n_steps, r):
    t = t0
    C = C0
    t_list = [t]
    C_list = [C]

    for _ in range(n_steps):
        k1 = logistic_rhs(t, C, r)
        k2 = logistic_rhs(t + 0.5*h, C + 0.5*h*k1, r)
        k3 = logistic_rhs(t + 0.5*h, C + 0.5*h*k2, r)
        k4 = logistic_rhs(t + h, C + h*k3, r)

        C = C + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        t = t + h

        t_list.append(t)
        C_list.append(C)

    return np.array(t_list), np.array(C_list)

def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))


# ======================================================================
# STREAMLIT UI
# ======================================================================

st.set_page_config(page_title="TA-10 Simulasi RK4", layout="wide")

st.title("📊 TA-10 | Simulasi Sistem Dinamis dengan RK4")
st.subheader("Pertumbuhan Kumulatif Views Trending YouTube (CAvideos)")

# Optional upload if file tidak tersedia di server
uploaded_file = st.sidebar.file_uploader("Upload CAvideos.csv (opsional)", type=["csv"])
uploaded_bytes = uploaded_file.getvalue() if uploaded_file else None

data_path = Path(__file__).resolve().parent / "CAvideos.csv"

try:
    t, y_norm, y_max = load_timeseries(uploaded_bytes)
except FileNotFoundError as e:
    st.error(str(e))
    st.info(
        f"Debug info: mencari file di `{data_path}`. "
        f"Working dir saat ini: `{Path.cwd()}`. "
        "Pastikan nama file persis `CAvideos.csv` (case-sensitive)."
    )
    # Tampilkan isi direktori untuk membantu debug
    dir_listing = "\n".join(str(p.name) for p in data_path.parent.glob("*"))
    st.code(dir_listing or "(direktori kosong)")
    st.stop()

# UI parameter input
st.sidebar.header("Parameter Simulator")
r_value = st.sidebar.slider("Nilai parameter r (growth rate):", 
                            min_value=0.001, max_value=0.2, value=0.05, step=0.001)

h_value = st.sidebar.select_slider("Step size (h):", options=[0.1, 0.5, 1.0], value=1.0)

# Simulate
t_sim, C_sim = rk4_logistic(0, y_norm[0], h_value, len(t)-1, r_value)

# Error
error_norm = rmse(C_sim, y_norm)
error_real = error_norm * y_max

# Layout
col1, col2 = st.columns(2)

with col1:
    st.write("### Hasil Error Analysis")
    st.write(f"**RMSE (Normalized)** : `{error_norm:.6f}`")
    st.write(f"**RMSE (Real Views)** : `{error_real:,.0f}` views")

with col2:
    st.write("### Parameter Model")
    st.write(f"Parameter r digunakan: `{r_value}`")
    st.write(f"Step size RK4 (h): `{h_value}`")

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(t, y_norm, label="Data Normalized", s=10)
ax.plot(t_sim, C_sim, label=f"RK4 Logistic (r={r_value:.3f})", linewidth=2)
ax.set_xlabel("t (hari)")
ax.set_ylabel("C_norm(t)")
ax.set_title("Pertumbuhan Kumulatif Views Trending YouTube")
ax.grid(True)
ax.legend()

st.pyplot(fig)

st.markdown("---")
st.caption("Metode Numerik: Runge-Kutta Orde 4 | Model: Pertumbuhan Logistik")
