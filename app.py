import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="MRV-P Navigator", layout="wide")

st.title("MRV-P Navigator")
st.caption("Measurement, Reporting & Verification – Predictive")

st.sidebar.header("Dados Operacionais")

horas_corte = st.sidebar.number_input("Horas de corte", min_value=0.0, value=120.0)
energia = st.sidebar.number_input("Energia consumida (kWh)", min_value=0.0, value=4500.0)
num_viagens = st.sidebar.number_input("Número de viagens", min_value=0, value=15)
area = st.sidebar.number_input("Área operacional (m²)", min_value=0.0, value=1800.0)
peso_estimado = st.sidebar.number_input("Peso estimado inicial (t)", min_value=0.0, value=900.0)

st.header("Predição (MVP)")
aco_previsto = peso_estimado * 0.95
st.metric("Aço reciclável previsto (t)", f"{aco_previsto:.2f}")

st.header("MRV Score")

completude = 1.0 if horas_corte > 0 and energia > 0 else 0.5
consistencia = 1.0 if num_viagens > 0 else 0.6
evidencia = 0.8

mrv_score = 0.4 * completude + 0.3 * consistencia + 0.3 * evidencia
st.metric("MRV Score", f"{mrv_score:.2f}")

if mrv_score >= 0.8:
    st.success("Status: CONFORME")
elif mrv_score >= 0.6:
    st.warning("Status: ATENÇÃO")
else:
    st.error("Status: NÃO CONFORME")
