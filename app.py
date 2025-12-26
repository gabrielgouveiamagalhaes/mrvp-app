import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="MRV-P Navigator", layout="wide")

# -----------------------------
# STATE
# -----------------------------
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "runs" not in st.session_state:
    st.session_state.runs = []  # histórico de registros (cada linha = uma execução)

def log_event(event: str, details: dict | None = None):
    st.session_state.audit_log.append({
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "event": event,
        "details": details or {}
    })

def safe_div(a, b, default=np.nan):
    return a / b if b not in (0, 0.0, None) else default

def compute_features(row: dict) -> dict:
    # Entradas
    horas = float(row.get("horas_corte", 0))
    energia = float(row.get("energia_kwh", 0))
    viagens = float(row.get("num_viagens", 0))
    area = float(row.get("area_m2", 0))
    peso = float(row.get("peso_estimado_t", 0))

    # KPIs (eficiência)
    aco_por_hora = safe_div(peso, max(horas, 1e-9), default=0.0)
    aco_por_kwh = safe_div(peso, max(energia, 1e-9), default=0.0)
    aco_por_viagem = safe_div(peso, max(viagens, 1e-9), default=0.0)
    aco_por_m2 = safe_div(peso, max(area, 1e-9), default=0.0)

    # OEI simples (placeholder do paper): (produtividade) / energia
    oei = safe_div(aco_por_hora, max(energia, 1e-9), default=0.0)

    return {
        "horas_corte": horas,
        "energia_kwh": energia,
        "num_viagens": viagens,
        "area_m2": area,
        "peso_estimado_t": peso,
        "aco_por_hora": aco_por_hora,
        "aco_por_kwh": aco_por_kwh,
        "aco_por_viagem": aco_por_viagem,
        "aco_por_m2": aco_por_m2,
        "OEI": oei
    }

def predict_steel_t(features: dict) -> tuple[float, float, float]:
    """
    MVP sem modelo:
    - Usa peso_estimado_t como baseline
    - Ajusta levemente por eficiência (OEI) para não ficar bobo
    - Retorna (pred, p05, p95) como faixa simples
    """
    base = features["peso_estimado_t"] * 0.95

    # ajuste fraco por OEI (normalizado) só para dar sinal
    oei = features["OEI"]
    adj = 1.0 + np.clip(oei * 10_000, -0.05, 0.05)  # controle de magnitude
    pred = base * adj

    # intervalo simples (p05/p95) como placeholder de incerteza
    p05 = pred * 0.90
    p95 = pred * 1.10
    return float(pred), float(p05), float(p95)

def mrv_score(features: dict, w_comp=0.4, w_cons=0.3, w_evid=0.3, evidence_level=0.8) -> dict:
    # Completude: entradas mínimas
    required = ["horas_corte", "energia_kwh", "num_viagens", "area_m2", "peso_estimado_t"]
    filled = sum(1 for k in required if features.get(k, 0) not in (0, 0.0, None, np.nan))
    completude = filled / len(required)

    # Consistência: heurísticas simples
    cons = 1.0
    if features["horas_corte"] <= 0 or features["energia_kwh"] <= 0:
        cons -= 0.3
    if features["peso_estimado_t"] <= 0:
        cons -= 0.4
    if features["aco_por_hora"] > 200:  # regra conservadora
        cons -= 0.2
    consistencia = float(np.clip(cons, 0.0, 1.0))

    evidencia = float(np.clip(evidence_level, 0.0, 1.0))

    score = w_comp * completude + w_cons * consistencia + w_evid * evidencia
    status = "CONFORME" if score >= 0.80 else "ATENÇÃO" if score >= 0.60 else "NÃO CONFORME"
    return {
        "completude": float(completude),
        "consistencia": float(consistencia),
        "evidencia": float(evidencia),
        "score": float(score),
        "status": status
    }

# -----------------------------
# UI HEADER
# -----------------------------
st.title("MRV-P Navigator")
st.caption("MVP funcional: inputs → KPIs → predição + intervalo → MRV score → audit trail → export")

tabs = st.tabs(["Entrada", "Resultados", "Auditoria", "Export"])

# -----------------------------
# TAB 1: ENTRADA
# -----------------------------
with tabs[0]:
    st.subheader("1) Entrada de dados")
    mode = st.radio("Modo", ["Manual", "Upload CSV"], horizontal=True)

    template = pd.DataFrame([{
        "obra_id": "P2-ICTSI",
        "data": "2025-12-26",
        "horas_corte": 120,
        "energia_kwh": 4500,
        "num_viagens": 15,
        "area_m2": 1800,
        "peso_estimado_t": 900
    }])

    if mode == "Upload CSV":
        st.write("Formato esperado (colunas): `obra_id,data,horas_corte,energia_kwh,num_viagens,area_m2,peso_estimado_t`")
        st.download_button(
            "Baixar template CSV",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="template_mrvp.csv",
            mime="text/csv"
        )
        up = st.file_uploader("Envie o CSV", type=["csv"])
        if up:
            df = pd.read_csv(up)
            st.dataframe(df, use_container_width=True)
            if st.button("Processar todas as linhas"):
                for _, r in df.iterrows():
                    row = r.to_dict()
                    feats = compute_features(row)
                    pred, p05, p95 = predict_steel_t(feats)
                    mrv = mrv_score(feats)
                    out = {**row, **feats, "aco_previsto_t": pred, "p05_t": p05, "p95_t": p95, **{f"mrv_{k}": v for k, v in mrv.items()}}
                    st.session_state.runs.append(out)
                log_event("batch_processed", {"rows": int(len(df))})
                st.success(f"Processado: {len(df)} linha(s). Vá para a aba Resultados.")
    else:
        c1, c2, c3 = st.columns(3)
        obra_id = c1.text_input("Obra / Asset ID", value="P2-ICTSI")
        data = c2.text_input("Data (YYYY-MM-DD)", value="2025-12-26")
        evidence_level = c3.slider("Evidência (placeholder)", 0.0, 1.0, 0.8, 0.05)

        c1, c2, c3, c4, c5 = st.columns(5)
        horas = c1.number_input("Horas de corte", min_value=0.0, value=120.0)
        energia = c2.number_input("Energia (kWh)", min_value=0.0, value=4500.0)
        viagens = c3.number_input("Viagens", min_value=0.0, value=15.0)
        area = c4.number_input("Área (m²)", min_value=0.0, value=1800.0)
        peso = c5.number_input("Peso estimado (t)", min_value=0.0, value=900.0)

        # pesos MRV ajustáveis
        st.markdown("**Pesos do MRV Score**")
        w1, w2, w3 = st.columns(3)
        w_comp = w1.slider("Completude (α)", 0.0, 1.0, 0.4, 0.05)
        w_cons = w2.slider("Consistência (β)", 0.0, 1.0, 0.3, 0.05)
        w_evid = w3.slider("Evidência (γ)", 0.0, 1.0, 0.3, 0.05)

        # normaliza pesos (evita soma diferente de 1)
        s = max(w_comp + w_cons + w_evid, 1e-9)
        w_comp, w_cons, w_evid = w_comp/s, w_cons/s, w_evid/s

        if st.button("Rodar MRV-P"):
            row = {
                "obra_id": obra_id,
                "data": data,
                "horas_corte": horas,
                "energia_kwh": energia,
                "num_viagens": viagens,
                "area_m2": area,
                "peso_estimado_t": peso
            }
            feats = compute_features(row)
            pred, p05, p95 = predict_steel_t(feats)
            mrv = mrv_score(feats, w_comp=w_comp, w_cons=w_cons, w_evid=w_evid, evidence_level=evidence_level)

            out = {**row, **feats, "aco_previsto_t": pred, "p05_t": p05, "p95_t": p95, **{f"mrv_{k}": v for k, v in mrv.items()}}
            st.session_state.runs.append(out)
            log_event("single_run", {"obra_id": obra_id, "data": data})
            st.success("Execução salva. Vá para a aba Resultados.")

# -----------------------------
# TAB 2: RESULTADOS
# -----------------------------
with tabs[1]:
    st.subheader("2) Resultados")
    if not st.session_state.runs:
        st.info("Nenhuma execução ainda. Vá em Entrada e rode o MRV-P.")
    else:
        df_runs = pd.DataFrame(st.session_state.runs)

        last = df_runs.iloc[-1].to_dict()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Aço previsto (t)", f"{last['aco_previsto_t']:.2f}")
        c2.metric("P05–P95 (t)", f"{last['p05_t']:.2f} – {last['p95_t']:.2f}")
        c3.metric("MRV Score", f"{last['mrv_score']:.2f}")
        c4.metric("Status", f"{last['mrv_status']}")

        st.markdown("**KPIs operacionais (última execução)**")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Aço / hora (t/h)", f"{last['aco_por_hora']:.2f}")
        k2.metric("Aço / kWh (t/kWh)", f"{last['aco_por_kwh']:.6f}")
        k3.metric("Aço / viagem (t)", f"{last['aco_por_viagem']:.2f}")
        k4.metric("Aço / m² (t/m²)", f"{last['aco_por_m2']:.6f}")

        st.markdown("**Histórico (todas as execuções)**")
        st.dataframe(df_runs.sort_values(by=["data", "obra_id"], ascending=False), use_container_width=True)

# -----------------------------
# TAB 3: AUDITORIA
# -----------------------------
with tabs[2]:
    st.subheader("3) Auditoria (Audit Trail)")
    if not st.session_state.audit_log:
        st.info("Sem eventos ainda.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        if st.button("Limpar log (somente sessão)"):
            st.session_state.audit_log = []
            log_event("audit_cleared", {})
            st.success("Log limpo (da sessão atual).")
    with colB:
        st.caption("Sessão = enquanto essa aba estiver aberta. Persistência/assinatura entra na versão 2.")

# -----------------------------
# TAB 4: EXPORT
# -----------------------------
with tabs[3]:
    st.subheader("4) Export")
    if not st.session_state.runs:
        st.info("Nada para exportar ainda.")
    else:
        df_runs = pd.DataFrame(st.session_state.runs)
        df_audit = pd.DataFrame(st.session_state.audit_log)

        st.download_button(
            "Baixar resultados (CSV)",
            data=df_runs.to_csv(index=False).encode("utf-8"),
            file_name="mrvp_resultados.csv",
            mime="text/csv"
        )
        st.download_button(
            "Baixar audit trail (CSV)",
            data=df_audit.to_csv(index=False).encode("utf-8"),
            file_name="mrvp_audit_log.csv",
            mime="text/csv"
        )

        if st.button("Resetar execuções (somente sessão)"):
            st.session_state.runs = []
            log_event("runs_cleared", {})
            st.success("Execuções resetadas (da sessão atual).")
