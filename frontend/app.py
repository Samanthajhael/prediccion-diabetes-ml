"""Interfaz gráfica (Streamlit) del sistema de predicción de diabetes.

Consume la API de FastAPI. Ejecutar con:
    streamlit run frontend/app.py
"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
AZULES = ["#0B2E59", "#2171B5", "#6BAED6", "#9ECAE1", "#C6DBEF"]

st.set_page_config(page_title="Predicción de Diabetes · ML", page_icon="🩺", layout="wide")


# Funciones de apoyo
def api(metodo: str, ruta: str, **kwargs):
    try:
        r = requests.request(metodo, f"{API_URL}{ruta}", timeout=120, **kwargs)
    except requests.ConnectionError:
        st.error(f"No se pudo conectar con la API en {API_URL}. Inicie el backend con: "
                 "`uvicorn backend.main:app --reload`")
        st.stop()
    if r.status_code >= 400:
        detalle = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        st.error(f"Error {r.status_code}: {detalle}")
        return None
    return r.json()


def tarjetas_metricas(m: dict):
    nombres = {"accuracy": "Exactitud", "precision": "Precisión", "recall": "Sensibilidad (Recall)",
               "f1": "F1-score", "roc_auc": "ROC AUC"}
    cols = st.columns(len(nombres))
    for col, (k, etiqueta) in zip(cols, nombres.items()):
        col.metric(etiqueta, f"{m[k]:.3f}")


NOMBRES_ES = {"Pregnancies": "Embarazos", "Glucose": "Glucosa", "BloodPressure": "Presión diastólica",
              "SkinThickness": "Pliegue cutáneo", "Insulin": "Insulina", "BMI": "IMC",
              "DiabetesPedigreeFunction": "Antecedente familiar", "Age": "Edad"}


def graficos_metricas(info: dict, prefijo: str):
    """Dibuja los gráficos de métricas; `prefijo` da claves únicas porque se usa en dos pestañas."""
    c1, c2 = st.columns(2)
    cm = info["matriz_confusion"]
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=["Pred. sin diabetes", "Pred. con diabetes"], y=["Real sin diabetes", "Real con diabetes"],
                    title="Matriz de confusión (conjunto de prueba)")
    fig.update_layout(coloraxis_showscale=False, height=380)
    c1.plotly_chart(fig, use_container_width=True, key=f"{prefijo}_g0")

    roc = info["curva_roc"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", line=dict(color=AZULES[1], width=3),
                             name=f"AUC = {info['metricas_prueba']['roc_auc']:.3f}", fill="tozeroy",
                             fillcolor="rgba(33,113,181,0.15)"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash="dash"),
                             name="Azar"))
    fig.update_layout(title="Curva ROC", xaxis_title="Tasa de falsos positivos",
                      yaxis_title="Tasa de verdaderos positivos", height=380, legend=dict(x=0.55, y=0.08))
    c2.plotly_chart(fig, use_container_width=True, key=f"{prefijo}_g1")

    c3, c4 = st.columns(2)
    imp = pd.Series(info["importancia_variables"]).rename(index=NOMBRES_ES).sort_values()
    fig = px.bar(imp, orientation="h", title="Importancia de variables (permutación, caída de AUC)",
                 color_discrete_sequence=[AZULES[1]])
    fig.update_layout(showlegend=False, xaxis_title="Importancia", yaxis_title="", height=380)
    c3.plotly_chart(fig, use_container_width=True, key=f"{prefijo}_g2")

    cv = pd.DataFrame(info["validacion_cruzada"]).T.rename(
        index={"accuracy": "Exactitud", "f1": "F1", "roc_auc": "ROC AUC", "recall": "Sensibilidad"})
    fig = go.Figure(go.Bar(x=cv.index, y=cv["media"], error_y=dict(type="data", array=cv["desv"]),
                           marker_color=AZULES[0], text=cv["media"].round(3), textposition="outside"))
    fig.update_layout(title=f"Validación cruzada ({info['folds_cv']} particiones, media ± desv.)",
                      yaxis_range=[0, 1.05], height=380)
    c4.plotly_chart(fig, use_container_width=True, key=f"{prefijo}_g3")


tab_datos, tab_entrenar, tab_metricas, tab_predecir, tab_historial = st.tabs(
    ["📊 Datos", "⚙️ Entrenamiento", "📈 Métricas", "🧑‍⚕️ Nuevo caso", "🕘 Historial"])

# Pestaña: Datos
with tab_datos:
    st.header("Exploración del dataset")
    r = api("GET", "/dataset/resumen")
    if r:
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros", r["filas"])
        c2.metric("Con diabetes", f"{r['clases']['con_diabetes']} ({100 * r['clases']['con_diabetes'] / r['filas']:.1f} %)")
        c3.metric("Sin diabetes", f"{r['clases']['sin_diabetes']} ({100 * r['clases']['sin_diabetes'] / r['filas']:.1f} %)")
        c1, c2 = st.columns(2)
        ceros = pd.Series(r["ceros_como_faltantes"]).rename(index=r["nombres_es"])
        fig = px.bar(ceros, title="Valores en 0 tratados como faltantes (se imputan con la mediana)",
                     color_discrete_sequence=[AZULES[1]], text_auto=True)
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Registros", height=380)
        c1.plotly_chart(fig, use_container_width=True)
        fig = px.pie(names=["Sin diabetes", "Con diabetes"],
                     values=[r["clases"]["sin_diabetes"], r["clases"]["con_diabetes"]],
                     title="Distribución de la variable objetivo", color_discrete_sequence=[AZULES[3], AZULES[0]], hole=0.45)
        fig.update_layout(height=380)
        c2.plotly_chart(fig, use_container_width=True)
        st.subheader("Estadística descriptiva (tras la limpieza)")
        st.dataframe(pd.DataFrame(r["descripcion"]).rename(columns=r["nombres_es"]), use_container_width=True)
        st.subheader("Primeros registros")
        st.dataframe(pd.DataFrame(r["muestra"]), use_container_width=True)

# Pestaña: Entrenamiento
with tab_entrenar:
    st.header("Configurar y entrenar el modelo")
    algoritmos = api("GET", "/algoritmos") or {}
    if algoritmos:
        clave = st.selectbox("Algoritmo", list(algoritmos), index=list(algoritmos).index("random_forest"),
                             format_func=lambda k: algoritmos[k]["nombre"])
        st.markdown("**Hiperparámetros**")
        parametros = {}
        cols = st.columns(3)
        for i, (nombre, s) in enumerate(algoritmos[clave]["parametros"].items()):
            col = cols[i % 3]
            if s["tipo"] == "int":
                parametros[nombre] = col.number_input(nombre, int(s["min"]), int(s["max"]), int(s["defecto"]),
                                                      step=1, help=s["ayuda"], key=f"{clave}_{nombre}")
            elif s["tipo"] == "float":
                parametros[nombre] = col.number_input(nombre, float(s["min"]), float(s["max"]), float(s["defecto"]),
                                                      format="%.3f", help=s["ayuda"], key=f"{clave}_{nombre}")
            else:
                parametros[nombre] = col.selectbox(nombre, s["opciones"], index=s["opciones"].index(s["defecto"]),
                                                   help=s["ayuda"], key=f"{clave}_{nombre}")
        st.markdown("**Validación**")
        c1, c2, c3 = st.columns(3)
        test_size = c1.slider("Proporción de prueba", 0.1, 0.5, 0.2, 0.05)
        folds = c2.slider("Particiones de validación cruzada", 2, 10, 5)
        semilla = c3.number_input("Semilla aleatoria", 0, 10_000, 42)
        if st.button("Entrenar modelo", type="primary"):
            with st.spinner("Entrenando y evaluando..."):
                res = api("POST", "/entrenar", json={"algoritmo": clave, "parametros": parametros,
                                                      "test_size": test_size, "folds_cv": folds, "semilla": semilla})
            if res:
                st.success(f"Modelo {res['nombre_algoritmo']} entrenado en {res['segundos_entrenamiento']} s "
                           f"con {res['n_entrenamiento']} registros; evaluado con {res['n_prueba']}. "
                           "Ahora es el modelo activo de la API.")
                tarjetas_metricas(res["metricas_prueba"])
                graficos_metricas(res, "entrenar")

# Pestaña: Métricas
with tab_metricas:
    st.header("Métricas del modelo activo")
    info = api("GET", "/metricas")
    if info:
        st.markdown(f"**Algoritmo:** {info['nombre_algoritmo']} · **Parámetros:** `{info['parametros']}` · "
                    f"**Prueba:** {int(info['test_size'] * 100)} % · **Fecha:** {info['fecha']}")
        tarjetas_metricas(info["metricas_prueba"])
        graficos_metricas(info, "metricas")
        with st.expander("¿Cómo interpretar las métricas?"):
            st.markdown(
                "- **Exactitud:** porcentaje total de aciertos.\n"
                "- **Precisión:** de los casos que el modelo marca como positivos, cuántos lo son realmente.\n"
                "- **Sensibilidad (Recall):** de las personas con diabetes, cuántas detecta el modelo. "
                "En un tamizaje médico es la métrica más importante, porque un falso negativo deja a alguien sin atención.\n"
                "- **F1-score:** equilibrio entre precisión y sensibilidad.\n"
                "- **ROC AUC:** capacidad de separar ambas clases (1 = perfecto, 0,5 = azar).")

# Pestaña: Nuevo caso
with tab_predecir:
    st.header("Evaluar un caso nuevo")
    st.caption("Ingrese los datos clínicos. Pliegue cutáneo e insulina son opcionales: si se dejan en 0, el modelo los imputa.")
    with st.form("caso"):
        c1, c2, c3, c4 = st.columns(4)
        caso = {
            "Pregnancies": c1.number_input("Embarazos", 0, 20, 2),
            "Glucose": c2.number_input("Glucosa (mg/dL)", 1.0, 300.0, 120.0),
            "BloodPressure": c3.number_input("Presión diastólica (mm Hg)", 1.0, 200.0, 70.0),
            "SkinThickness": c4.number_input("Pliegue tríceps (mm) · opcional", 0.0, 100.0, 0.0),
            "Insulin": c1.number_input("Insulina 2 h (µU/mL) · opcional", 0.0, 900.0, 0.0),
            "BMI": c2.number_input("IMC (kg/m²)", 1.0, 80.0, 30.0, format="%.1f"),
            "DiabetesPedigreeFunction": c3.number_input("Antecedente familiar (pedigrí)", 0.0, 3.0, 0.45, format="%.3f"),
            "Age": c4.number_input("Edad (años)", 21, 120, 35),
        }
        enviar = st.form_submit_button("Predecir", type="primary")
    if enviar:
        for k in ("SkinThickness", "Insulin"):
            if caso[k] == 0:
                caso[k] = None
        res = api("POST", "/predecir", json=caso)
        if res:
            prob = res["probabilidad_diabetes"]
            c1, c2 = st.columns([1, 1.3])
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100, number={"suffix": " %"},
                title={"text": "Probabilidad de diabetes"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": AZULES[0]},
                       "steps": [{"range": [0, 30], "color": AZULES[4]}, {"range": [30, 60], "color": AZULES[3]},
                                 {"range": [60, 100], "color": AZULES[2]}],
                       "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
            fig.update_layout(height=320)
            c1.plotly_chart(fig, use_container_width=True)
            mensaje = c2.error if res["prediccion"] == 1 else c2.success
            mensaje(f"**{res['etiqueta']}**")
            c2.markdown(f"- **Nivel de riesgo:** {res['nivel_riesgo']}\n"
                        f"- **Probabilidad:** {prob:.1%}\n- **Modelo usado:** {res['modelo']}")
            c2.info(res["aviso"])

# Pestaña: Historial
with tab_historial:
    st.header("Historial de entrenamientos")
    hist = api("GET", "/historial") or []
    if hist:
        filas = [{"Fecha": h["fecha"], "Algoritmo": h["nombre_algoritmo"], "Prueba": h["test_size"],
                  **{k.upper(): v for k, v in h["metricas_prueba"].items()}, "Parámetros": str(h["parametros"])}
                 for h in reversed(hist)]
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
        cron = pd.DataFrame(filas[::-1])
        cron["N.º"] = [f"#{i + 1}" for i in range(len(cron))]
        fig = px.bar(cron, x="N.º", y="ROC_AUC", color="Algoritmo", text_auto=".3f",
                     color_discrete_sequence=AZULES, title="ROC AUC por entrenamiento (orden cronológico)")
        fig.update_layout(xaxis_title="Entrenamiento", yaxis_range=[0, 1], height=360)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Todavía no hay entrenamientos registrados.")

# Barra lateral (al final, para mostrar el modelo activo tras entrenar)
with st.sidebar:
    st.title("🩺 Predicción de Diabetes")
    st.caption("Proyecto Final · Proyectos de Machine Learning · CENESTUR")
    estado = api("GET", "/")
    if estado and estado["modelo_cargado"]:
        st.success(f"API conectada\n\nModelo activo: **{estado['modelo']}**\n\nEntrenado: {estado['entrenado_en']}")
    st.markdown(f"API: `{API_URL}` · [Documentación]({API_URL}/docs)")
    st.divider()
    st.caption("Dataset: Pima Indians Diabetes Database (NIDDK; Smith et al., 1988). "
               "Herramienta educativa: no reemplaza un diagnóstico médico.")
