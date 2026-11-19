"""Lógica de Machine Learning: carga y limpieza de datos, entrenamiento, métricas y predicción.

Dataset: Pima Indians Diabetes Database (National Institute of Diabetes and Digestive and
Kidney Diseases; Smith et al., 1988). 768 mujeres de ascendencia pima, mayores de 21 años.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "diabetes.csv"
CARPETA_MODELOS = RAIZ / "modelos"
RUTA_MODELO = CARPETA_MODELOS / "modelo_actual.joblib"
RUTA_HISTORIAL = CARPETA_MODELOS / "historial.json"

OBJETIVO = "Outcome"
VARIABLES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
             "BMI", "DiabetesPedigreeFunction", "Age"]
NOMBRES_ES = {
    "Pregnancies": "Embarazos",
    "Glucose": "Glucosa (mg/dL)",
    "BloodPressure": "Presión diastólica (mm Hg)",
    "SkinThickness": "Pliegue cutáneo del tríceps (mm)",
    "Insulin": "Insulina a las 2 h (µU/mL)",
    "BMI": "Índice de masa corporal (kg/m²)",
    "DiabetesPedigreeFunction": "Antecedente familiar (pedigrí)",
    "Age": "Edad (años)",
}
# En el dataset original un 0 en estas columnas significa "dato no registrado"
CEROS_COMO_FALTANTES = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Especificación de algoritmos y sus hiperparámetros configurables (la usa el frontend)
ALGORITMOS: dict[str, dict[str, Any]] = {
    "regresion_logistica": {
        "nombre": "Regresión Logística",
        "parametros": {
            "C": {"tipo": "float", "defecto": 1.0, "min": 0.001, "max": 100.0,
                  "ayuda": "Inverso de la regularización: valores bajos = modelo más simple"},
            "max_iter": {"tipo": "int", "defecto": 1000, "min": 100, "max": 10000,
                         "ayuda": "Iteraciones máximas del optimizador"},
            "class_weight": {"tipo": "opcion", "defecto": "balanced", "opciones": ["ninguno", "balanced"],
                             "ayuda": "'balanced' compensa el desbalance de clases"},
        },
    },
    "random_forest": {
        "nombre": "Random Forest",
        "parametros": {
            "n_estimators": {"tipo": "int", "defecto": 300, "min": 10, "max": 1000,
                             "ayuda": "Número de árboles del bosque"},
            "max_depth": {"tipo": "int", "defecto": 6, "min": 0, "max": 50,
                          "ayuda": "Profundidad máxima de cada árbol (0 = sin límite)"},
            "min_samples_leaf": {"tipo": "int", "defecto": 2, "min": 1, "max": 50,
                                 "ayuda": "Mínimo de muestras en cada hoja"},
            "class_weight": {"tipo": "opcion", "defecto": "balanced", "opciones": ["ninguno", "balanced"],
                             "ayuda": "'balanced' compensa el desbalance de clases"},
        },
    },
    "svm": {
        "nombre": "Máquina de Vectores de Soporte (SVM)",
        "parametros": {
            "C": {"tipo": "float", "defecto": 1.0, "min": 0.001, "max": 100.0,
                  "ayuda": "Penalización del error: valores altos = margen más estrecho"},
            "kernel": {"tipo": "opcion", "defecto": "rbf", "opciones": ["rbf", "linear", "poly", "sigmoid"],
                       "ayuda": "Función kernel"},
            "gamma": {"tipo": "opcion", "defecto": "scale", "opciones": ["scale", "auto"],
                      "ayuda": "Coeficiente del kernel"},
            "class_weight": {"tipo": "opcion", "defecto": "balanced", "opciones": ["ninguno", "balanced"],
                             "ayuda": "'balanced' compensa el desbalance de clases"},
        },
    },
    "knn": {
        "nombre": "K Vecinos más Cercanos (KNN)",
        "parametros": {
            "n_neighbors": {"tipo": "int", "defecto": 15, "min": 1, "max": 100,
                            "ayuda": "Número de vecinos que votan"},
            "weights": {"tipo": "opcion", "defecto": "distance", "opciones": ["uniform", "distance"],
                        "ayuda": "'distance' da más peso a los vecinos cercanos"},
        },
    },
    "gradient_boosting": {
        "nombre": "Gradient Boosting",
        "parametros": {
            "n_estimators": {"tipo": "int", "defecto": 150, "min": 10, "max": 1000,
                             "ayuda": "Número de etapas de boosting"},
            "learning_rate": {"tipo": "float", "defecto": 0.05, "min": 0.001, "max": 1.0,
                              "ayuda": "Tasa de aprendizaje"},
            "max_depth": {"tipo": "int", "defecto": 3, "min": 1, "max": 10,
                          "ayuda": "Profundidad de cada árbol"},
        },
    },
}


def cargar_datos(ruta: Path = RUTA_DATOS) -> pd.DataFrame:
    """Lee el CSV original, valida columnas y convierte los ceros imposibles en NaN."""
    df = pd.read_csv(ruta)
    faltan = set(VARIABLES + [OBJETIVO]) - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en el dataset: {sorted(faltan)}")
    df = df[VARIABLES + [OBJETIVO]].drop_duplicates().reset_index(drop=True)
    df[CEROS_COMO_FALTANTES] = df[CEROS_COMO_FALTANTES].replace(0, np.nan)
    return df


def resumen_datos() -> dict[str, Any]:
    original = pd.read_csv(RUTA_DATOS)
    df = cargar_datos()
    return {
        "filas": int(len(df)),
        "columnas": VARIABLES,
        "nombres_es": NOMBRES_ES,
        "clases": {"sin_diabetes": int((df[OBJETIVO] == 0).sum()), "con_diabetes": int((df[OBJETIVO] == 1).sum())},
        "ceros_como_faltantes": {c: int((original[c] == 0).sum()) for c in CEROS_COMO_FALTANTES},
        "descripcion": df[VARIABLES].describe().round(2).to_dict(),
        "muestra": df.head(10).replace({np.nan: None}).to_dict(orient="records"),
    }


def _limpiar_parametros(algoritmo: str, parametros: dict[str, Any] | None) -> dict[str, Any]:
    if algoritmo not in ALGORITMOS:
        raise ValueError(f"Algoritmo desconocido: {algoritmo}. Opciones: {list(ALGORITMOS)}")
    spec = ALGORITMOS[algoritmo]["parametros"]
    parametros = parametros or {}
    desconocidos = set(parametros) - set(spec)
    if desconocidos:
        raise ValueError(f"Parámetros no válidos para {algoritmo}: {sorted(desconocidos)}")
    limpios = {}
    for nombre, s in spec.items():
        valor = parametros.get(nombre, s["defecto"])
        if s["tipo"] == "int":
            valor = int(valor)
        elif s["tipo"] == "float":
            valor = float(valor)
        if s["tipo"] in ("int", "float") and not (s["min"] <= valor <= s["max"]):
            raise ValueError(f"{nombre} debe estar entre {s['min']} y {s['max']}")
        if s["tipo"] == "opcion" and valor not in s["opciones"]:
            raise ValueError(f"{nombre} debe ser uno de {s['opciones']}")
        limpios[nombre] = valor
    return limpios


def construir_pipeline(algoritmo: str, p: dict[str, Any], semilla: int = 42) -> Pipeline:
    cw = None if p.get("class_weight") == "ninguno" else p.get("class_weight")
    if algoritmo == "regresion_logistica":
        modelo = LogisticRegression(C=p["C"], max_iter=p["max_iter"], class_weight=cw, random_state=semilla)
    elif algoritmo == "random_forest":
        modelo = RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"] or None,
                                        min_samples_leaf=p["min_samples_leaf"], class_weight=cw,
                                        random_state=semilla)
    elif algoritmo == "svm":
        # SVC no entrega probabilidades por sí mismo: se calibra con validación cruzada interna
        modelo = CalibratedClassifierCV(SVC(C=p["C"], kernel=p["kernel"], gamma=p["gamma"], class_weight=cw,
                                            random_state=semilla), ensemble=False, cv=5)
    elif algoritmo == "knn":
        modelo = KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"])
    else:
        modelo = GradientBoostingClassifier(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                                            max_depth=p["max_depth"], random_state=semilla)
    return Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),   # rellena los faltantes con la mediana
        ("escalado", StandardScaler()),                      # media 0 y desviación 1
        ("modelo", modelo),
    ])


def entrenar(algoritmo: str = "random_forest", parametros: dict[str, Any] | None = None,
             test_size: float = 0.2, semilla: int = 42, folds_cv: int = 5,
             guardar: bool = True) -> dict[str, Any]:
    """Entrena el pipeline, evalúa en el conjunto de prueba y con validación cruzada, y guarda el modelo."""
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size debe estar entre 0.1 y 0.5")
    if not 2 <= folds_cv <= 10:
        raise ValueError("folds_cv debe estar entre 2 y 10")
    p = _limpiar_parametros(algoritmo, parametros)
    df = cargar_datos()
    X, y = df[VARIABLES], df[OBJETIVO]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=semilla)

    pipe = construir_pipeline(algoritmo, p, semilla)
    cv = cross_validate(construir_pipeline(algoritmo, p, semilla), X_tr, y_tr,
                        cv=StratifiedKFold(folds_cv, shuffle=True, random_state=semilla),
                        scoring=["accuracy", "f1", "roc_auc", "recall"])
    inicio = datetime.now()
    pipe.fit(X_tr, y_tr)
    segundos = (datetime.now() - inicio).total_seconds()

    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_te, proba)
    imp = permutation_importance(pipe, X_te, y_te, n_repeats=10, random_state=semilla, scoring="roc_auc")

    metricas = {
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "precision": round(precision_score(y_te, pred, zero_division=0), 4),
        "recall": round(recall_score(y_te, pred), 4),
        "f1": round(f1_score(y_te, pred), 4),
        "roc_auc": round(roc_auc_score(y_te, proba), 4),
    }
    resultado = {
        "algoritmo": algoritmo,
        "nombre_algoritmo": ALGORITMOS[algoritmo]["nombre"],
        "parametros": p,
        "test_size": test_size,
        "semilla": semilla,
        "folds_cv": folds_cv,
        "n_entrenamiento": int(len(X_tr)),
        "n_prueba": int(len(X_te)),
        "metricas_prueba": metricas,
        "validacion_cruzada": {k.replace("test_", ""): {"media": round(float(v.mean()), 4), "desv": round(float(v.std()), 4)}
                               for k, v in cv.items() if k.startswith("test_")},
        "matriz_confusion": confusion_matrix(y_te, pred).tolist(),
        "curva_roc": {"fpr": np.round(fpr, 4).tolist(), "tpr": np.round(tpr, 4).tolist()},
        "importancia_variables": dict(sorted(
            {v: round(float(m), 4) for v, m in zip(VARIABLES, imp.importances_mean)}.items(),
            key=lambda kv: -kv[1])),
        "segundos_entrenamiento": round(segundos, 3),
        "fecha": datetime.now().isoformat(timespec="seconds"),
    }
    if guardar:
        CARPETA_MODELOS.mkdir(exist_ok=True)
        joblib.dump({"pipeline": pipe, "info": resultado}, RUTA_MODELO)
        historial = leer_historial()
        historial.append({k: resultado[k] for k in ("fecha", "algoritmo", "nombre_algoritmo", "parametros",
                                                    "test_size", "metricas_prueba")})
        RUTA_HISTORIAL.write_text(json.dumps(historial, ensure_ascii=False, indent=2), encoding="utf-8")
    return resultado


def leer_historial() -> list[dict[str, Any]]:
    if RUTA_HISTORIAL.exists():
        return json.loads(RUTA_HISTORIAL.read_text(encoding="utf-8"))
    return []


def cargar_modelo() -> dict[str, Any] | None:
    if RUTA_MODELO.exists():
        return joblib.load(RUTA_MODELO)
    return None


def nivel_riesgo(probabilidad: float) -> str:
    if probabilidad < 0.30:
        return "Bajo"
    if probabilidad < 0.60:
        return "Moderado"
    return "Alto"


def predecir(paciente: dict[str, Any], paquete: dict[str, Any]) -> dict[str, Any]:
    """Predice la probabilidad de diabetes para un caso nuevo (los campos ausentes se imputan)."""
    fila = pd.DataFrame([{v: paciente.get(v) for v in VARIABLES}], dtype=float)
    fila[CEROS_COMO_FALTANTES] = fila[CEROS_COMO_FALTANTES].replace(0, np.nan)
    prob = float(paquete["pipeline"].predict_proba(fila)[0, 1])
    return {
        "probabilidad_diabetes": round(prob, 4),
        "prediccion": int(prob >= 0.5),
        "etiqueta": "Positivo: riesgo de diabetes" if prob >= 0.5 else "Negativo: sin indicios de diabetes",
        "nivel_riesgo": nivel_riesgo(prob),
        "modelo": paquete["info"]["nombre_algoritmo"],
        "aviso": "Herramienta educativa: no reemplaza el diagnóstico de un profesional de la salud.",
    }
