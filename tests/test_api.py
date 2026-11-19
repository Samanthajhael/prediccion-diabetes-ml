"""Pruebas automáticas de la API. Ejecutar con: python -m pytest -q"""
import pytest
from fastapi.testclient import TestClient

from backend import ml
from backend.main import app

CASO = {"Pregnancies": 2, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 120,
        "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50}


@pytest.fixture(scope="module")
def cliente(tmp_path_factory):
    # Las pruebas guardan modelos en una carpeta temporal para no tocar el modelo real
    carpeta = tmp_path_factory.mktemp("modelos")
    originales = (ml.CARPETA_MODELOS, ml.RUTA_MODELO, ml.RUTA_HISTORIAL)
    ml.CARPETA_MODELOS, ml.RUTA_MODELO, ml.RUTA_HISTORIAL = (
        carpeta, carpeta / "modelo_actual.joblib", carpeta / "historial.json")
    with TestClient(app) as c:
        yield c
    ml.CARPETA_MODELOS, ml.RUTA_MODELO, ml.RUTA_HISTORIAL = originales


def test_estado(cliente):
    r = cliente.get("/")
    assert r.status_code == 200 and r.json()["modelo_cargado"] is True


def test_resumen_dataset(cliente):
    r = cliente.get("/dataset/resumen").json()
    assert r["filas"] == 768
    assert r["clases"] == {"sin_diabetes": 500, "con_diabetes": 268}


@pytest.mark.parametrize("algoritmo", ["regresion_logistica", "random_forest", "svm", "knn", "gradient_boosting"])
def test_entrenar_cada_algoritmo(cliente, algoritmo):
    r = cliente.post("/entrenar", json={"algoritmo": algoritmo, "folds_cv": 3})
    assert r.status_code == 200, r.text
    m = r.json()["metricas_prueba"]
    assert 0.6 < m["roc_auc"] <= 1.0
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_parametros_configurables(cliente):
    r = cliente.post("/entrenar", json={"algoritmo": "random_forest",
                                         "parametros": {"n_estimators": 50, "max_depth": 4}, "test_size": 0.3})
    assert r.status_code == 200
    assert r.json()["parametros"]["n_estimators"] == 50
    assert r.json()["n_prueba"] == 231


def test_parametro_invalido(cliente):
    r = cliente.post("/entrenar", json={"algoritmo": "knn", "parametros": {"n_neighbors": 0}})
    assert r.status_code == 422


def test_predecir_caso_nuevo(cliente):
    r = cliente.post("/predecir", json=CASO)
    assert r.status_code == 200
    res = r.json()
    assert 0 <= res["probabilidad_diabetes"] <= 1
    assert res["nivel_riesgo"] in {"Bajo", "Moderado", "Alto"}


def test_predecir_campos_opcionales(cliente):
    caso = {k: v for k, v in CASO.items() if k not in ("SkinThickness", "Insulin")}
    assert cliente.post("/predecir", json=caso).status_code == 200


def test_predecir_valor_fuera_de_rango(cliente):
    assert cliente.post("/predecir", json={**CASO, "Glucose": -5}).status_code == 422
