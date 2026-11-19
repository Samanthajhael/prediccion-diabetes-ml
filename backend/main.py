"""API REST del sistema de predicción de diabetes (FastAPI).

Ejecutar desde la carpeta del proyecto:
    uvicorn backend.main:app --reload
Documentación interactiva: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend import ml

ESTADO: dict[str, Any] = {"paquete": None}


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    # Al iniciar: carga el último modelo guardado o entrena uno por defecto
    ESTADO["paquete"] = ml.cargar_modelo()
    if ESTADO["paquete"] is None:
        ml.entrenar()
        ESTADO["paquete"] = ml.cargar_modelo()
    yield


app = FastAPI(
    title="API de Predicción de Diabetes",
    description="Entrena modelos de Machine Learning con el dataset Pima Indians Diabetes y predice el riesgo "
                "de diabetes para casos nuevos. Proyecto Final - Proyectos de Machine Learning (CENESTUR).",
    version="1.0.0",
    lifespan=ciclo_de_vida,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SolicitudEntrenamiento(BaseModel):
    algoritmo: str = Field("random_forest", description="Clave del algoritmo (ver GET /algoritmos)")
    parametros: dict[str, Any] = Field(default_factory=dict, description="Hiperparámetros del algoritmo")
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Proporción de datos para prueba")
    semilla: int = Field(42, ge=0, le=10_000, description="Semilla aleatoria (reproducibilidad)")
    folds_cv: int = Field(5, ge=2, le=10, description="Número de particiones de la validación cruzada")


class Paciente(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20, description="Número de embarazos")
    Glucose: float = Field(..., gt=0, le=300, description="Glucosa plasmática a las 2 h (mg/dL)")
    BloodPressure: float = Field(..., gt=0, le=200, description="Presión arterial diastólica (mm Hg)")
    SkinThickness: Optional[float] = Field(None, ge=0, le=100, description="Pliegue cutáneo del tríceps (mm); opcional")
    Insulin: Optional[float] = Field(None, ge=0, le=900, description="Insulina sérica a las 2 h (µU/mL); opcional")
    BMI: float = Field(..., gt=0, le=80, description="Índice de masa corporal (kg/m²)")
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3, description="Función de pedigrí de diabetes")
    Age: int = Field(..., ge=21, le=120, description="Edad en años (el dataset incluye personas de 21 años o más)")

    model_config = {"json_schema_extra": {"example": {
        "Pregnancies": 2, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 120,
        "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50}}}


def _paquete() -> dict[str, Any]:
    if ESTADO["paquete"] is None:
        raise HTTPException(status_code=409, detail="No hay un modelo entrenado. Use POST /entrenar.")
    return ESTADO["paquete"]


@app.get("/", tags=["Estado"])
def estado():
    info = ESTADO["paquete"]["info"] if ESTADO["paquete"] else None
    return {"estado": "ok", "modelo_cargado": info is not None,
            "modelo": info["nombre_algoritmo"] if info else None,
            "entrenado_en": info["fecha"] if info else None}


@app.get("/algoritmos", tags=["Configuración"])
def algoritmos():
    return ml.ALGORITMOS


@app.get("/dataset/resumen", tags=["Datos"])
def dataset_resumen():
    return ml.resumen_datos()


@app.post("/entrenar", tags=["Entrenamiento"])
def entrenar(solicitud: SolicitudEntrenamiento):
    try:
        resultado = ml.entrenar(solicitud.algoritmo, solicitud.parametros, solicitud.test_size,
                                solicitud.semilla, solicitud.folds_cv)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    ESTADO["paquete"] = ml.cargar_modelo()
    return resultado


@app.get("/metricas", tags=["Entrenamiento"])
def metricas():
    return _paquete()["info"]


@app.get("/historial", tags=["Entrenamiento"])
def historial():
    return ml.leer_historial()


@app.post("/predecir", tags=["Predicción"])
def predecir(paciente: Paciente):
    return ml.predecir(paciente.model_dump(), _paquete())
