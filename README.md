# Sistema de Predicción de Riesgo de Diabetes con Machine Learning

Proyecto Final · **Proyectos de Machine Learning** · Instituto Superior Tecnológico CENESTUR
Tecnología Superior en Big Data e Inteligencia de Negocios · Docente: Ing. Sergio Granizo, MSc. · Período 2026-2

Aplicación en Python con **backend** (API REST en FastAPI) y **frontend** (interfaz gráfica en Streamlit). La aplicación:

- **Entrena** modelos de clasificación: Regresión Logística, Random Forest, SVM, KNN y Gradient Boosting.
- **Configura** los hiperparámetros, la proporción de prueba, la semilla y la validación cruzada.
- **Muestra métricas**: exactitud, precisión, sensibilidad, F1, ROC AUC, matriz de confusión, curva ROC, importancia de variables e historial.
- **Recibe un caso nuevo** y devuelve la probabilidad de diabetes y el nivel de riesgo.

> Herramienta educativa: no reemplaza el diagnóstico de un profesional de la salud.

## Dataset (real)

**Pima Indians Diabetes Database**: 768 mujeres de ascendencia pima mayores de 21 años, 8 variables clínicas y la variable objetivo `Outcome` (1 = diabetes). Lo recopiló el National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).

- Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Annual Symposium on Computer Application in Medical Care*, 261–265. https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/
- Sigillito, V. (1990). *Pima Indians Diabetes Database* [Conjunto de datos; propietario original: NIDDK]. OpenML, dataset 37. https://www.openml.org/d/37

**Limpieza:** en `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` y `BMI` un valor de 0 es fisiológicamente imposible y significa «dato no registrado». Esos ceros se convierten en faltantes y se imputan con la mediana dentro del pipeline, lo que evita fuga de datos entre entrenamiento y prueba.

## Estructura

```
proyecto_final_ml/
├── backend/
│   ├── ml.py            # datos, limpieza, pipelines, entrenamiento, métricas, predicción
│   └── main.py          # API REST (FastAPI)
├── frontend/
│   └── app.py           # interfaz gráfica (Streamlit) que consume la API
├── data/diabetes.csv    # dataset original
├── modelos/             # modelo entrenado (.joblib) e historial (se generan al ejecutar)
├── tests/test_api.py    # pruebas automáticas (pytest)
├── .streamlit/config.toml
├── requirements.txt
└── iniciar_app.bat      # inicia backend y frontend en Windows
```

## Instalación y ejecución

```bash
python -m venv .venv
.venv\Scripts\activate          # en Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1: backend
uvicorn backend.main:app --port 8000
# Terminal 2: frontend
streamlit run frontend/app.py
```

En Windows también se puede hacer doble clic en `iniciar_app.bat`.

- Interfaz: http://localhost:8501
- API y documentación interactiva (Swagger): http://127.0.0.1:8000/docs

## Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Estado de la API y del modelo activo |
| GET | `/algoritmos` | Algoritmos disponibles y sus hiperparámetros |
| GET | `/dataset/resumen` | Resumen del dataset (clases, faltantes, estadística) |
| POST | `/entrenar` | Entrena con el algoritmo y los parámetros indicados; el resultado pasa a ser el modelo activo |
| GET | `/metricas` | Métricas del modelo activo |
| GET | `/historial` | Historial de entrenamientos |
| POST | `/predecir` | Predice el riesgo de diabetes de un caso nuevo |

Ejemplo:

```bash
curl -X POST http://127.0.0.1:8000/predecir -H "Content-Type: application/json" -d "{\"Pregnancies\":2,\"Glucose\":148,\"BloodPressure\":72,\"SkinThickness\":35,\"Insulin\":120,\"BMI\":33.6,\"DiabetesPedigreeFunction\":0.627,\"Age\":50}"
```

## Pruebas

```bash
python -m pytest -q
```

## Trabajo en equipo (GitHub)

Cada integrante debe clonar el repositorio público y subir **commits progresivos desde su propia cuenta**, por ejemplo con ramas por funcionalidad (`feature/frontend-metricas`, `feature/api-prediccion`) y *pull requests*.
