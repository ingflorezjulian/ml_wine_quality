# 🍷 Pipeline de ML Automatizado - Predicción de Calidad de Vino

## Autor

**Julián David Flórez Sánchez**
- GitHub: [@ingflorezjulian](https://github.com/ingflorezjulian)
- Email: ingflorezjulian@gmail.com

## Dataset Seleccionado

### Wine Quality Dataset (UCI Machine Learning Repository)

**Fuente:** [UCI ML Repository - Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)
**url:** "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

**Justificación de selección:**

1. **Relevancia práctica**: El dataset permite predecir la calidad del vino basándose en características fisicoquímicas, un problema real en la industria vinícola.

2. **Calidad de datos**: 
   - Dataset bien estructurado con 1,599 muestras
   - 11 features numéricas (acidez, pH, alcohol, sulfatos, etc.)
   - Target numérico (calidad de 0-10) convertido a clasificación binaria

3. **Características del problema**:
   - Problema de clasificación supervisada
   - Desbalanceo moderado que permite explorar técnicas de ML
   - Features continuas que requieren preprocesamiento

4. **Disponibilidad**: Dataset de acceso público bajo licencia Creative Commons, cumpliendo los requisitos del proyecto.

**Estadísticas del dataset:**
- Filas: 1,599 vinos tintos
- Columnas: 12 (11 features + 1 target)
- Tipo: CSV con separador ';'
- Target: Calidad (3-8), convertida a binaria (≥6 = bueno, <6 = malo)

---

## Instalación

### Prerrequisitos
- Python 3.9+
- pip
- Git

### Pasos de instalación
```bash
# Clonar repositorio
git clone https://github.com/TU-USUARIO/ml-wine-quality.git
cd ml-wine-quality

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
make install
```

---

## Uso

### Entrenar el modelo
```bash
make train
```

### Ejecutar tests
```bash
make test
```

### Verificar calidad de código
```bash
make lint
```

### Limpiar archivos temporales
```bash
make clean
```

---

## Arquitectura del Proyecto
```
ml-wine-quality/
├── .github/
│   └── workflows/
│       └── ml.yml              # Pipeline CI/CD
├── src/
│   ├── train.py                # Pipeline principal
│   ├── preprocessing.py        # Preprocesamiento de datos
│   └── evaluation.py           # Evaluación del modelo
├── tests/
│   └── test_pipeline.py       # Tests unitarios
├── mlruns/                     # Tracking MLflow (generado)
├── config.yaml                 # Configuración del pipeline
├── requirements.txt            # Dependencias Python
├── Makefile                    # Automatización de tareas
└── README.md                   # Documentación
```

---

## Pipeline de Machine Learning

### 1. Preprocesamiento
- **Limpieza**: Eliminación de valores nulos
- **Transformación**: Conversión de target a clasificación binaria (≥6 = bueno)
- **División**: 80% entrenamiento, 20% prueba (stratified)
- **Escalamiento**: StandardScaler para normalización

### 2. Modelo
- **Algoritmo**: Random Forest Classifier
- **Hiperparámetros**:
  - n_estimators: 100
  - max_depth: 10
  - random_state: 42

### 3. Evaluación
Se calculan 4 métricas principales:
- **Accuracy**: Precisión general del modelo
- **F1-Score**: Media armónica entre precision y recall
- **Precision**: Proporción de positivos correctos
- **Recall**: Capacidad de detectar positivos reales

---

## 📈 Resultados

### Métricas de rendimiento

| Métrica    | Valor  |
|------------|--------|
| Accuracy   | ~0.79  |
| F1-Score   | ~0.81  |
| Precision  | ~0.81  |
| Recall     | ~0.81  |

### Visualización en MLflow

Para ver los experimentos y modelos registrados:
```bash
mlflow ui
```

Navega a: `http://localhost:5000`

**Características registradas en MLflow:**
- Parámetros del modelo (n_estimators, max_depth, random_state)
- Métricas de evaluación (4 métricas)
- Modelo serializado como artefacto
- Signature del modelo (tipos de entrada/salida)
- Input example (primeras 5 filas de test)
- Modelo registrado en Model Registry

---

## CI/CD con GitHub Actions

### Flujo automatizado

El pipeline se ejecuta automáticamente en cada:
- Push a `main`
- Pull request a `main`
- Ejecución manual (workflow_dispatch)

### Etapas del pipeline

1. **Setup**: Configuración de Python 3.9
2. **Install**: Instalación de dependencias (`make install`)
3. **Lint**: Verificación de estilo de código (`make lint`)
4. **Test**: Ejecución de tests unitarios (`make test`)
5. **Train**: Entrenamiento del modelo (`make train`)

### Ver ejecuciones

Visita la pestaña "Actions" en tu repositorio de GitHub:
`https://github.com/ingflorezjulian/ml_wine_quality/actions/`

---

## Tests

El proyecto incluye tests automatizados:

- `test_preprocess_data`: Verifica el preprocesamiento completo
- `test_data_split`: Valida la división correcta de datos
```bash
# Ejecutar tests con coverage
pytest tests/ -v --cov=src
```

---

## Configuración

Modifica `config.yaml` para ajustar el pipeline:
```yaml
data:
  url: "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
  test_size: 0.2
  random_state: 42

model:
  type: "RandomForest"
  params:
    n_estimators: 100
    max_depth: 10
    random_state: 42

mlflow:
  experiment_name: "wine-quality-experiment"
  tracking_uri: "file:./mlruns"
```

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.9+**: Lenguaje principal
- **scikit-learn**: Machine Learning
- **MLflow**: Tracking y registro de modelos
- **pandas/numpy**: Manipulación de datos
- **pytest**: Testing
- **flake8**: Linting
- **GitHub Actions**: CI/CD

---

## 📦 Dependencias

Ver `requirements.txt` para lista completa:
```txt
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.3.0
mlflow>=2.9.0
pyyaml>=6.0.1
pytest>=7.4.0
flake8>=6.1.0
```

---

## Licencia

Este proyecto es parte de un trabajo académico para el curso de MLOps.

---
