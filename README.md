# 💼 Salary Prediction Challenge - PwC

🧾 Descripción del Proyecto
Este proyecto fue desarrollado como parte de un desafío técnico de PwC orientado a la predicción de salarios utilizando técnicas de ciencia de datos. Se trabajó con datasets estructurados que incluían información demográfica, laboral y descripciones textuales de los puestos. El objetivo fue construir un modelo robusto y replicable para estimar salarios en USD a partir de múltiples variables.

Durante el desarrollo se aplicaron buenas prácticas como:

Limpieza de datos y validación.

Codificación de variables categóricas.

Generación de features nuevas, como longitud del título.

Vectorización de texto (TF-IDF) a partir de la descripción de cada puesto.

Entrenamiento y evaluación de modelos supervisados, destacando el Random Forest como el modelo de mejor desempeño.

Modularización completa del código (src/) y uso de un notebook (notebooks/exploration.ipynb) como evidencia visual y explicativa.

El resultado fue un modelo con R² ≈ 0.87, capaz de capturar la mayoría de la varianza del salario, y con buen desempeño general.
---

## 📁 Estructura del Proyecto

```bash
salary-prediction-pwc/
├── notebooks/                  # Exploración y análisis inicial
│   └── exploration.ipynb
├── src/                        # Código modular
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── inference.py
│   ├── save_model.py
│   └── main.py
├── data/                       # Directorio para datasets locales
│   ├── salary.csv
│   ├── people.csv
│   └── descriptions.csv
├── requirements.txt            # Requisitos mínimos para ejecutar
├── requirements_full.txt       # Requisitos completos del entorno
├── .gitignore                  # Archivos a excluir en Git
├── README.md                   # Este archivo
└── Data Scientist - Challenge.pdf  # Enunciado original del desafío


🚀 Cómo ejecutar el proyecto
1. Clonar el repositorio

git clone https://github.com/Adevita1/salary-prediction-pwc.git
cd salary-prediction-pwc


2. Crear un entorno virtual (recomendado)

python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate


3. Instalar las dependencias

pip install -r requirements.txt



4. Ejecutar desde el script principal

# En PowerShell (Windows)
$env:PYTHONPATH="."; python src/main.py

# En Linux/macOS
PYTHONPATH=. python src/main.py

🧠 Inference de prueba
Se integró una predicción simulada con una fila nueva (no vista) con las mismas transformaciones aplicadas durante el entrenamiento, asegurando compatibilidad de columnas y estructura.

Ejemplo de salida:

💡 Predicción salarial estimada: 35800.0


📚 Herramientas utilizadas
Python, Pandas, NumPy

Scikit-learn, Seaborn, Matplotlib

Jupyter Notebook, VS Code, Git, GitHub

📦 Requisitos del sistema
Incluidos en requirements.txt (mínimos) y requirements_full.txt (completo del entorno local de desarrollo).

🤖 Técnicas de Machine Learning utilizadas
Este proyecto empleó técnicas de aprendizaje supervisado para resolver un problema de regresión (predicción de salarios a partir de variables demográficas y laborales).

🧠 Modelos utilizados
Linear Regression
Utilizado como modelo base simple para establecer una línea de comparación.

Dummy Regressor
Baseline que predice siempre la media del salario, útil para validar si los modelos agregan valor real.

Random Forest Regressor (modelo final)
Modelo de ensamble basado en múltiples árboles de decisión. Captura relaciones no lineales, es robusto frente a outliers y maneja bien datos mixtos. Mostró el mejor rendimiento general.

🧰 Técnicas de Preprocesamiento y Feature Engineering
One-Hot Encoding
Conversión de variables categóricas (como género o educación) en variables binarias.

TF-IDF Vectorization
Vectorización del campo description (texto libre) para capturar información semántica relevante para el salario.

Creación de nuevas features

title_length: longitud del título del puesto como proxy de nivel de responsabilidad.

📏 Evaluación y Validación
Cross-Validation
Validación cruzada para medir la robustez del modelo en distintos splits.

Bootstrap Confidence Intervals
Estimación de intervalos de confianza para métricas como R², MAE y MSE, aportando una medida de incertidumbre y estabilidad.

R² Score: 0.92
MSE: Aproximadamente 200M


✅ Conclusiones y Mejoras Futuras
Este proyecto permitió construir un modelo robusto para predecir salarios utilizando variables demográficas, laborales y texto libre. Se combinaron técnicas de ingeniería de features, codificación categórica, vectorización TF-IDF y modelos de regresión.

📊 Resultados clave:

R² ≈ 0.87: el modelo explica gran parte de la variabilidad del salario.

Las variables más importantes fueron edad, experiencia laboral y términos clave en la descripción del puesto.

🚀 Próximos pasos sugeridos:

Incorporar modelos de NLP más avanzados (como embeddings o transformers).

Realizar ajuste de hiperparámetros.

Implementar validación cruzada para mayor robustez.

Explorar más fuentes de datos y enriquecer las variables.


👤 Autor
Adrián Nicolás
Data Engineer & Científico de Datos
🔗 https://www.linkedin.com/in/adevita1/
💻 https://github.com/Adevita1
