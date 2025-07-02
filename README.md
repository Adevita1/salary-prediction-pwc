# 💼 Salary Prediction Challenge - PwC

Este proyecto aborda un desafío de ciencia de datos para predecir salarios utilizando características como edad, género, educación, experiencia, título laboral y descripción del puesto.

Se realizaron tareas de análisis exploratorio, ingeniería de features, entrenamiento de modelos de regresión y evaluación de rendimiento, siguiendo las mejores prácticas de código modular y notebook explicativa.

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


📊 Resultados y Métricas
Se probaron varios modelos de regresión (Lineal, Random Forest, etc.), y se aplicó:

One-Hot Encoding de variables categóricas.

Limpieza y validación de datos.

Incorporación de nuevas features:

Title Length: longitud del campo Job Title.

Modelo final
Modelo: Random Forest Regressor

R² Score: 0.92

MSE: Aproximadamente 200M

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

👤 Autor
Adrián Nicolás
Data Engineer & Científico de Datos
🔗 https://www.linkedin.com/in/adevita1/
💻 https://github.com/Adevita1