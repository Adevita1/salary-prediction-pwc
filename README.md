salary-prediction-pwc/
├── notebooks/ # Exploración y experimentación inicial
│ └── exploration.ipynb
├── src/ # Código modular
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── model_training.py
│ ├── model_evaluation.py
│ ├── inference.py
│ ├── save_model.py
│ ├── utils.py
│ └── init.py
├── data/ # Directorio para datasets locales (vacío)
├── requirements.txt # Requisitos mínimos
├── requirements_full.txt # Todos los paquetes de desarrollo usados
├── README.md
└── Data Scientist - Challenge (1) 1.pdf

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

python main.py

5. Alternativamente, podés explorar la notebook:

jupyter notebook notebooks/exploration.ipynb


📊 Resultados y Métricas
Se probaron varios modelos de regresión (Lineal, Random Forest, etc.), y se aplicó:

One-Hot Encoding de variables categóricas.

Limpieza y validación de datos.

Incorporación de nuevas features:

Description_Length: longitud del campo Description.

Sentiment_Polarity: análisis de polaridad semántica del texto.

Modelo final
Modelo: Random Forest Regressor

R² Score: 0.93

MSE: Aproximadamente 189M

📚 Herramientas utilizadas
Python, Pandas, NumPy

Scikit-learn, Seaborn, Matplotlib

TextBlob (para análisis de sentimiento)

Optuna (para tuning opcional)

Jupyter Notebook, VS Code, Git, GitHub

📦 Requisitos del sistema
Incluidos en requirements.txt (mínimos) y requirements_full.txt (completo del entorno local de desarrollo).

👤 Autor
Adrián Nicolás
Data Engineer & Científico de Datos
🔗 [LinkedIn](https://www.linkedin.com/in/adevita1/)
💻 [GitHub](https://github.com/Adevita1)