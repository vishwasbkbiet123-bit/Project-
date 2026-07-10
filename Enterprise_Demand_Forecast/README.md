# 📈 Enterprise Demand Forecasting System 🚀

## 📝 Overview
An end-to-end Machine Learning web application designed to predict future product demand for retail e-commerce. This project optimizes supply chain operations and minimizes inventory costs by combining historical time-series data with Natural Language Processing (NLP) for seasonal market sentiment.

## ✨ Key Features
* **Dynamic Time-Series Analysis:** Extracts time-based features (Month, Day of Week, Weekends) for accurate forecasting.
* **NLP-Powered Seasonality:** Uses NLTK to parse product descriptions and automatically flag seasonal items (e.g., Christmas, Valentine's) which usually experience sudden demand spikes.
* **Robust ML Engine:** Powered by a Random Forest Regressor, evaluated using MAE and RMSE metrics to ensure enterprise-grade accuracy.
* **Interactive UI:** A multi-page, dark-themed responsive web dashboard built with Flask, HTML5, and CSS3. Features dynamic dropdowns directly hydrated from the ML model's label encoder.

## 🛠️ Tech Stack
* **Data Engineering & EDA:** Python, Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (Random Forest Regressor)
* **Natural Language Processing:** NLTK (Text processing & tokenization)
* **Backend Framework:** Flask
* **Frontend:** HTML5, CSS3, Jinja2 Templating

## 📂 Project Structure
```text
Enterprise_Demand_Forecast/
│
├── data/                  # Contains raw and feature-engineered CSV datasets
├── notebooks/             # Jupyter notebooks (01_EDA, 02_Feature_Engineering, 03_Model)
├── models/                # Trained ML models (.pkl files)
├── static/                # CSS styling files
├── templates/             # HTML templates (index.html, result.html)
├── app.py                 # Main Flask server application
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation



🚀 How to Run Locally
1. Clone the repository
cd Enterprise-Demand-Forecasting



2. Install dependencies

Bash
pip install -r requirements.txt
3. Run the Flask Server

Bash
python app.py
4. Access the Dashboard
Open your web browser and navigate to: http://127.0.0.1:5000

🔮 Future Scope
Integration of OpenCV to analyze live CCTV camera feeds for real-time store footfall tracking, directly feeding into the demand forecasting pipeline.

Transitioning from Random Forest to Deep Learning architectures like LSTMs for complex long-term sequence predictions.


