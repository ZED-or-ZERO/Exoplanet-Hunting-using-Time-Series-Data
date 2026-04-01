# Exoplanet Detection: Time Series Classification
---
## 📌 Project Overview
Этот проект посвящен обнаружению экзопланет с использованием данных транзитной фотометрии (Transit Photometry). Главная цель исследования — сравнить классический подход машинного обучения (**Manual Feature Engineering**) и современные нейросетевые методы (**End-to-End Deep Learning**) на данных с экстремально низким соотношением сигнал/шум (*Low Signal-to-Noise Ratio*).

## 🚀 Key Insights & Results
В ходе экспериментов было доказано, что для зашумленных астрономических временных рядов (где падение яркости от транзита планеты составляет ~0.01%):
* **Classic ML (Random Forest + FFT)** показал выдающиеся результаты (**Recall = 0.93**, **Accuracy = 0.94**). Перевод сигнала из временной области (*Time Domain*) в частотную (*Frequency Domain*) через преобразование Фурье позволил эффективно изолировать паттерны экзопланет.
* **Deep Learning (1D-CNN + Time Domain)** показал низкую производительность (эффект случайного угадывания, **Recall = ~0.51**). Несмотря на применение *Savitzky-Golay Detrending*, *Batch Normalization* и *Z-Score Normalization*, архитектуре не хватило контекста для выделения микроскопических транзитов на фоне звездной переменности без математических подсказок.
* **Вывод:** Правильный *Signal Processing* (математическая обработка сигналов) оказался эффективнее сложной архитектуры нейросети.

## 🗄️ Dataset
* **Source:** Данные получены из Kaggle-датасета [Exoplanet Detection Dataset](https://www.kaggle.com/datasets/ronaldkroening/exoplanet-detection-dataset).
* **Preprocessed Data:** Полный архив с очищенными данными, признаками и весами моделей доступен на [Google Drive](https://drive.google.com/drive/folders/1Bq1bLAs5mcZg8LKSCG_P3ACppRvxIGIx?usp=sharing).

## 🛠️ Data Pipeline Architecture

1. **Data Preprocessing (`01_data_cleaning.ipynb`):**
   - *Linear Interpolation* для заполнения пропущенных значений (NaNs).
   - Выравнивание массивов временных рядов до единой длины (4608 шагов).
2. **Feature Engineering (`02_feature_engineering_fft.ipynb`):**
   - Устранение низкочастотной звездной переменности с помощью *Savitzky-Golay Filter* (Detrending).
   - Применение *Fast Fourier Transform (FFT)* для генерации частотных признаков (Power Spectrum).
3. **Modeling & Evaluation (`03_baseline_model.ipynb` & `04_deep_learning_cnn.ipynb`):**
   - Обучение `RandomForestClassifier` на извлеченных FFT-признаках.
   - Разработка архитектуры `1D-CNN` с использованием PyTorch для анализа сырых/detrended временных рядов.
   - Сравнение моделей с помощью *ROC-AUC* и *Confusion Matrix*.

## 📂 Project Structure
```text
├── data/
│   ├── raw/             # Исходные файлы
│   └── processed/       # Очищенные временные ряды и FFT-признаки
├── models/              # Сохраненные веса (например, rf_baseline.pkl)
├── notebooks/           # Jupyter notebooks с экспериментами (01-04)
├── src/                 # Модули с Python-скриптами
│   ├── preprocessing.py # Логика интерполяции и detrending
│   └── features.py      # Логика извлечения FFT
├── train.py             # Entry point для обучения Baseline-модели
└── README.md

--- 
# How to Run the Project

1. Clone the repository

`git clone [https://github.com/ZED-or-ZERO/Exoplanet_Hunting.git](https://github.com/ZED-or-ZERO/Exoplanet_Hunting.git)`
`cd Exoplanet_Hunting`

2. Install dependencies

 ! It's not ready yet, but it requires all the basic tools for AI, if you're working with AI, then you don't really need anything.
`pip install -r requirements.txt`

3. Start training the basic model

**bash:**
`python train.py`


