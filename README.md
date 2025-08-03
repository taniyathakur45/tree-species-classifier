# 🌳 Tree Species Classification using Machine Learning

Welcome to my project on classifying tree species using structured data and neural networks!  
This prototype was developed as part of the **AI-ML Virtual Internship** by **Edunet Foundation & Microsoft**, with a vision to blend **ecology and AI** 🌿🤖

---

## 📌 Project Overview

Identifying tree species manually is time-consuming and error-prone, especially across large datasets.  
This project uses AI to predict a tree's species based on structured input features like scientific name, condition, and geographical location (latitude & longitude).  
The goal is to assist **botanists, researchers, and city planners** with a quick, intelligent identification system.

---

## 🧠 Model Summary

- **Model Type:** Feedforward Neural Network (Dense Layers)
- **Framework:** TensorFlow + Keras
- **Accuracy Achieved:** ⭐ **43.71%**
- **Data Input:** Tabular (CSV) — no images required
- **Trained on:** Google Colab
- **Output:** Predicted common tree species name 🌲

---

## 🗂️ Dataset Details

- **Filename:** `tree_species_dataset.csv`
- **Source:** Cleaned version of *Albuquerque_Final_2022-06-18.csv*
- **Features Used:**
  - `scientific_name`
  - `condition`
  - `latitude_coordinate`
  - `longitude_coordinate`
- **Target Label:** `common_name`

---

## 🛠️ Tools & Technologies

| Tool/Tech      | Use Case                        |
|----------------|----------------------------------|
| Python         | Programming Language             |
| Pandas, NumPy  | Data handling & preprocessing    |
| TensorFlow     | Neural Network training          |
| Scikit-learn   | Encoding, scaling, splitting     |
| Google Colab   | Cloud-based training platform    |

---

## 📊 Results

- **Training Epochs:** 10  
- **Final Test Accuracy:** `43.71%`  
- **Evaluation Metrics:** Accuracy, Confusion Matrix, Classification Report  
- **Model File:** [`tree_species_model.h5`](tree_species_model.h5)

---

## 🔮 Future Scope

- Integrate image-based classification (leaf/bark photos) using CNNs 📸
- Deploy as a web/mobile app for real-time field use 🌐
- Add GPS auto-fill & voice input for smart forest patrol assistants

---

## 📸 Sample Outputs

> ✅ Check the project PPT for screenshots of model performance, metrics, and code implementation!  
> 🎯 Visuals include accuracy plots, confusion matrix, and predictions.

---

## 🚀 Getting Started

```python
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

model = load_model("tree_species_model.h5")
input_data = np.array([[12, 1, 35.06, -106.65]])  # [scientific_name, condition, lat, long]
input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)
