# 🩸 Automated Anemia Detection using OCR + Machine Learning

This project is an **AI-powered medical analyzer** that automatically detects **Anemia** from **CBC report images**.

It combines:
- 🧠 Machine Learning (Random Forest)
- 📄 Optical Character Recognition (OCR) using Tesseract
- 🌐 Streamlit Web App Interface

Users can upload their CBC report, and the app will:
1. Extract blood parameters (Hemoglobin, MCV, MCH, MCHC) using OCR.
2. Predict whether the patient is **Anemic** or **Non-Anemic** using a trained model.
3. Provide insights into anemia type (Microcytic / Macrocytic).


## 📊 Tech Stack Summary

| Component | Technology Used |
|:---|:---|
| **OCR** | Tesseract OCR ([Tesseract OCR](https://github.com/tesseract-ocr/tesseract)) |
| **Model** | RandomForestClassifier (scikit-learn) |
| **Frontend** | Streamlit |
| **Backend** | Python |
| **Deployment** | Render (Free Hosting) |


## 🚀 Features

✅ OCR-based value extraction from uploaded CBC report images  
✅ Manual correction fields for missing or inaccurate values  
✅ Random Forest classifier trained on medical dataset  
✅ Real-time anemia prediction and interpretation  
✅ Web interface powered by Streamlit  
✅ Deployable on Render or any cloud platform  

