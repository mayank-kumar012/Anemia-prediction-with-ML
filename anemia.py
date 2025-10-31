import streamlit as st
import pandas as pd
import numpy as np
import cv2
import pytesseract
from PIL import Image
import re
import joblib
from sklearn.ensemble import RandomForestClassifier

# ====================== CONFIGURATION ======================
# Set Tesseract executable path (adjust if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load trained model (Pipeline)
model = joblib.load("pipeline_random_forest_retrained.pkl")


# ====================== IMAGE PREPROCESSING ======================
def preprocess_image(image_path):
    """Cleans the input image for better OCR results."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    temp_path = "processed_image.png"
    cv2.imwrite(temp_path, thresh)
    return temp_path


# ====================== OCR EXTRACTION ======================
def extract_cbc_values(image_path):
    """Extracts relevant CBC values from image text using regex."""
    processed_path = preprocess_image(image_path)
    extracted_text = pytesseract.image_to_string(Image.open(processed_path))

    # Regex patterns for typical CBC values
    hb_pattern = re.compile(r"HAEMOGLOBIN.*?([\d\.]+)\s*g", re.IGNORECASE)
    mcv_pattern = re.compile(r"MCV.*?([\d\.]+)\s*f", re.IGNORECASE)
    mch_pattern = re.compile(r"MCH[^C].*?([\d\.]+)\s*p", re.IGNORECASE)
    mchc_pattern = re.compile(r"MCHC.*?([\d\.]+)\s*", re.IGNORECASE)

    # Extract values
    haemoglobin = float(hb_pattern.search(extracted_text).group(1)) if hb_pattern.search(extracted_text) else None
    mcv = float(mcv_pattern.search(extracted_text).group(1)) if mcv_pattern.search(extracted_text) else None
    mch = float(mch_pattern.search(extracted_text).group(1)) if mch_pattern.search(extracted_text) else None
    mchc = float(mchc_pattern.search(extracted_text).group(1)) if mchc_pattern.search(extracted_text) else None

    return extracted_text, haemoglobin, mcv, mch, mchc


# ====================== DATA PREPROCESSING ======================
def preprocess_data(hemoglobin, gender, mcv):
    """Converts user input into model-compatible DataFrame."""
    gender_mapping = {'Male': 0, 'Female': 1}
    gender = gender_mapping.get(gender, 0)
    hemoglobin = max(hemoglobin, 0)
    mcv = max(mcv, 0)
    df = pd.DataFrame({'Gender': [gender], 'Hemoglobin': [hemoglobin], 'MCV': [mcv]})
    return df


# ====================== MODEL PREDICTION ======================
def predict_anemia(hemoglobin, gender, mcv):
    """Predicts anemia using the pre-trained RandomForest model."""
    df = preprocess_data(hemoglobin, gender, mcv)
    prediction = model.predict(df)
    return prediction[0]


# ====================== STREAMLIT APP ======================
def main():
    st.title("📄 Automated Anemia Detection from CBC Report (OCR + ML + Manual Correction)")
    st.write("""
    Upload your CBC report image. The system will extract blood parameters automatically using OCR.
    If some values are missing or incorrect, you can manually correct them before prediction.
    """)

    uploaded_file = st.file_uploader("Upload CBC Report Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save uploaded image temporarily
        with open("uploaded_report.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, caption="Uploaded CBC Report", width="stretch")

        st.write("⏳ Processing and extracting values...")
        extracted_text, haemoglobin, mcv, mch, mchc = extract_cbc_values("uploaded_report.jpg")

        st.subheader("🧾 OCR Extracted Text")
        st.text_area("Extracted Text", extracted_text, height=200)

        st.subheader("🔍 Extracted Values (Editable)")
        st.write("You can correct any missing or inaccurate values before running prediction:")

        # Editable numeric input fields (auto-filled if available)
        haemoglobin = st.number_input("Hemoglobin (g/dL)", value=haemoglobin if haemoglobin else 0.0, min_value=0.0, step=0.1)
        mcv = st.number_input("MCV (fL)", value=mcv if mcv else 0.0, min_value=0.0, step=0.1)
        mch = st.number_input("MCH (pg)", value=mch if mch else 0.0, min_value=0.0, step=0.1)
        mchc = st.number_input("MCHC (g/dL)", value=mchc if mchc else 0.0, min_value=0.0, step=0.1)

        gender = st.radio("Select Gender", ['Male', 'Female'], index=0)

        if st.button("🔎 Predict Anemia"):
            if haemoglobin > 0 and mcv > 0:
                prediction = predict_anemia(haemoglobin, gender, mcv)
                prediction_label = "🩸 **Anemic**" if prediction == 1 else "✅ **Non-Anemic**"
                st.success(f"**Prediction Result:** {prediction_label}")

                # Detailed insights
                if mcv < 80:
                    st.info("🔸 Low MCV suggests *Microcytic Anemia* (possibly Iron Deficiency).")
                elif mcv > 100:
                    st.info("🔸 High MCV suggests *Macrocytic Anemia* (possibly Vitamin B12/Folate Deficiency).")
            else:
                st.warning("⚠️ Please enter valid Hemoglobin and MCV values for accurate prediction.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("⚠️ This app is for educational and informational purposes only. Please consult a qualified doctor for medical diagnosis.")


# ====================== RUN APP ======================
if __name__ == "__main__":
    main()
