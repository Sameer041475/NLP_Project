import streamlit as st
import pickle

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Resume Category Predictor",
    page_icon="📄",
    layout="centered"
)

# --------------------------------------------------
# Load Model and TF-IDF Vectorizer
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


model, vectorizer = load_model()

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("📄 Resume Category Predictor")
st.write("Enter your resume text below and the AI model will predict the suitable job category.")

st.divider()

resume_text = st.text_area(
    "Enter Resume Text",
    height=300,
    placeholder="Example: Python, Machine Learning, TensorFlow, Pandas, SQL..."
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Predict Category", use_container_width=True):

    if resume_text.strip() == "":
        st.warning("Please enter some resume text.")

    else:
        # Convert text into TF-IDF
        text_vector = vectorizer.transform([resume_text])

        # Prediction
        prediction = model.predict(text_vector)

        predicted_category = prediction[0]

        st.success(f"### Predicted Category: {predicted_category}")

        st.divider()

        st.info(
            "The prediction was generated using a TF-IDF + LinearSVC machine learning model."
        )