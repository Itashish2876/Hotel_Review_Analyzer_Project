# ================================================================
# 📦 Imports
# ================================================================

import streamlit as st
import joblib
import re
from io import BytesIO
import PyPDF2
import docx
import pycountry
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder






# ================================================================
# ⚙️ Page Config
# ================================================================


st.set_page_config(page_title="Hotel Sentiment App", page_icon="🏨", layout="centered")





# ================================================================
# 🎨 Custom CSS
# ================================================================


st.markdown(
    '''
    <style>
    .stButton>button {
        background-color: #3dba27;
        color: white;
        padding: 8px 16px;
        border-radius: 5px;
    }
    </style>
    ''', unsafe_allow_html=True
)
st.markdown('<div class="main">', unsafe_allow_html=True)





# ================================================================
# 🧠 Load Models
# ================================================================


try:
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("logistic_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    st.error("❌ Model files not found.")
    st.stop()





# ================================================================
# 🧹 Text Cleaning Function
# ================================================================




def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text





# ================================================================
# 🔮 Sentiment Prediction Function
# ================================================================



def predict_sentiment(text):
    cleaned = clean_text(text)
    transformed = tfidf.transform([cleaned])
    prediction = model.predict(transformed)[0]

    label_map = {
        0: ("Negative Review", "❌", "red"),
        1: ("Neutral Review", "😐", "orange"),
        2: ("Positive Review", "✅", "green")
    }
    return label_map[prediction]






# ================================================================
# 🌍 Country Detection Preparation
# ================================================================


country_list = [country.name.lower() for country in pycountry.countries]






# ================================================================
# 🌍 Country Detection Function
# ================================================================



def detect_country(text):
    text_lower = text.lower()
    for country in country_list:
        if country in text_lower:
            return country.title()
    return None





# ================================================================
# 📂 File Handling Functions
# ================================================================




def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(BytesIO(file.read()))
    text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text.strip()

def extract_text_from_txt(file):
    return file.read().decode('utf-8').strip()

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs]).strip()






# ================================================================
# 🏨 App Title and Instructions
# ================================================================




st.title("🏨 Hotel Review Analyzer")

st.markdown("""
Welcome to the **AI-powered review analyzer**! 🤖
- 📂 Upload a hotel review file (PDF, TXT, DOCX) OR ✍️ Type manually.
- 🔍 Click Predict.
- Instantly see whether it’s **Positive ✅**, **Negative ❌**, or **Neutral 😐**.
""")
st.markdown("---")





# ================================================================
# 📌 Sidebar with Info
# ================================================================



st.sidebar.header("🧐 About the App")
st.sidebar.info("""
This Hotel Sentiment Detection App uses an AI model to analyze hotel reviews.
""")

st.sidebar.header("🛠️ How to Use")
st.sidebar.markdown("""
1. Upload a review file OR type a review manually.  
2. Click Predict Sentiment.  
3. See the prediction result!
4. This app predicts hotel review sentiment and detects mentioned country.
                    
👉 Input should be minimum 3 words to get a meaningful prediction 😊
""")






# ================================================================
# 📋 Tabs for Upload and Typing
# ================================================================




tab1, tab2 = st.tabs(["📂 Upload Review File", "✍️ Type Review Manually"])

# ================================================================
# 📋 Upload Review File Tab
# ================================================================


with tab1:
    uploaded_file = st.file_uploader("Upload a review file (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])

    if st.button("🔍 Predict Uploaded Review", key="predict_uploaded"):
        if uploaded_file is not None:
            filename = uploaded_file.name.lower()
            try:
                if filename.endswith(".txt"):
                    review_text = extract_text_from_txt(uploaded_file)
                elif filename.endswith(".pdf"):
                    review_text = extract_text_from_pdf(uploaded_file)
                elif filename.endswith(".docx"):
                    review_text = extract_text_from_docx(uploaded_file)
                else:
                    st.error("Unsupported file format.")
                    st.stop()

                if len(review_text.split()) < 3:
                    st.warning("⚠️ Review text too short for prediction.")
                    st.stop()

                st.markdown("### 📝 Extracted Review Text:")
                st.text_area("Extracted Text:", value=review_text, height=100, disabled=True)

                label, emoji, color = predict_sentiment(review_text)
                detected_country = detect_country(review_text)

                st.markdown("## Prediction Result:")
                st.markdown(f"<span style='color: {color}; font-size: 24px;'>{emoji} {label}</span>", unsafe_allow_html=True)

                if detected_country:
                    st.success(f"✅ Country Detected: **{detected_country}**")

                if label == "Positive Review":
                    st.balloons()

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        else:
            st.warning("⚠️ Please upload a file first.")





# ================================================================
# 📋  Type Review Manually Tab
# ================================================================


with tab2:
    user_input = st.text_area("✍️ Type or Paste your review here:", height=150)

    if st.button("🔍 Predict Typed Review", key="predict_typed"):
        if user_input.strip() == "":
            st.warning("⚠️ Please type a review first.")
            st.stop()

        if len(user_input.split()) < 3:
            st.warning("⚠️ Typed review is too short for prediction.")
            st.stop()

        review_text = user_input.strip()

        label, emoji, color = predict_sentiment(review_text)
        detected_country = detect_country(review_text)

        st.markdown("## Prediction Result:")
        st.markdown(f"<span style='color: {color}; font-size: 24px;'>{emoji} {label}</span>", unsafe_allow_html=True)

        if detected_country:
            st.markdown(f"✅ Country Detected: **`{detected_country}`**")

        if label == "Positive Review":
            st.balloons()

st.markdown("</div>", unsafe_allow_html=True)