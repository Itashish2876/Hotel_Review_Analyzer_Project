# ================================================================
# 📦 Imports
# ================================================================
import streamlit as st
import joblib
import re
from io import BytesIO
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder






# ================================================================
# ⚙️ Page Config
# ================================================================
st.set_page_config(page_title="Hotel Sentiment App", page_icon="🏨", layout="centered")








#   ================================================================
# Custom CSS for better layout and colors
#   ================================================================
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
# 🏨 App Title and Instructions
# ================================================================
st.title("🏨 Hotel Review Analyzer")

st.markdown("""
Welcome to the **AI-powered review analyzer**! 🤖

- Paste or upload a hotel review.
- Click **Predict Sentiment**.
- Instantly see whether it’s **Positive ✅**, **Negative ❌**, or **Neutral 😐**.
""")
st.markdown("---")







# ================================================================
# 📌 Sidebar with Info
# ================================================================
st.sidebar.header("🧐 About the App")
st.sidebar.info("""
This **Hotel Sentiment Detection App** uses an AI model to analyze hotel reviews.
""")

st.sidebar.header("🛠️ How to Use")
st.sidebar.markdown("""
1. Upload a review file (TXT, PDF, DOCX) **or** type in the box.  
2. Paste your review in the input box.  
3. Click **Predict Sentiment**.  
4. See the result displayed below.

👉 Make sure your input has at least **3 words** to get a meaningful prediction 😊
""")









# ================================================================
# 🧠 Load Models
# ================================================================
try:
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("logistic_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    st.error("❌ Model files not found. Ensure 'tfidf_vectorizer.pkl', 'logistic_model.pkl', and 'label_encoder.pkl' exist.")
    st.stop()








# ================================================================
# 🧹 Text Cleaning Function
# ================================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text









# ================================================================
# 📂 File Handling Functions
# ================================================================
# From PDF Extract Text 
def extract_text_from_pdf(file):  # Opens the PDF reads each page, extracts text , joins all into one string , and remove extra spaces 
    reader = PyPDF2.PdfReader(BytesIO(file.read()))
    text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text.strip()

# From TEXT Extract Text 
def extract_text_from_txt(file):        # Reads the whole file , decodes it as text , and remove extra spaces 
    return file.read().decode('utf-8').strip()

# From DOCX Extract Text 
def extract_text_from_docx(file):
    doc = docx.Document(file)       # Opens the word documents , reads each paragraph , joins them into one string , and removes extra spaces  
    return "\n".join([para.text for para in doc.paragraphs]).strip()








# ================================ ================================
# 📝 Input Section (File + Text)
# ================================ ================================
uploaded_file = st.file_uploader("📂 Upload a review file (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"])

st.markdown("### ✍️ Or type/paste your review below:")
user_input = st.text_area("", height=200)








# ================================ ================================
# 🔮 Sentiment Prediction Function
# ================================ ================================
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







# ================================ ================================
# 🔘 Predict Button Logic
# ================================ ================================
if st.button("🔍 Predict Sentiment"):
    review_text = ""

    # Handle input from file
    if uploaded_file:
        filename = uploaded_file.name.lower()
        try:
            if filename.endswith(".txt"):
                review_text = extract_text_from_txt(uploaded_file)
            elif filename.endswith(".pdf"):
                review_text = extract_text_from_pdf(uploaded_file)
            elif filename.endswith(".docx"):
                review_text = extract_text_from_docx(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload PDF, TXT or DOCX.")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    # Handle input from text box
    else:
        review_text = user_input.strip()

    # Validate input
    if len(review_text.split()) < 3:
        st.warning("⚠️ Please enter a more detailed review (at least 3 words) for accurate prediction.")
    else:
        # Predict sentiment
        label, emoji, color = predict_sentiment(review_text)

        # Display result
        st.markdown("## Prediction Result:")
        st.markdown(
            f"<span style='color: {color}; font-size: 24px;'>{emoji} {label}</span>",
            unsafe_allow_html=True
        )
        # 🎉 Balloons on Positive Review
        if label == "Positive Review":
            st.balloons()



st.markdown("</div>", unsafe_allow_html=True)