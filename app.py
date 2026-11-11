

import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #1e293b);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #38bdf8;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 17px;
    margin-bottom: 30px;
}

.info-card {
    background: #263449;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #475569;
    margin-bottom: 15px;
}

.result-spam {
    background: #451a1a;
    border: 2px solid #ef4444;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: #fca5a5;
    font-size: 26px;
    font-weight: bold;
}

.result-safe {
    background: #064e3b;
    border: 2px solid #22c55e;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: #86efac;
    font-size: 26px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PREPROCESSING ----------------
ps = PorterStemmer()
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

download_nltk_data()

@st.cache_resource
def load_resources():
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return tfidf, model

tfidf, model = load_resources()

@st.cache_data
def get_stopwords():
    return set(stopwords.words('english'))

stop_words = get_stopwords()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">📩 Spam Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">An Intelligent Email & SMS Spam Classification System</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# ---------------- CLASSIFIER ----------------
st.markdown("### ✉️ Try the Spam Classifier")

st.write("Enter an email or SMS message in the box below:")

input_sms = st.text_area(
    "Your Message",
    placeholder="Example: Congratulations! You have won a free prize. Click here to claim...",
    height=150
)

if st.button("🔍 Analyze Message", use_container_width=True):

    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before clicking Analyze.")

    else:
        with st.spinner("Analyzing your message..."):

            # 1. Preprocess
            transformed_sms = transform_text(input_sms)

            # 2. Vectorize
            vector_input = tfidf.transform([transformed_sms])

            # 3. Predict
            result = model.predict(vector_input)[0]

        st.markdown("---")
        st.markdown("### 📊 Analysis Result")

        if result == 1:
            st.markdown("""
            <div class="result-spam">
                🚨 SPAM DETECTED
                <p style="font-size:16px;">
                    This message has been classified as spam.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.warning(
                "Be careful! Avoid clicking suspicious links "
                "or sharing personal information."
            )

        else:
            st.markdown("""
            <div class="result-safe">
                ✅ NOT SPAM
                <p style="font-size:16px;">
                    This message has been classified as legitimate.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.success(
                "This message appears legitimate according to the model."
            )

        with st.expander("🔬 View Text Preprocessing Details"):
            st.write("**Original Message:**")
            st.write(input_sms)

            st.write("**Processed Message:**")
            st.write(transformed_sms)

            st.write("**Number of Words:**", len(input_sms.split()))

# ---------------- HOW IT WORKS ----------------
st.markdown("---")
st.markdown("### ⚙️ How Does It Work?")

st.markdown("""
1. **Text Input:** User enters an email or SMS message.
2. **Text Preprocessing:** Converts text to lowercase, removes unnecessary
   tokens and stopwords, and applies stemming.
3. **TF-IDF Vectorization:** Converts processed text into numerical features.
4. **ML Prediction:** The trained model classifies the message.
5. **Result:** Displays whether the message is Spam or Not Spam.
""")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    Built with ❤️ using Python, Streamlit, NLTK & Machine Learning
    <br><br>
    Email/SMS Spam Classification Project
</div>
""", unsafe_allow_html=True)
