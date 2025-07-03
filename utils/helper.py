import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove special characters, punctuation, digits
    text = re.sub(r'[^a-z\s]', '', text)

    # 3. Tokenize
    words = word_tokenize(text)

    # 4. Remove stopwords & lemmatize
    clean_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

    # 5. Rejoin into string
    return " ".join(clean_words)
