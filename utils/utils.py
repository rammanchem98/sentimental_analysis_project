import nltk
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('sentiment_analysis.log')]
)
logger = logging.getLogger(__name__)


def preprocess_text(text):
    """Preprocess text by converting to lowercase, removing punctuation, and stopwords."""
    stop_words = set(stopwords.words('english'))
    if not isinstance(text, str) or text.strip() == '':
        return np.nan
    text = text.lower()
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    tokens = word_tokenize(text)
    return ' '.join([t for t in tokens if t not in stop_words])
