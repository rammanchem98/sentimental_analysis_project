import os

import joblib # for model training

# Set Kaggle credentials for API-based dataset download
os.environ['KAGGLE_USERNAME'] = "verababu"
os.environ['KAGGLE_KEY'] = "146ce97689eb875475c4a881428a5b4f"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback
from kaggle.api.kaggle_api_extended import KaggleApi
from utils.utils import preprocess_text, logger
import time


# Custom Keras callback for real-time epoch progress updates
class ProgressCallback(Callback):
    def __init__(self, emit_progress, base_percentage, total_epochs):
        super().__init__()
        self.emit_progress = emit_progress # To send updates to frontend via Socket.IO
        self.base_percentage = base_percentage  # Starting point of progress bar
        self.total_epochs = total_epochs # Total no of training epochs

    def on_epoch_begin(self, epoch, logs=None):
        progress = self.base_percentage + (epoch / self.total_epochs) * 10  # 10% range for LSTM training
        self.emit_progress(progress, f"Training LSTM model - Epoch {epoch + 1}/{self.total_epochs}")

    def on_epoch_end(self, epoch, logs=None):
        progress = self.base_percentage + ((epoch + 1) / self.total_epochs) * 10
        self.emit_progress(progress, f"Training LSTM model - Epoch {epoch + 1}/{self.total_epochs} complete")


# Ensure models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

import chardet


def fetch_dataset(dataset_name):
    try:
        logger.info("Starting dataset fetch...")
        api = KaggleApi()
        api.authenticate()
        logger.info("Kaggle API authenticated")
        logger.info(f"Downloading dataset: {dataset_name}")
        api.dataset_download_files(dataset_name, path='temp', unzip=True)
        logger.info("Dataset download complete")

        for file in os.listdir('temp'):
            if file.endswith('.csv'):
                file_path = os.path.join('temp', file)

                # Detect encoding
                with open(file_path, 'rb') as rawfile:
                    result = chardet.detect(rawfile.read(10000))  # check 10k bytes
                    encoding = result['encoding'] or 'utf-8'
                    logger.info(f"Detected encoding for {file_path}: {encoding}")
                    if encoding.lower() == 'ascii':
                        encoding = 'utf-8'

                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(df.columns)
                except Exception as e:
                    logger.warning(f"Encoding failed for {file_path} using {encoding}: {e}. Trying ISO-8859-1.")
                    try:
                        df = pd.read_csv(file_path, encoding='ISO-8859-1')
                    except Exception as e:
                        logger.error(f"Failed again with ISO-8859-1: {e}")
                        raise e
                print(df.columns)
                os.remove(file_path)
                text_candidates = ['review', 'text', 'comment', 'message', 'v2','clean_text','Tweet content']
                for col in text_candidates:
                    if col in df.columns:
                        df = df.rename(columns={col: 'text'})
                        break

                sentiment_candidates = ['sentiment', 'airline_sentiment', 'label', 'v1', 'category', 'target']
                for col in sentiment_candidates:
                    if col in df.columns:
                        df = df.rename(columns={col: 'sentiment'})
                        break

                if 'sentiment' in df.columns:
                    df = df[df['sentiment'].isin(['positive', 'negative', 0, 1, 4])]
                    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0, 0: 0, 1: 1, 4: 1})
                    return df[['text', 'sentiment']]

        raise ValueError("No suitable CSV with required columns found in dataset")
    except Exception as e:
        logger.error(f"Failed to fetch dataset: {str(e)}")
        raise

# Train or load both SVM and LSTM models; compute predictions and evaluation metrics
def train_evaluate_models(X_train, X_test, y_train, y_test, retrain=False, emit_progress=None):
    results = {}
    metrics = {}

    # SVM Model
    svm_model_path = 'models/svm_model.joblib'
    vectorizer_path = 'models/vectorizer.joblib'
    if not retrain and os.path.exists(svm_model_path) and os.path.exists(vectorizer_path):
        logger.info("Loading pre-trained SVM model")
        if emit_progress:
            emit_progress(62, "Loading pre-trained SVM model")
        svm = joblib.load(svm_model_path)
        vectorizer = joblib.load(vectorizer_path)
    else: #Train new SVM with TF-IDF feature
        logger.info("Training SVM model")
        if emit_progress:
            emit_progress(62, "Training SVM model - Vectorizing data")
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        if emit_progress:
            emit_progress(65, "Training SVM model - Fitting classifier")
        #svm = LinearSVC()
        svm = SVC(kernel='rbf')  # You can use 'linear', 'poly', 'rbf', or 'sigmoid'
        svm.fit(X_train_tfidf, y_train)
        joblib.dump(svm, svm_model_path)
        joblib.dump(vectorizer, vectorizer_path)
        if emit_progress:
            emit_progress(70, "SVM training complete")
        logger.info("SVM training complete")
    X_test_tfidf = vectorizer.transform(X_test)
    svm_pred = svm.predict(X_test_tfidf)
    results['SVM'] = svm_pred   #Evaluate SVM
    metrics['SVM'] = {
        'accuracy': float(accuracy_score(y_test, svm_pred)),
        'precision': float(precision_score(y_test, svm_pred)),
        'recall': float(recall_score(y_test, svm_pred)),
        'f1_score': float(f1_score(y_test, svm_pred)),
        'confusion_matrix': confusion_matrix(y_test, svm_pred).tolist()
    }

    # LSTM Model
    lstm_model_path = 'models/lstm_model.h5'
    tokenizer_path = 'models/tokenizer.joblib'
    if not retrain and os.path.exists(lstm_model_path) and os.path.exists(tokenizer_path):
        logger.info("Loading pre-trained LSTM model")
        if emit_progress:
            emit_progress(72, "Loading pre-trained LSTM model")
        model = load_model(lstm_model_path)
        tokenizer = joblib.load(tokenizer_path)
    else:
        logger.info("Training LSTM model")
        if emit_progress:
            emit_progress(72, "Training LSTM model - Tokenizing data")  # Tokenize, pad sequences, define and train LSTM model
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(X_train)
        X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
        X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)
        if emit_progress:
            emit_progress(75, "Training LSTM model - Building model")
        model = Sequential([
            Embedding(5000, 128, input_length=100),
            LSTM(64),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        if emit_progress:
            emit_progress(78, "Training LSTM model - Starting epochs")
        model.fit(X_train_seq, y_train, epochs=3, batch_size=32, verbose=1,  # epoch size
                  callbacks=[ProgressCallback(emit_progress, 78, 3)])
        model.save(lstm_model_path)
        joblib.dump(tokenizer, tokenizer_path)
        if emit_progress:
            emit_progress(88, "LSTM training complete")
        logger.info("LSTM training complete")
    # Evaluate LSTM
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)
    lstm_pred = (model.predict(X_test_seq) > 0.5).astype(int)
    results['LSTM'] = lstm_pred.flatten()
    metrics['LSTM'] = {
        'accuracy': float(accuracy_score(y_test, lstm_pred.flatten())),
        'precision': float(precision_score(y_test, lstm_pred.flatten())),
        'recall': float(recall_score(y_test, lstm_pred.flatten())),
        'f1_score': float(f1_score(y_test, lstm_pred.flatten())),
        'confusion_matrix': confusion_matrix(y_test, lstm_pred.flatten()).tolist()
    }

    return results, metrics

# Compare models based on F1 score, explain reasons for performance, and provide tuning suggestions
def analyze_results(results, metrics, dataset_size):
    best_algo = max(metrics, key=lambda k: metrics[k]['f1_score'])
    why = {
        'SVM': 'Powerful for linear separation, excels with TF-IDF features.',
        'LSTM': 'Captures sequence and context, effective with sequential data.'
    }
    tuning_suggestions = {}

    for model in metrics:
        precision = metrics[model]['precision']
        recall = metrics[model]['recall']
        f1 = metrics[model]['f1_score']
        cm = metrics[model]['confusion_matrix']
        fp = cm[0][1]
        fn = cm[1][0]
        suggestions = []
        if precision < 0.8:
            suggestions.append("Increase regularization (e.g., C in SVM, dropout in LSTM) to reduce false positives.")
        if recall < 0.8:
            suggestions.append("Add more diverse training data or adjust class weights to reduce false negatives.")
        if f1 < 0.8:
            suggestions.append("Tune hyperparameters (e.g., learning rate, max_features) or use ensemble methods.")
        if model == 'SVM':
            suggestions.append("Experiment with n-grams in TF-IDF or kernel tricks (e.g., RBF).")
        if model == 'LSTM':
            suggestions.append("Increase epochs, adjust batch size, or add bidirectional LSTM layers.")
        tuning_suggestions[model] = "\n".join(suggestions) if suggestions else "Performance is solid; consider minor tweaks."

    best_metrics = metrics[best_algo]
    other_algo = 'LSTM' if best_algo == 'SVM' else 'SVM'
    other_metrics = metrics[other_algo]

    # Detailed analysis
    best_model_details = {
        "why_won": f"{best_algo} achieved an F1-score of {best_metrics['f1_score']:.4f} vs. {other_metrics['f1_score']:.4f} for {other_algo}. {why.get(best_algo)}",
        "dataset_context": (
            f"With {dataset_size} samples, "
            f"{best_algo} {'excels with moderate-sized, well-preprocessed datasets' if best_algo == 'SVM' else 'benefits from enough data to learn sequential patterns'}. "
            f"{'Larger datasets (e.g., >1000 samples) with complex sequential patterns might favor LSTM' if best_algo == 'SVM' else 'Smaller datasets (e.g., <100 samples) might prefer SVM’s simplicity'}."
        ),
        "alternative": {
            "model": other_algo,
            "why_consider": (
                f"Consider {other_algo} for {'strong sequential dependencies (e.g., long reviews)' if other_algo == 'LSTM' else 'computational efficiency or simpler feature engineering'}. "
                f"Current F1-score: {other_metrics['f1_score']:.4f}."
            ),
            "improve": (
                f"{'Increase dataset size or use pre-trained embeddings (e.g., GloVe)' if other_algo == 'LSTM' else 'Experiment with non-linear kernels (e.g., RBF) or ensemble methods'} "
                f"to boost {other_algo}’s performance{' (current F1 < 0.7 suggests room for improvement)' if other_metrics['f1_score'] < 0.7 else ''}."
            )
        }
    }

    return {
        "best_algorithm": best_algo,
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1_score": best_metrics["f1_score"],
        "confusion_matrix": best_metrics["confusion_matrix"],
        "why": why.get(best_algo, "Model rationale not available."),
        "tuning_suggestions": tuning_suggestions,
        "best_model_details": best_model_details
    }


def process_data_and_train(dataset_name, retrain, emit_progress, emit_metrics, emit_results):
    try:
        emit_progress(5, f"Fetching dataset: {dataset_name}...")
        df = fetch_dataset(dataset_name)
        if 'text' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'sentiment' columns")

        emit_progress(20, "Preprocessing data...")
        df['text'] = df['text'].apply(preprocess_text)
        df.dropna(subset=['text'], inplace=True)
        df = df[df['text'].str.strip() != '']
        dataset_size = len(df)  # Capture dataset size here
        X = df['text']
        y = df['sentiment']

        emit_progress(40, "Splitting data into training & test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        emit_progress(60, "Training models...")
        results, metrics = train_evaluate_models(X_train, X_test, y_train, y_test, retrain, emit_progress)

        step = 88
        for model in metrics:
            step += 6
            emit_progress(step, f"Evaluating {model} model...")
            emit_metrics(
                model,
                metrics[model]['accuracy'],
                metrics[model]['precision'],
                metrics[model]['recall'],
                metrics[model]['f1_score']
            )
            time.sleep(1)

        analysis = analyze_results(results, metrics, dataset_size)  # Pass dataset_size
        emit_progress(100, f"Analysis completed! Best model: {analysis['best_algorithm']}")
        logger.info("Emitting final_results with metrics and analysis")
        emit_results({
            "metrics": metrics,
            "analysis": analysis
        })
        logger.info(f"Best model: {analysis['best_algorithm']} with F1-score: {analysis['f1_score']:.4f}")
        logger.info(f"Tuning suggestions: {analysis['tuning_suggestions']}")

    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        emit_progress(-100, f"Error occurred: {str(e)}")
        raise