# Sentiment Analysis Project

This project is a web application that predicts the sentiment (positive/negative) of user-provided text. It features a user-friendly web interface built with **Flask** and leverages pre-trained Machine Learning models for accurate predictions.

## 📂 Project Structure

The project is organized into modular folders for better maintainability:

* **`app.py`**: The main Flask application entry point.
* **`models/`**: Contains the pre-trained models and tokenizers:
    * `lstm_model.h5`: Deep Learning model (Long Short-Term Memory).
    * `svm_model.joblib`: Support Vector Machine model.
    * `tokenizer.joblib` & `vectorizer.joblib`: For text preprocessing.
* **`service/`**: Scripts used for training the models (e.g., `training.py`).
* **`utils/`**: Helper functions for text cleaning and processing (`utils.py`).
* **`templates/`**: HTML files for the web interface (`index.html`).

## 🚀 How to Run

1.  **Clone the repository**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the application:**
    ```bash
    python app.py
    ```
4.  **Open your browser:**
    Go to `http://127.0.0.1:5000/` to use the sentiment analyzer.
