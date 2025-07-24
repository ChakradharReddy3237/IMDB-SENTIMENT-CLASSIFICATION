# IMDB Movie Review Sentiment Analysis

This project performs sentiment analysis on IMDB movie reviews to classify them as either "positive" or "negative". It involves a complete machine learning pipeline, from data cleaning and preprocessing to model training, evaluation, and finally, deployment in a simple web application.

## ✨ Features

-   **Comprehensive Text Preprocessing**: A robust pipeline cleans the review text by:
    -   Converting to lowercase
    -   Removing emails, URLs, HTML tags, punctuation, and emojis
    -   Removing stopwords
    -   Performing lemmatization
-   **Feature Extraction**: Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into meaningful numerical features.
-   **Model Comparison**: Trains and evaluates three different classification models:
    -   Logistic Regression
    -   LinearSVC (Support Vector Machine)
    -   Random Forest
-   **Model Persistence**: The best-performing model (Logistic Regression) is saved for easy inference.
-   **Web Interface**: A simple web application (`app.py`) allows users to input a review and get a real-time sentiment prediction.

## 🛠️ Tech Stack

-   **Python 3**
-   **Scikit-learn**: For machine learning modeling, feature extraction, and evaluation.
-   **Pandas & NumPy**: For data manipulation and analysis.
-   **NLTK**: For Natural Language Processing tasks like stopword removal and lemmatization.
-   **Matplotlib & Seaborn**: For data visualization of results and confusion matrices.
-   **Joblib**: For saving and loading the trained model and vectorizer.
-   **Flask/Streamlit** (in `app.py`): For building and serving the web application.

## ⚙️ Setup and Installation

Follow these steps to get the project running on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ChakradharReddy3237/imdb-sentiment-analysis.git
    cd imdb-sentiment-analysis
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # Create a virtual environment
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate

    # Activate on macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    You'll need to download the necessary NLTK packages. Run the following commands in a Python interpreter:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

5.  **Run the Web Application:**
    Start the application to begin making predictions.
    ```bash
    streamlit run app.py
    ```
    Open your web browser and navigate to `http://127.0.0.1:5000` (or the address provided in your terminal).

## 📋 Project Workflow

1.  **Data Loading**: The [IMDB Dataset](https://github.com/laxmimerit/All-CSV-ML-Data-Files-Download/raw/refs/heads/master/IMDB-Dataset.csv).
2.  **Text Preprocessing**: Each review undergoes the rigorous cleaning process described in the features section to prepare it for the model.
3.  **Feature Engineering**: The cleaned text is converted into a numerical matrix using `TfidfVectorizer`, which reflects the importance of each word relative to the entire corpus of reviews.
4.  **Model Training & Evaluation**: The dataset is split into training and validation sets. Three models are trained and their performance is evaluated to find the most accurate one.
5.  **Model Selection & Saving**: Logistic Regression was identified as the best-performing model. Both the model and the TF-IDF vectorizer are saved as `.pkl` files using `joblib`.
6.  **Inference**: The `app.py` file loads the saved model and vectorizer to provide real-time sentiment predictions on new reviews entered by the user.

## 📊 Results

The models were evaluated on the validation set. **Logistic Regression** was chosen as the final model due to its superior performance.

### Model Accuracy Comparison

| Model                 | Accuracy |
| --------------------- | :------: |
| **Logistic Regression** | **88.60%** |
| LinearSVC             |  87.74%  |
| Random Forest         |  84.89%  |

## 📊 Detailed Classification Reports

### **Logistic Regression**
          precision    recall  f1-score   support

       0       0.90      0.87      0.88      5000
       1       0.88      0.90      0.89      5000

accuracy                           0.89     10000
macro avg 0.89 0.89 0.89 10000
weighted avg 0.89 0.89 0.89 10000

### **LinearSVC**
          precision    recall  f1-score   support

       0       0.88      0.87      0.88      5000
       1       0.87      0.88      0.88      5000

accuracy                           0.88     10000
macro avg 0.88 0.88 0.88 10000
weighted avg 0.88 0.88 0.88 10000



### **Random Forest**
          precision    recall  f1-score   support

       0       0.84      0.86      0.85      5000
       1       0.85      0.84      0.85      5000

accuracy                           0.85     10000
macro avg 0.85 0.85 0.85 10000
weighted avg 0.85 0.85 0.85 10000


## 🚀 How to Use the App

1.  Ensure all setup steps are complete and run `streamlit run app.py`.
2.  Open the provided URL in your browser.
3.  Enter a movie review into the text box.
4.  Click the "Predict" button to see the predicted sentiment ("Positive" or "Negative").