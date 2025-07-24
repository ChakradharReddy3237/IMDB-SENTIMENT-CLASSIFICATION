import streamlit as st
import joblib

# Load the pre-trained model
model = joblib.load('imdb_sentiment_classifier.pkl')    

# Function to predict sentiment
def predict_sentiment(review):
    prediction = model.predict([review])
    return prediction[0]

# Streamlit app
st.title("IMDB Sentiment Classification")
review = st.text_area("Enter your movie review:")
if st.button("Predict Sentiment"):
    if review:
        sentiment = predict_sentiment(review)
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.write("Please enter a review to get the sentiment prediction.")
else:
    st.write("Enter a movie review and click the button to predict its sentiment.") 