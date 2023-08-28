import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Set Streamlit title and description
st.title("Sentiment Analysis App")
st.write("Enter a message to analyze its sentiment.")

# Create a text input for user to enter the message
user_input = st.text_input("Enter a message:")

if st.button("Get Sentiment"):
    # Tokenize the input text
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)

    # Perform sentiment analysis
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax().item()

    sentiment_map = {
        0: "Very Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Very Positive"
    }

    st.write("Predicted Sentiment:", sentiment_map[predicted_class])
