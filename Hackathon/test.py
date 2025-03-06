import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("cyberbullying_tweets.csv")

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  
    text = text.lower()  
    text = text.split()  
    text = [word for word in text if word not in stop_words]  
    text = [lemmatizer.lemmatize(word) for word in text]  
    return " ".join(text)

# Apply text cleaning
df['clean_text'] = df['tweet_text'].apply(clean_text)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df['clean_text'])

# Convert to DataFrame
X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())

# Define target variable
X = X_tfidf_df
y = df['cyberbullying_type']

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Category Descriptions
category_descriptions = {
    "not_cyberbullying": "The input text does not contain cyberbullying content.",
    "gender": "The input text contains cyberbullying based on gender.",
    "religion": "The input text contains cyberbullying based on religion.",
    "other_cyberbullying": "The input text contains other forms of cyberbullying.",
    "age": "The input text contains cyberbullying based on age.",
    "ethnicity": "The input text contains cyberbullying based on ethnicity."
}

# Detailed Explanation
category_definitions = {
    "not_cyberbullying": "Cyberbullying is not detected in the given input.",
    "gender": "Gender-based cyberbullying involves targeting someone based on their gender identity, using sexist remarks, stereotypes, or discrimination.",
    "religion": "Religion-based cyberbullying involves attacking or mocking someone due to their religious beliefs, often leading to hate speech.",
    "other_cyberbullying": "This category includes various forms of cyberbullying that do not fit into specific categories like age, gender, or religion.",
    "age": "Age-based cyberbullying targets individuals based on their age, often discriminating against younger or older groups.",
    "ethnicity": "Ethnicity-based cyberbullying involves discrimination, stereotypes, or offensive comments directed at a person’s ethnic background."
}

# Streamlit UI
st.title("🔍 Cyberbullying Detection System")

st.write("Enter a tweet or text below to check if it contains cyberbullying.")

user_input = st.text_area("Enter your text here:")

if st.button("Detect Cyberbullying"):
    if user_input.strip():
        cleaned_text = clean_text(user_input)
        transformed_text = tfidf.transform([cleaned_text])
        predicted_label = model.predict(transformed_text)[0]

        st.subheader(f"The input value is **{predicted_label}**.")
        st.write(category_descriptions[predicted_label])
        st.write(f"📌 **What is {predicted_label}?** {category_definitions[predicted_label]}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
