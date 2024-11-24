# Install required packages (uncomment if needed)
# !pip install pandas numpy tensorflow scikit-learn nltk flask pyngrok joblib

import nltk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras import layers, models
from flask import Flask, render_template, request, jsonify
from pyngrok import ngrok
import joblib

# Download NLTK data (run only once)
nltk.download('punkt')
nltk.download('wordnet')

# Load the dataset
data = pd.read_csv('questions_answers_dataset.csv')  # Ensure the file is present in the same directory

# Preprocessing
data['Question'] = data['Question'].str.lower()  # Convert to lowercase

lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized)

data['Processed_Question'] = data['Question'].apply(preprocess_text)

# Encode answers
label_encoder = LabelEncoder()
data['Encoded_Answer'] = label_encoder.fit_transform(data['Answer'])

# Split into training and test sets
X = data['Processed_Question']
y = data['Encoded_Answer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
X_test_vectorized = vectorizer.transform(X_test).toarray()

# Define the model
model = models.Sequential()
model.add(layers.Input(shape=(X_train_vectorized.shape[1],)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_vectorized, y_train, epochs=20, batch_size=8, validation_split=0.1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_vectorized, y_test)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Save the model, vectorizer, and label encoder
model.save('chatbot_model.h5')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Flask Application
app = Flask(__name__)

# Example chatbot response function
def chatbot_response(question):
    tokens = word_tokenize(question.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    vectorized_input = vectorizer.transform([' '.join(lemmatized)]).toarray()
    prediction = model.predict(vectorized_input)
    predicted_class = np.argmax(prediction)
    answer = label_encoder.inverse_transform([predicted_class])[0]
    return answer

@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            #chatbox { border: 1px solid #ccc; padding: 10px; height: 300px; overflow-y: scroll; }
            #input { width: 80%; padding: 10px; }
            #send { padding: 10px 20px; }
        </style>
    </head>
    <body>
        <h1>Chatbot</h1>
        <div id="chatbox"></div>
        <input type="text" id="input" placeholder="Type your question here...">
        <button id="send">Send</button>

        <script>
            const chatbox = document.getElementById("chatbox");
            const input = document.getElementById("input");
            const send = document.getElementById("send");

            send.addEventListener("click", () => {
                const question = input.value;
                if (question.trim()) {
                    chatbox.innerHTML += `<p><strong>You:</strong> ${question}</p>`;
                    fetch("/get_response", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ question })
                    })
                    .then(response => response.json())
                    .then(data => {
                        chatbox.innerHTML += `<p><strong>Chatbot:</strong> ${data.response}</p>`;
                        chatbox.scrollTop = chatbox.scrollHeight;
                    });
                    input.value = "";
                }
            });
        </script>
    </body>
    </html>
    """

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("question", "")
    if user_input.strip():
        response = chatbot_response(user_input)
        return jsonify({"response": response})
    return jsonify({"response": "I didn't understand that. Can you try rephrasing?"})

if __name__ == "__main__":
    # Authenticate ngrok
    # ngrok.set_auth_token("2o7e6FIrw04xxyYxW8do2GRy66P_5PKGStDabos8nGqBfBPCT")  # Replace with your ngrok token
    # public_url = ngrok.connect(5000)
    # print(f"Public URL: {public_url}")

    # Run Flask app
    app.run(port=5000)
