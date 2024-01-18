import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from flask import Flask, render_template, request, jsonify

# Existing chatbot setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents and model
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Jiq"

# Flask app setup
app = Flask(__name__)

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send', methods=['POST'])
def send():
    user_input = request.json['user_input']
    response = chatbot_response(user_input)
    return jsonify({'response': response})

def chatbot_response(user_input):
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I am sorry, I cannot help with your query. Feel free to ask our customer help desk to enquire them about your problem. You can click this link to get their contact number: <a href='http://localhost/Chatbot/contact.php' target='_blank'>Ask Agent</a>"

if __name__ == '__main__':
    app.run(debug=True)
