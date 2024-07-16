from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the model and CountVectorizer
with open(r'C:\Users\amant\render-demo\model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open(r'C:\Users\amant\render-demo\cv.pkl', 'rb') as cv_file:
    cv = pickle.load(cv_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    transformed_text = cv.transform([text])
    prediction = model.predict(transformed_text)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
