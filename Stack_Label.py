from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)
tfidf = TfidfVectorizer(analyzer='word', max_features=10000, ngram_range=(1,3), stop_words='english')

loaded_model = pickle.load(open('custom.pkl', 'rb'))
model_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
multilabel= pickle.load(open('multilabel.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        input1=[message]
        vectorized_input_data=model_vectorizer.transform(input1)
        prediction = loaded_model.predict(vectorized_input_data)
        main_result=multilabel.inverse_transform(prediction)
        output=main_result


        return render_template('index.html', prediction=output)
    else:
        return render_template('index.html', prediction="Something went wrong")
        

if __name__ == '__main__':
    app.run(debug=True)