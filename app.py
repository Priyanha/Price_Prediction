# Data Handling
import joblib
import numpy as np
import pandas as pd
import os

from flask import Flask, request, redirect, render_template, jsonify
from backend.preprocessing import preprocess
from backend.encoders import get_encodings

from backend.models import f_regr

app = Flask(__name__)

scaler = joblib.load('./backend/assets/files/scaler.pickle')

model = f_regr()
model.load_weights('./backend/assets/models/model.h5')

columns = ['name', 'item_condition_id', 'brand_name', 'category_name', 'shipping', 'item_description']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():

    name = request.args.get('name')
    item_condition_id = request.args.get('item_condition_id')
    brand_name = request.args.get('brand_name')
    category_name = request.args.get('category_name')
    shipping = request.args.get('shipping')
    item_description = request.args.get('item_description')

    features = [name, item_condition_id, brand_name, category_name, shipping, item_description]

    data = np.array(features)

    df = pd.DataFrame([data], columns=columns)

    processed_data = preprocess(df)

    encoded_data = get_encodings(processed_data)

    pred = model.predict(encoded_data)
    prediction = np.expm1(scaler.inverse_transform(pred.reshape(-1, 1))[:,0])

    return jsonify(name=name, item_condition_id=item_condition_id, brand_name=brand_name, category_name=category_name, shipping=shipping, item_description=item_description, price='${:.2f}'.format(prediction[0]))

@app.route('/predict1', methods=['POST'])
def predict1():

    features = [x for x in request.form.values()]

    data = np.array(features)

    df = pd.DataFrame([data], columns=columns)

    processed_data = preprocess(df)

    encoded_data = get_encodings(processed_data)

    pred = model.predict(encoded_data)
    prediction = np.expm1(scaler.inverse_transform(pred.reshape(-1, 1))[:,0])

    return render_template('index.html', price='Recommended Price : ${:.2f}'.format(prediction[0]))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
