import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
xgbmodel = pickle.load(open('adult.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    # print(data)
    # print(np.array(list(data.values())).reshape(1, -1))
    columns = np.array(list(data.keys()))
    values = np.array(list(data.values())).reshape(1, -1)
    new_data = pd.DataFrame(data=values, columns=columns)
    output = xgbmodel.predict(new_data)[0]
    # print(output)
    return jsonify(str(output))


@app.route('/predict', methods=['POST'])
def predict():
    columns = ["age", "workclass", "education", "marital.status", "occupation", "relationship", "race",
               "sex", "capital.gain", "capital.loss", "hours.per.week", "native.country"]
    # print(f'COLUMNS: {columns}')
    values = np.array(list(request.form.values())).reshape(1, -1)
    # print(f'VALUES: {values}')
    # print(f'VALUES type: {type(values)}')
    new_data = pd.DataFrame(data=values, columns=columns)
    output = xgbmodel.predict(new_data)[0]
    return render_template('home.html', prediction_text=f'The predicted income is {output}')


if __name__ == "__main__":
    app.run(debug=True)