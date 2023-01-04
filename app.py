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
    print(data)
    # print(np.array(list(data.values())).reshape(1, -1))
    columns = np.array(list(data.keys()))
    values = np.array(list(data.values())).reshape(1, -1)
    new_data = pd.DataFrame(data=values, columns=columns)
    output = xgbmodel.predict(new_data)
    print(output[0])
    return jsonify(str(output[0]))


if __name__ == "__main__":
    app.run(debug=True)
