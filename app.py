from flask import Flask, request, render_template
import pandas as pd


from models.model import preprocessing
from models.predict import predict_value

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route('/predict', methods = ['GET', 'POST'])
def data():
    if request.method == 'POST':
        f = request.form['csvfile']
        with open(f) as file:
            test_data = pd.read_csv(file)

        # preprocessing from the function imported from model
        features_test = preprocessing(test_data)

        # predicting the value from function imported from predict
        pred_ = predict_value(features_test, test_data)

        pred_table = pred_.to_html()

        return pred_table

if __name__ == "__main__":
    app.run(debug=True)