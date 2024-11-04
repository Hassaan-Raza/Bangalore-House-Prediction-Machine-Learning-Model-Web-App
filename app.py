import numpy as np
from flask import Flask, request,jsonify,render_template
import pickle

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction = model.predict(final_features)
    out = round(float(prediction[0]), 2)
    out = out*100
    return render_template('index.html',prediction_text="Estimated Price of this house is: {} Rupees".format(out))

if __name__ == "__main__":
    app.run(debug=True)