from flask import Flask, render_template, request
import numpy as np     #Importing numpy package
import joblib
app = Flask(__name__)

model = joblib.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_features=[float(x) for x in request.form.values()]
    features=[np.array(float_features)]
    prediction = model.predict(features)
    return render_template('index.html', prediction_text= "Predicted area is {} ha.".format(pow(10,prediction[0]-1)))

if __name__ == '__main__':
    app.run(port=3000,debug=True)
