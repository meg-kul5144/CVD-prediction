import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

heart_dt = pd.read_excel("heart.xlsx")
oldpeak = np.array(heart_dt['oldpeak']).reshape(-1,1)
scaler1 = MinMaxScaler()
scaler1.fit(oldpeak)

heart_rate = np.array(heart_dt['maximum heart rate achieved']).reshape(-1,1)
scaler2 = MinMaxScaler()
scaler2.fit(heart_rate)

flouroscopy = np.array(heart_dt['number of major vessels coloured by fluoroscopy']).reshape(-1,1)
scaler3 = MinMaxScaler()
scaler3.fit(flouroscopy)

# Create flask app
flask_app = Flask(__name__, template_folder='Templates/')
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    #float_features = [float(x) for x in request.form.values()]
    heart_rate = float(np.sum(scaler2.transform(np.array(request.form.get('Maximum heart rate achieved')).reshape(-1,1))))
    old_peak =  float(np.sum(scaler1.transform(np.array(request.form.get('Old peak')).reshape(-1,1))))
    flouroscopy =  float(np.sum(scaler3.transform(np.array(request.form.get('Major vessels coloured by fluoroscopy')).reshape(-1,1))))
    upsloping =  float(request.form.get('Upsloping'))
    flat =  float(request.form.get('Flat'))
    downsloping =  float(request.form.get('Downsloping'))
    stwave =  float(request.form.get('ST-T wave abnormality'))
    lvhyper =  float(request.form.get('Left ventricular hypertrophy'))
    fd =  float(request.form.get('Fixed defect'))
    notdes =  float(request.form.get('Not described'))
    revd =  float(request.form.get('Reversible defect'))
    asy =  float(request.form.get('Asymptomatic'))
    ex_an =  float(request.form.get('Exercise induced angina'))
    at_an =  float(request.form.get('Atypical angina'))
    non_an =  float(request.form.get('Non-anginal pain'))
    ty_an  = float(request.form.get('Typical angina'))

    features = np.array([ty_an,revd,flouroscopy,old_peak,heart_rate,non_an,at_an,asy,lvhyper,ex_an,downsloping,notdes,flat,upsloping,fd,stwave,
                         ]).reshape(1,-1)
    prediction = model.predict(features)
    if(prediction==1):
        prediction = "Heart disease is likely to prevail. Please consult a physician for confirmation"
    else:
        prediction = "There are no signs of heart disease."
    return render_template("index.html", prediction_text = "Heart disease prediction:  {}".format(prediction), data=features)
if __name__ == "__main__":
    flask_app.run(debug=True)