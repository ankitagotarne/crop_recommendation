



import tensorflow
from flask import Flask,render_template,request
import pandas as pd


app=Flask(__name__)
df=pd.read_csv("Crop_recommendation.csv")
model=tensorflow.keras.models.load_model("crop_recommendation.h5")
print("Model Loaded Sucessfully")

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    N = int(request.form.get("N"))
    P = int(request.form.get("P"))
    K = int(request.form.get("K"))
    temperature = float(request.form.get("temperature"))
    humidity = float(request.form.get("humidity"))
    ph = float(request.form.get("ph"))
    rainfall = float(request.form.get("rainfall"))
    print(N, P, K, temperature, humidity, ph, rainfall)
    input = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
    print(input)
    if model.predict(input)[0][0] == 1.0:
        prediction = "apple"
    elif model.predict(input)[0][1] == 1.0:
        prediction = "banana"
    elif model.predict(input)[0][2] == 1.0:
        prediction = "blackgram"
    elif model.predict(input)[0][3] == 1.0:
        prediction = "chickpea"
    elif model.predict(input)[0][4] == 1.0:
        prediction = "coconut"
    elif model.predict(input)[0][5] == 1.0:
        prediction = "coffee"
    elif model.predict(input)[0][6] == 1.0:
        prediction = "cotton"
    elif model.predict(input)[0][7] == 1.0:
        prediction = "grapes"
    elif model.predict(input)[0][8] == 1.0:
        prediction = "jute"
    elif model.predict(input)[0][9] == 1.0:
        prediction = "kidneybeans"
    elif model.predict(input)[0][10] == 1.0:
        prediction = "lentil"
    elif model.predict(input)[0][11] == 1.0:
        prediction = "maize"
    elif model.predict(input)[0][12] == 1.0:
        prediction = "mango"
    elif model.predict(input)[0][13] == 1.0:
        prediction = "mothbeans"
    elif model.predict(input)[0][14] == 1.0:
        prediction = "mungbean"
    elif model.predict(input)[0][15] == 1.0:
        prediction = "muskmelon"
    elif model.predict(input)[0][16] == 1.0:
        prediction = "orange"
    elif model.predict(input)[0][17] == 1.0:
        prediction = "papaya"
    elif model.predict(input)[0][18] == 1.0:
        prediction = "pigeonpeas"
    elif model.predict(input)[0][19] == 1.0:
        prediction = "pomegranate"
    elif model.predict(input)[0][20] == 1.0:
        prediction = "rice"
    elif model.predict(input)[0][21] == 1.0:
        prediction = "watermelon"
    else:
        prediction = "Sorry but We can not predict"
    return prediction


app.run(debug=True)


