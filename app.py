import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load(open('forestmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    print(request.form.values())
    output = [x for x in request.form.values()]
    country = output[1]
    gender = output[2]
    hascard = output[7]
    activemember = output[8]
    if country == 'France':
        output[1] = 810
    elif country == 'Germany':
        output[1] = 814
    else:
        output[1]=413

    if gender == 'Female':
        output[2] = 1139
    else:
        output[2] = 898
    if hascard == 'Yes':
        output[7]=1
    else:
        output[7]=0
    if activemember == 'Yes':
        output[8]=1
    else:
        output[8]=0
    int_features = [float(x) for x in output]
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    if prediction == [1]:
        prediction = 'The Customer is likely to churn'
    elif prediction == [0]:
        prediction = 'The Customer will not churn'

    # output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='{}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)