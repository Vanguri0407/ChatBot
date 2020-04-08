import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
@app.route('/predictsalary',methods=['POST'])
def predictsalary():
    '''
    For rendering results to DialogFlow
    '''
    sessionID=req.get('responseId')


    result = req.get("queryResult")
    user_says=result.get("queryText")
    #log.write_log(sessionID, "User Says: "+user_says)
    parameters = result.get("parameters")
    interview_score=parameters.get("interview_score")
    #print(cust_name)
    test_score = parameters.get("test_score")
    experience=parameters.get("experience")
    #course_name= parameters.get("course_name")  
    
    int_features = [interview_score,test_score,experience]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    fulfillmentText = output
    return {
            "fulfillmentText": fulfillmentText
        }

if __name__ == "__main__":
    app.run(debug=True)
