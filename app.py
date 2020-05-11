import numpy as np
from flask import Flask, request, jsonify, render_template, make_response
import pickle
from flask_cors import cross_origin
import json
import os
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
#import numpy as np
import tflearn
import tensorflow  as tf
import random
#import json
#import pickle

app = Flask(__name__)
str = 'chatbot_app'
if(str =='chatbot_app'):
    @app.route("/")
    def home():
        return render_template("chat.html")
    with open("intents.json","rb") as file:
        data = json.load(file)
    try:
        with open("data.pickle", "rb") as f:
            words, labels, training, output = pickle.load(f)
    except:

        words = []
        labels = []
        docs_x = []
        docs_y = []

        for intent in data['intents']:
            for pattern in intent['patterns']:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent['tag'])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

        words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
        words = sorted(list(set(words)))
        labels = sorted(labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []
            wrds = [stemmer.stem(w) for w in doc]
            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)

        trining = np.array(training)
        output = np.array(output)
        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)

    tf.reset_default_graph()  # Tensorflow as tf

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)
    model = tflearn.DNN(net)

    try:
        model.load("model.tflearn")
    except:
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save("model.tflearn")


    def bag_of_words(s, words):
        bag = [0 for _ in range(len(words))]
        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]
        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return np.array(bag)


    app.route('/chatting')
    def chat():
        print("Start talking with bot! (type quit to stop!)")
        inp = str(request.args.get('msg'))

        results = model.predict([bag_of_words(inp, words)])[0]
        # print("Results : ",results)
        results_index = np.argmax(results)
        tag = labels[results_index]
        if results[results_index] > 0.5:
            print("Intent: ", tag)
            print("Confidence: ", results[results_index])
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            resp = random.choice(responses)
            print("bot: ", random.choice(responses))
        else:
            print("I didn't get that, Please try again!!")
            resp = "I didn't get that, Please try again!!"

        return resp


    @app.route('/chat')
    def chatter():
        inp = request.args.get('msg')
        resp = chat()
        return str(resp)
else:
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
    @cross_origin()
    def predictsalary():
        '''
        For rendering results to DialogFlow
        '''
        req = request.get_json(silent=True, force=True)

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
        res = { "fulfillmentText" : "Your predicted Salary is $ {}".format(output)  }
        res = json.dumps(res, indent=4)
        print(res)
        r = make_response(res)
        r.headers['Content-Type'] = 'application/json'
        return r

if __name__ == "__main__":
    app.run(debug=True)
