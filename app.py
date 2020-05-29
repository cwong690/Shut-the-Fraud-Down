from flask import Flask, render_template, request, jsonify 
import pickle
import numpy as np
from predict import predict
from pymongo import MongoClient


# Create app
app = Flask(__name__)

# Render home page
@app.route('/',methods=['GET'])
def home():

    events = table.find()

    # render the template and pass the events
    return render_template('home.html', data=events)

# Execute logic of prediction
@app.route('/score', methods=['POST'])
def score():
    pass

    # return ''' <p> nothing here, friend but a link to
    #                <a href="/hello">hello</a> and an 
    #                <a href="/form_example">example form</a> </p> '''

# @app.route('/hello', methods=['GET'])
# def hello_word():
#     return ''' <h1> Hello, World!</h1> '''

# @app.route('/form_example', methods=['GET'])
# def form_display():
#     return ''' <form action="/string_reverse" method="POST">
#                 <input type="text" name="some_string" />
#                 <input type="submit" />
#                </form>
#              '''

# @app.route('/string_reverse', methods=['POST'])
# def reverse_string():
#     text=str(request.form['some_string'])
#     reversed_string = text[-1::-1]
#     return ''' output: {}   '''.format(reversed_string)

if __name__ == '__main__':
    # load saved pickled model
    with open('models/grad_boost_model.p', 'rb') as mod:
        model = pickle.load(mod)

    # connect to database
    client = MongoClient('localhost', 27017)
    db = client['frauds_test']
    table = db['new_events_test3']
    
    # run flask app
    app.run(host='0.0.0.0', port=8080, debug=True)