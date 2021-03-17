from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse

import configparser
import pickle
import numpy as np
from urllib import parse

app = Flask(__name__) 
api = Api(app) 

# load configuration
env = 'DEV' #PROD
config = configparser.ConfigParser()
config.read('config.ini')
config = config[env]

# load model
model = pickle.load(open(config['model_path'],'rb'))
sc = pickle.load(open(config['scaler_path'],'rb'))


class ModelPrediction(Resource): 
    def get(self): 
        # parse parameters passed in url
        parser = reqparse.RequestParser()
        parser.add_argument('sepal_length', type=float)
        parser.add_argument('sepal_width', type=float)
        parser.add_argument('petal_length', type=float)
        parser.add_argument('petal_width', type=float)
        params = parser.parse_args(strict=True)

        # Create vector and compute model prediction
        X = np.array([[params['sepal_length'], params['sepal_width'], params['petal_length'], params['petal_width']]])
        X = sc.transform(X)
        pred = model.predict(X)
        return jsonify({'prediction': pred[0]}) 

# adding the defined resources along with their corresponding urls 
api.add_resource(ModelPrediction, '/model') 


if __name__ == '__main__':
    app.run(debug = config['debug'])

    # Once the API is up and running, you can request your model preidction via
    # http://localhost:5000/model?sepal_length=5.0&sepal_width=2.9&petal_length=1.5&petal_width=0.2