from flask import Flask, jsonify
from flask_restful import Resource, Api 
import pickle
import configparser
import numpy as np


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


def parse_params(params):
    dict_params = {}
    for p in params.split('&'):
        u = p.split("=")
        if len(u) != 2:
            raise ValueError("URL parameters not in proper format")
        dict_params[u[0]] = u[1]
    return dict_params


class ModelPrediction(Resource): 
    # Implements a get and post requests
    def get(self, params): 
        # parse parameters passed in url
        dict_params = parse_params(params)
        X = np.array([[dict_params['sepal_length'], dict_params['sepal_width'], dict_params['petal_length'], dict_params['petal_width']]])
        X = sc.transform(X)
        pred = model.predict(X)
        return jsonify({'prediction': pred[0]}) 

# adding the defined resources along with their corresponding urls 
api.add_resource(ModelPrediction, '/model/<string:params>') 


if __name__ == '__main__':
    app.run(debug = config['debug'])

    # Once the API is up and running, you can request your model preidction via
    # http://localhost:5000/model/sepal_length=5.0&sepal_width=2.9&petal_length=1.5&petal_width=0.2