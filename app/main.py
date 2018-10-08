from keras.models import model_from_json
import tensorflow as tf

import numpy as np
import os

from flask import jsonify, Flask
from flask import request

app = Flask(__name__)

global graph, model
graph = tf.get_default_graph()

# wczytaj model z pliku o nazwie model.json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# załaduj wagi do nowego modelu
model.load_weights('model.h5')
model.compile(loss='binary_crossentropy',optimizer='adam')


@app.route('/keras_predict', methods=['POST'])
def index():
    # getting an array of features from the request's body
    feature_array = request.get_json(force=True)['feature_array']
    #print(feature_array)
    # creating a response object
    # storing the model's prediction in the object
    response = {}
    with graph.as_default():
        response['prediction'] = model.predict_classes(np.array([feature_array]))[0].tolist()

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)