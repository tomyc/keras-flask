from keras.models import model_from_json
import numpy as np

# odczytaj model z dysku
json_file = open('..\\app\\model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# załaduj wagi do nowego modelu
model.load_weights('..\\app\\model.h5')
print('Model załadowany')

predict = model.predict_classes(np.array([[6,148,72,35,0,33.6,0.627,50]]))
print(predict[0])