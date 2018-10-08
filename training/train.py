"""
Ten zestaw danych pochodzi z Narodowego Instytutu Cukrzycy i Chorób Przewodu Pokarmowego i Chorób Nerek. Celem zestawu
danych jest diagnostyczne przewidzenie, czy pacjent choruje na cukrzycę, korzystając z pewnych pomiarów diagnostycznych
zawartych w zbiorze danych. Kilka ograniczeń zostało nałożonych na wybór tych instancji danych z większej bazy danych.
W szczególności wszyscy pacjenci są tu kobietami w wieku co najmniej 21 lat z rodowodem Indian Pima.

Źródło: Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning
algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and
Medical Care (pp. 261--265). IEEE Computer Society Press.
Kolumny:
preg = Liczba razy w ciąży
plas = Stężenie glukozy w osoczu wynosi 2 godziny w doustnym teście tolerancji glukozy
pres = rozkurczowe ciśnienie krwi (mm Hg)
Skóra = grubość fałdu skórnego Triceps (mm)
test = 2-godzinna insulina w surowicy (mu U/ml)
masa = wskaźnik masy ciała (waga w kg/(wzrost w m) ^ 2)
pedi = Diabetes pedigree function
wiek = wiek (lata)
class = zmienna klasy (1: test pozytywny na cukrzycę, 0: test ujemny na cukrzycę)

Bardzo dobry przykład obrazujący jak tworzyć różne wiarianty sieci NN
https://www.samyzaf.com/ML/pima/pima.html

"""

from keras.models import Sequential
from keras.layers import Dense


import numpy as np
import os

np.random.seed(7)

#załaduj dane
# struktura pliku składa się z 9 kolum i 768 wierszy
dataset = np.loadtxt('..\\data\\pima-indians-diabetes.data.csv', delimiter=',')

#podziel na zmienne wejściowe (X) i  wyjściowe (Y)
X = dataset[:,0:8]
Y = dataset[:,8]

#utwórz model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

#zbuduj model
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

#wytrenuj model
model.fit(X, Y, nb_epoch=150, batch_size=10, verbose=0)

#ewalauuj model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#serializuj model do formatu JSON
model_json = model.to_json()
with open('..\\app\model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('..\\app\\model.h5')
print('Model zapisany na dysk')

predict = model.predict_classes(np.array([[6,148,72,35,0,33.6,0.627,50]]))
print(predict)