import numpy as np
import tensorflow as tf
from keras.models import Sequential
import time

start = time.time()
 
# cargamos las 4 combinaciones de las compuertas OR
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
 
# y estos son los resultados que se obtienen, en el mismo orden
target_data = np.array([[0],[1],[1],[0]], "float32")
 
model = Sequential([
    tf.keras.Input(shape = 2, name='input'),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid', name='output')
])

print('Pesos Iniciales Entrada-Oculta: '+ str(model.layers[0].get_weights()[0]))
print('Pesos Iniciales Oculta-Salida: '+ str(model.layers[1].get_weights()[0]))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'],
              loss_weights=0.25)

model.fit(training_data, target_data, epochs=1000, )

print('Pesos Finales Entrada-Oculta: '+ str(model.layers[0].get_weights()[0]))
print('Pesos Finales Oculta-Salida: '+ str(model.layers[1].get_weights()[0]))

# evaluamos el modelo
scores = model.evaluate(training_data, target_data)
 
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print (model.predict(training_data).round())

print("Tiempo de ejecucion:")
print("--- %s seconds ---" % (time.time() - start))
