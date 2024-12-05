import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# inputs
celsius    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit = np.array([-40,  14, 32, 46.4, 59, 71.6, 100],  dtype=float)

# criação dos layers
l1 = tf.keras.layers.Dense(units=1, input_shape=[1])

model = tf.keras.Sequential([l1])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

# treinamento do modelo
history = model.fit(celsius, fahrenheit, epochs=500, verbose=False)
print("Finished training the model")

# testando com uma predição de 100ºC
predicao = model.predict(np.array([100.0]))
print(f"Predição para 100ºC: {predicao[0][0]:.2f}°F")

# fórmula da conversão
print("These are the layer variables: {}".format(l1.get_weights()))

# estatística (exibe o gráfico depois das impressões)
plt.xlabel('Quantidade de nós')
plt.ylabel("Perda")
plt.plot(history.history['loss'])
plt.show()  # Mostra o gráfico no final do script
