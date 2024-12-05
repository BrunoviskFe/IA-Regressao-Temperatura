# Previsão de Fahrenheit a partir de Celsius com TensorFlow

Este projeto demonstra como utilizar um **modelo de machine learning simples** para prever temperaturas em Fahrenheit com base em valores de Celsius. Utilizando **TensorFlow** e **Keras**, o script aborda a construção, treinamento e avaliação de uma rede neural para uma tarefa básica de regressão.

---

## 📚 Funcionalidades

- Implementa uma **rede neural com uma camada densa**.
- Realiza a previsão de temperaturas Fahrenheit a partir de Celsius.
- Visualiza o processo de treinamento com um **gráfico de perda (loss)**.
- Utiliza o otimizador **Adam** e a função de perda **mean squared error**.

---

## 📂 Estrutura do Código

1. **Dependências**  
   As bibliotecas necessárias são importadas: `tensorflow` para machine learning, `numpy` para manipulação de arrays e `matplotlib` para visualização de dados.

2. **Conjunto de Dados**  
   Os valores de entrada (Celsius) e saída (Fahrenheit) são definidos:

   ```python
   celsius = np.array([-40, -10,  0,  8, 15, 22,  38], dtype=float)
   fahrenheit = np.array([-40, 14, 32, 46.4, 59, 71.6, 100], dtype=float)

3. **Arquitetura do Modelo**  
    O modelo consiste em uma camada densa com um único neurônio:

    ```python
    l1 = tf.keras.layers.Dense(units=1, input_shape=[1])
    model = tf.keras.Sequential([l1])

4. **Treinamento**  
    O modelo é treinado por 500 épocas com a seguinte configuração:

    ```python
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
    istory = model.fit(celsius, fahrenheit, epochs=500, verbose=False)

5. **Predição**  
    Após o treinamento, o modelo realiza uma previsão para 100°C:

    ```python
    predicao = model.predict(np.array([100.0]))
    print(f"Predição para 100ºC: {predicao[0][0]:.2f}°F")
    
6. **Visualização**  
    Um gráfico da perda (loss) ao longo das épocas de treinamento é gerado:

    ```python
    plt.xlabel('Épocas')
    plt.ylabel("Perda")
    plt.plot(history.history['loss'])
    plt.show()

## 🚀 Como Usar

1. **Instale as Dependências**  
    Certifique-se de que o Python e as bibliotecas necessárias estão instaladas:

    ```python
    pip install tensorflow numpy matplotlib

2. **Execute o Script**  
    Rode o script no seu ambiente Python:

    ```python
    python previsao_temperatura.py

3. **Resultados**  
- A previsão da temperatura em Fahrenheit para 100°C será exibida no terminal.
- Um gráfico mostrando a redução da perda ao longo do treinamento será exibido.

## 🛠️ Resultados e Insights

- **Previsão do Modelo**  
    O modelo aprende a aproximar a fórmula F= C × 1.8 + 32

- **Pesos Treinados**  
    Os pesos do modelo, que são exibidos após o treinamento, aproximam-se dos coeficientes da fórmula matemática.

- **Gráfico de Perda**   
    O gráfico ilustra a redução da perda durante o treinamento, indicando a melhoria do modelo.

