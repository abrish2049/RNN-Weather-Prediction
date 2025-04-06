# Temperature Prediction with Deep Learning Models (RNN, LSTM, GRU)

This repository demonstrates the use of three deep learning models — **SimpleRNN**, **LSTM**, and **GRU** — to predict temperature (`T (degC)`) using various weather-related features. The models are implemented using Keras (TensorFlow backend), and data preprocessing steps, including feature scaling and handling missing values, are included.

## Table of Contents

- [Introduction](#introduction)
- [Data Loading & Preprocessing](#data-loading-and-preprocessing)
- [Model Architectures](#model-architectures)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction

This project aims to predict the temperature (`T (degC)`) based on other weather-related features such as atmospheric pressure (`p (mbar)`), relative humidity (`sh (g/kg)`), wind speed (`wv (m/s)`), and others. The dataset spans multiple years (2009-2016) of weather data, and the goal is to compare the performance of three different deep learning models:

- **Simple RNN (Recurrent Neural Network)**
- **LSTM (Long Short-Term Memory)**
- **GRU (Gated Recurrent Unit)**

## Data Loading & Preprocessing

The dataset used for this project contains weather-related features and is stored in a CSV file. The preprocessing steps include:

- **Handling missing values**: Interpolating missing data.
- **Feature scaling**: Normalizing the feature values using Min-Max Scaling.
- **Time series transformation**: Using sliding windows to create time-dependent features for model input.

### Data Structure

The dataset contains multiple weather-related features, including:

- `p (mbar)`: Atmospheric pressure
- `T (degC)`: Temperature (target variable)
- `Tpot (K)`: Potential temperature
- `rh (%)`: Relative humidity
- `VPmax (mbar)`: Maximum vapor pressure
- `wv (m/s)`: Wind speed

The target variable is the `T (degC)` column, representing the temperature.

## Model Architectures

### SimpleRNN

The **SimpleRNN** model consists of two RNN layers. Each layer uses `tanh` activation and returns sequences for further processing in the next RNN layer. The output is passed through a dense layer to predict the temperature.

```python
SimpleRNN_model = keras.Sequential([
    keras.layers.SimpleRNN(125, activation='tanh', return_sequences=True, input_shape=(hist_window, n_features)),
    keras.layers.SimpleRNN(125, activation='tanh', return_sequences=True),
    keras.layers.Dense(1)
])
```
### LSTM
The LSTM model uses two LSTM layers to capture long-term dependencies in the time series data. Like the SimpleRNN, the output is passed through a dense layer.

```python
LSTM_model = keras.Sequential([
    keras.layers.LSTM(125, activation='tanh', return_sequences=True, input_shape=(hist_window, n_features)),
    keras.layers.LSTM(125, activation='tanh', return_sequences=True),
    keras.layers.Dense(1)
])


```
### GRU
The GRU model utilizes two GRU layers, which are similar to LSTMs but with fewer parameters.

```python
GRU_model = keras.Sequential([
    keras.layers.GRU(125, activation='tanh', return_sequences=True, input_shape=(hist_window, n_features)),
    keras.layers.GRU(125, activation='tanh', return_sequences=True),
    keras.layers.Dense(1)
])



```


## Training and Evaluation

The models are trained using early stopping to prevent overfitting. The training process involves using the Mean Squared Error (MSE) loss function and Adam optimizer.

```python
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping])
```

The models are evaluated using Root Mean Squared Error (RMSE) to compare the performance of each model.

```python
from sklearn.metrics import mean_squared_error
import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

```


## Results

After training the models, the predictions are compared to the true values, and their performance is evaluated based on RMSE.

```python
SimpleRNN_y_pred = Y_scaler.inverse_transform(SimpleRNN_pred[:,-1])
LSTM_y_pred = Y_scaler.inverse_transform(LSTM_pred[:,-1])
GRU_y_pred = Y_scaler.inverse_transform(GRU_pred[:,-1])

# RMSE Results
print("SimpleRNN RMSE: ", rmse(SimpleRNN_y_test, SimpleRNN_y_pred))
print("LSTM RMSE: ", rmse(LSTM_y_test, LSTM_y_pred))
print("GRU RMSE: ", rmse(GRU_y_test, GRU_y_pred))

```


## Installation 


To run the project locally, follow the steps below:

- Clone Repo


```python
git clone https://github.com/abrish2049/RNN-Weather-Prediction.git
cd RNN-Weather-Prediction
```


- Install Dependencies 

I would recommend creating a new enviroment , Tensorflow is known for messing with your packages 


```python
pip install -r requirements.txt
```



## Usage

Once the notebook is running, you can train and evaluate the models by running the cells sequentially. You will be able to see the predictions for each model and compare their performance.


## Contributing

Feel free to fork the repository and contribute to the project. If you find any issues or have suggestions, please open an issue or submit a pull request.


## License
This project is licensed under the MIT License - see the LICENSE file for details.