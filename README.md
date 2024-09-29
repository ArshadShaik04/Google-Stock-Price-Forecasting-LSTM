# Google Stock Analysis & Prediction Using LSTM

In this project, we analyze Google's stock market data and use LSTM (Long Short-Term Memory) networks to predict future stock prices. This analysis aims to provide informed decision-making tools based on data patterns and trends.

# Table of Contents

1. [Introduction](#introduction)
2. [Data Exploration](#data-exploration)
3. [Data Preprocessing](#data-preprocessing)
4. [LSTM Dataset Creation](#lstm-dataset-creation)
5. [Model Building](#model-building)
6. [Model Training](#model-training)
7. [Evaluation and Visualization](#evaluation-and-visualization)
8. [Future Predictions](#future-predictions)
9. [Dataset](#dataset)
10. [Conclusion](#conclusion)

---

## Introduction

This project explores Google's stock market dynamics using LSTM networks to forecast future trends. The focus is on understanding past data patterns to make accurate predictions.

---

## Data Exploration

### Initial Analysis

The dataset contains 14 columns and 1257 rows of data, including attributes like close, high, low, open, and volume. This initial exploration helps identify key patterns and trends.

---

## Data Preprocessing

### Data Cleaning and Preparation

- **Handling Missing Values**: Missing values are handled using forward fill or interpolation.
- **Scaling Data**: Data is normalized using MinMaxScaler to ensure features contribute equally.

---

## LSTM Dataset Creation

### Data Sequencing

We create sequences of data suitable for LSTM training by sliding a window over the time series data, ensuring effective temporal learning.

---

## Model Building

### Constructing the Model

An LSTM model is built using LSTM, Dropout, and Dense layers. This architecture captures temporal dependencies and prevents overfitting.

---

## Model Training

### Training the Model

The model is trained over 100 epochs using historical data to enhance its predictive capabilities.

---

## Evaluation and Visualization

### Model Performance

The model's performance is evaluated using metrics like RMSE. Visual comparisons between actual and predicted prices provide insights into the model's effectiveness.

---

## Future Predictions

### Forecasting

The model predicts the next 20 days' open and close stock prices using the most recent data.

---

## Dataset

The dataset used for this project is available on Kaggle:

- [Google Stock Prediction Dataset](https://www.kaggle.com/datasets/shreenidhihipparagi/google-stock-prediction)

---

## Conclusion

This project demonstrates the use of LSTM networks for stock price forecasting, providing valuable insights for financial decision-making.
