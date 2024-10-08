# Google Stock Price Prediction using LSTM-GRU Hybrid Model

This project involves building a machine learning model to predict Google's stock prices using a combination of LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) networks. The model aims to predict future stock prices based on historical data, leveraging time series analysis and various financial features.

## Project Overview

The goal of this project is to predict Google's stock closing prices using a deep learning approach. The model is built with a hybrid of LSTM and GRU layers, which are well-suited for handling sequential data such as stock prices. We also perform hyperparameter tuning using Keras Tuner to optimize the performance of the model.

### Key Features of the Project:

- **Data Preprocessing**: Historical stock data including open, close, high, low, and volume were used. Additional features such as moving averages, volatility, daily returns, and price range were added.
- **Feature Engineering**: Added important financial indicators such as 20-day and 50-day moving averages, volatility, and 200-day moving averages to enhance model performance.
- **Model Architecture**: The model consists of a combination of Bidirectional LSTM and GRU layers, with dropout layers added to prevent overfitting.
- **Hyperparameter Tuning**: Used Keras Tuner's RandomSearch to tune the hyperparameters for optimal performance.
- **Model Evaluation**: Evaluated the model using metrics like Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). Residual plots were also used to understand prediction errors.

## Dataset

The dataset used for this project includes historical stock prices of Google. The features used include:

- **Date**: The date of the stock data.
- **Open, Close, High, Low**: The respective stock prices.
- **Volume**: The trading volume on that particular day.

Additional derived features:

- **MA\_20, MA\_50, MA\_200**: 20-day, 50-day, and 200-day moving averages.
- **Volatility**: 20-day rolling standard deviation of the closing price.
- **Daily Return**: Percentage change in closing price.
- **Price Range**: Difference between the high and low prices.

## Model Architecture

The model was built using **Keras** and includes the following:

- **Bidirectional LSTM Layer**: Captures both past and future context in the sequence.
- **GRU Layer**: Captures relevant patterns in sequential data while using fewer computational resources compared to LSTM.
- **Dropout Layers**: Prevents overfitting by randomly setting a fraction of input units to zero during training.
- **Dense Layer**: Outputs the final predicted price.

## Hyperparameter Tuning

**Keras Tuner** was used for hyperparameter tuning, specifically using **RandomSearch**. The parameters tuned include:

- **Number of LSTM and GRU Units**
- **Dropout Rates**
- **Learning Rate**

The tuning process aimed to minimize the validation loss, and the best combination of hyperparameters was used to train the final model.

## Results

- **Root Mean Squared Error (RMSE)**: 41.02
- **Mean Absolute Error (MAE)**: 32.18

These values reflect a significant improvement over initial runs, showing that the model captures the trend in Google's stock prices quite well.

## Visualizations

1. **Training and Validation Loss**: Plotted to visualize model performance during training.
   ![Training and Validation Loss](Visualization%20Plots/Training%20and%20Validation%20Loss.png)

2. **Training and Validation MAE**: Plotted to visualize model MAE during training.
   ![Training and Validation MAE](Visualization%20Plots/Training%20and%20Validation%20MAE.png)

3. **Actual vs. Predicted Prices**: A comparison of the predicted prices with the actual stock prices.
   ![Actual vs Predicted Prices](Visualization%20Plots/Actual%20vs.%20Predicted%20Prices.png)

4. **Residual Plot**: Shows the difference between actual and predicted prices, providing insight into prediction errors.
   ![Residual Plot](Visualization%20Plots/Residual%20Plot.png)

## Requirements

- Python 3.7+
- Keras
- TensorFlow
- Keras Tuner
- Pandas, NumPy, Matplotlib, Scikit-learn

## Future Improvements

- **More Features**: Incorporate additional financial features such as sentiment analysis from news articles or social media to improve model accuracy.
- **Advanced Architectures**: Explore attention mechanisms to focus on the most relevant time steps in the sequence.
- **More Tuning**: Perform finer hyperparameter tuning to further optimize model performance.

## Conclusion

The hybrid LSTM-GRU model performs well in capturing the trends in Google's stock price, and hyperparameter tuning played a crucial role in improving its performance. The model can be further enhanced by incorporating additional data sources and more advanced deep learning techniques.

