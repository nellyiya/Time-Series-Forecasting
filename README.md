# Beijing PM2.5 Air Quality Forecasting Project

## Overview
This project tackles the problem of predicting fine particulate matter (PM2.5) concentrations in Beijing using historical air quality and weather data. Accurate forecasting of PM2.5 helps mitigate adverse health impacts and supports urban planning.

## Dataset
The dataset consists of hourly records from July 2013 to early 2015, including:
- Meteorological features: temperature, dew point, pressure, wind speed, and others.
- Target variable: PM2.5 concentration in µg/m³.

Files:
- `train.csv`: training dataset with features and PM2.5.
- `test.csv`: test dataset with features only.
- `sample_submission.csv`: submission format template.

## Approach
- Data preprocessing: missing value imputation, feature scaling via MinMaxScaler, and sequence creation using windows of 24 hours.
- Model design: a two-layer LSTM with dropout for regularization.
- Training: model trained using mean squared error loss and Adam optimizer for 20 epochs.
- Evaluation: RMSE metric used to assess performance both during training and on the Kaggle leaderboard.

## Results
- The model achieves an RMSE score in the target range of 2000-3000 on the private leaderboard.
- Experiments with different hyperparameters (learning rate, batch size, layers) were conducted to optimize results.
- Visualizations of training loss and prediction vs. actual values are included in the notebook.

## How to Run
1. Clone the repository.
2. Place `train.csv` and `test.csv` files in the working directory.
3. Install necessary dependencies:
4. Run the notebook or script to train the model and generate `submission.csv`.
5. Submit results to the Kaggle competition following the submission format.

## Repository Structure

/data
train.csv
test.csv
/notebooks
air_quality_forecasting.ipynb
/models
lstm_model.h5
/submission
submission.csv
README.md



## Future Improvements
- Experiment with more complex architectures (stacked LSTM, bidirectional LSTM).
- Feature engineering with lagged variables or external data.
- Hyperparameter optimization using grid or random search.
- Address challenges like vanishing gradients with gradient clipping.
- Deploy model for real-time air quality forecasting.

## References
- Kaggle Beijing PM2.5 Forecasting Challenge
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation.
- Documentation for TensorFlow and Keras LSTM layers.

---


