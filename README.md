# Stock Market Prediction - MTH1060 Capstone Project
![Matlab Image](matlab.png)
## Introduction
Machine learning models in MATLAB that predicts stock market prices based on historical data. 

They utilise time series analysis and neural networks to evaluate the effectiveness in price prediction.

## Data Acquisition
Data was obtained from Yahoo Finance. The data spans from 1st July 2014 to 28th June 2024. The data is in the following general form:

```csv
Date,Open,High,Low,Close,Adj Close,Volume
2014-07-01,23.3799991607666,23.517499923706055,23.282499313354492,23.3799991607666,20.68042755126953,152892000
2014-07-02,23.467500686645508,23.514999389648438,23.272499084472656,23.3700008392334,20.671592712402344,113860000
2014-07-03,23.417499542236328,23.524999618530273,23.299999237060547,23.50749969482422,20.793209075927734,91567200
```

## Data Preprocessing
Data was imported into MatLab in bulk and stored in an array. This enables the use of iteration and control structures to access the data. Warnings are suppressed for cleaner output and if there are any missing values they are pointed out.

This allows for missing values to be dropped in order to ensure that operations are not skewed or curtailed by this data.

Furthermore, data is normalized between the range of 0 and 1 in order for our Time Series Analysis and Neural Network models to be able to work with our data. Only numerical columns are normalized. The column `date` is not affected in any way.

## Time Series Analysis

## Neural Networks
We decided to use a Neural Network that:

- Has 40 hidden nodes
- Trains for a maximum of 1000 epochs
- Uses the adam optimizer
- Utilises 75% of the data to train
- Utilises 25% of the data to test
- Uses the Mean Squared Error to test the error of the results


## Model Comparison and Analysis

### Time Series Analysis


### Neural Network
