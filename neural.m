clear;
clc;

% Read the CSV data
data = readtable('./data/apple_stock_data.csv'); % Assuming 'stock_data.csv' is the filename

% Extract the Close variable as the target
target = data.Close;

% Remove Date and Adj Close columns as they are not needed for prediction
data(:, {'Date', 'AdjClose'}) = [];

% Convert table to matrix
data = table2array(data);

% Normalize data (optional but often recommended for neural networks)
data_normalized = normalize(data);
target_normalized = normalize(target);

% Number of data points
num_data = size(data, 1);

% Split ratio (e.g., 70% training, 30% testing)
train_ratio = 0.75;
train_size = round(train_ratio * num_data);

% Training data
train_data = data_normalized(1:train_size, :);
train_target = target_normalized(1:train_size, :);

% Testing data
test_data = data_normalized(train_size+1:end, :);
test_target = target_normalized(train_size+1:end, :);

% Define the neural network architecture
hidden_layer_size = 40;
net = feedforwardnet(hidden_layer_size);

% Set training options
train_options = trainingOptions('adam', ...  % Use Adam optimizer (or other optimizer)
    'MaxEpochs', 3000, ...  % Number of epochs to train
    'Verbose', true, ...   % Show training progress
    'Plots', 'training-progress');  % Show training progress plot

% Train the neural network with specified options
[net, tr] = train(net, train_data', train_target', [], [], [], [], [], [], train_options);
% Predict on test data
predictions = net(test_data');
% Measure the performance (e.g., Mean Squared Error)
mse = mse(predictions - test_target');
disp(['Mean Squared Error: ', num2str(mse)]);

% Denormalize the predicted values
predicted_close = predictions' * std(target) + mean(target);
actual_close = target(train_size+1:end);

% Plotting
figure;
plot(actual_close, '-o', 'DisplayName', 'Actual Close');
hold on;
plot(predicted_close, '-x', 'DisplayName', 'Predicted Close');
hold off;
xlabel('Data Points');
ylabel('Close');
title('Actual vs Predicted Close');
legend;
grid on;

