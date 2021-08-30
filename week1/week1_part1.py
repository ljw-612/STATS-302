import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## load data
def load_data(data_file, train_test_proportion):
    data = pd.read_csv(data_file, encoding='utf-8')
    data = data.sample(frac=1.0); data = data.reset_index(drop=True)
    num_data = len(data)
    # Divide the training and testing sets
    p = train_test_proportion
    # Divide the output set
    y = data.iloc[:, -1]
    y_train = y[0 : int(p * num_data)].reset_index(drop=True)
    y_test = y[int(p * num_data):].reset_index(drop=True)
    y_train = np.array([y_train]).T
    y_test = np.array([y_test]).T

    # Divide the input set
    data.drop(data.columns[[-1]], axis=1, inplace=True)
    x = data
    x_train = x[0 : int(p * num_data)].reset_index(drop=True)
    x_test = x[int(p * num_data):].reset_index(drop=True)

    return x_train, x_test, y_train, y_test, num_data

def error_calculation(theta_list, x_train, y_train, x_test, y_test, num_data_x_train, num_data_x_test):
    # Calculate the MSE for training data
    h_x_train = np.matmul(x_train, theta_list)
    MSE_x_train = sum( (h_x_train - y_train)**2 ) / (2 * num_data_x_train)
    # Calculate the MSE for testing data
    h_x_test = np.matmul(x_test, theta_list)
    MSE_x_test = sum( (h_x_test - y_test)**2 ) / (2 * num_data_x_test)
    return MSE_x_train[0], MSE_x_test[0]

def gradient_descent(x_train, x_test, y_train, y_test, num_data, num_iter=50, alpha=0.00001):
    # get how long is the data (row number)
    num_data_x_train = len(x_train); num_data_x_test = len(x_test)
    # Add the "1" column to the input matrix
    len_x_train = x_train.shape[1]; one_array_train = [1 for i in range(num_data_x_train)]
    x_train = np.array(x_train).T; x_train = np.concatenate([[one_array_train], x_train])
    x_train = x_train.T
    # Add the "1" column to the input matrix
    len_x_test = x_test.shape[1]; one_array_test = [1 for i in range(num_data_x_test)]
    x_test = np.array(x_test).T; x_test = np.concatenate([[one_array_test], x_test])
    x_test = x_test.T

    theta_list = np.zeros((len_x_train + 1, 1), dtype=float)
    h = 0
    for i in range(num_iter):
        h = np.matmul(x_train, theta_list)
        for i in range(0, len_x_train + 1):
            if i == 0:
                theta_list[i] = theta_list[i] - alpha * (1 / num_data) * sum(h - y_train)
            else:
                m = np.matmul( np.array([x_train[:,i]]), (h - y_train))
                theta_list[i] = theta_list[i] - alpha * (1 / num_data) * m[0][0]

    MSE_x_train, MSE_x_test = error_calculation(theta_list=theta_list, x_train=x_train, y_train=y_train, x_test=x_test,
                      y_test=y_test, num_data_x_train=num_data_x_train, num_data_x_test=num_data_x_test)
    return theta_list, MSE_x_train, MSE_x_test

def run(num_iter, alpha, train_test_proportion):
    x_train, x_test, y_train, y_test, num_data = load_data(data_file='boston_house_prices-rev-norm.csv',
                                                           train_test_proportion=train_test_proportion)
    theta_list, MSE_x_train, MSE_x_test = gradient_descent(x_train, x_test, y_train, y_test,
                                                           num_data=num_data, num_iter=num_iter, alpha=alpha)
    return theta_list, MSE_x_train, MSE_x_test

if __name__ == '__main__':
    num_iter = 500
    alpha = 0.001
    train_test_proportion = 0.8
    theta_list, MSE_x_train, MSE_x_test = run(num_iter=num_iter, alpha=alpha, train_test_proportion=train_test_proportion)
    print('Theta_list: \n', theta_list)
    print('MSE_train: ', MSE_x_train)
    print('MSE_test: ', MSE_x_test)
