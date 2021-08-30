import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## load data

def load_data(data_file, train_test_proportion):
    data = pd.read_csv(data_file, encoding='utf-8')
    data = data.sample(frac=1.0);
    data = data.reset_index(drop=True)
    num_data = len(data)
    # Divide the training and testing sets
    p = train_test_proportion
    # Divide the output set
    y = data.iloc[:, -1]
    y_train = y[0: int(p * num_data)].reset_index(drop=True)
    y_test = y[int(p * num_data):].reset_index(drop=True)
    y_train = np.array([y_train]).T
    y_test = np.array([y_test]).T
    # Divide the input set

    data.drop(data.columns[[-1]], axis=1, inplace=True)
    return data, y, num_data


def gradient_descent(data, y, num_data, num_iter=50, alpha=0.00001):
    len_data = data.shape[1];
    one_array = [1 for i in range(num_data)]
    x = np.array(data).T;
    x = np.concatenate([[one_array], x])
    x = x.T

    theta_list = np.zeros((14, 1), dtype=float)
    h = 0
    for i in range(num_iter):
        h = np.matmul(x, theta_list)
        for i in range(0, len_data + 1):
            if i == 0:
                theta_list[i] = theta_list[i] - alpha * (1 / num_data) * sum(h - y)

            else:
                m = np.matmul(np.array([x[:, i]]), (h - y))
                theta_list[i] = theta_list[i] - alpha * (1 / num_data) * m[0][0]

    # print(theta_list)
    return


def run(num_iter, alpha, train_test_proportion):
    data, y, num_data = load_data(data_file='boston_house_prices-rev-norm.csv',
                                  train_test_proportion=train_test_proportion)
    len_data = gradient_descent(data=data, y=y, num_data=num_data, num_iter=num_iter, alpha=alpha)
    return len_data


if __name__ == '__main__':
    num_iter = 500
    alpha = 0.001
    train_test_proportion = 0.7
    len_data = run(num_iter=num_iter, alpha=alpha, train_test_proportion=train_test_proportion)
    print(len_data)
