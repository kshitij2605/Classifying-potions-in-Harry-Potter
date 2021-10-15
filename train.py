import csv
import numpy as np
import concurrent.futures



def import_data():
    X = np.genfromtxt("train_X_lg_v2.csv", delimiter=',', dtype=np.float64, skip_header=1)
    Y = np.genfromtxt("train_Y_lg_v2.csv", delimiter=',', dtype=np.float64)
    return X, Y


def get_train_data_for_class(train_X, train_Y, class_label):
    class_X = np.copy(train_X)
    class_Y = np.copy(train_Y)
    class_Y = np.where(class_Y == class_label, 1, 0)
    return class_X, class_Y


def calculate_accuracy(pred_Y, actual_Y):
    correct = 0
    for i in range(len(pred_Y)):
        if pred_Y[i] == actual_Y[i]:
            correct += 1

    return correct / len(pred_Y)


def calculate_precision(pred_Y, actual_Y):
    TP = 0
    for i in range(len(pred_Y)):
        if pred_Y[i] == 1 and actual_Y[i] == 1:
            TP += 1

    return TP / (pred_Y[pred_Y == 1]).size


def calculate_recall(pred_Y, actual_Y):
    TP = 0
    for i in range(len(pred_Y)):
        if pred_Y[i] == 1 and actual_Y[i] == 1:
            TP += 1

    return TP / (actual_Y[actual_Y == 1]).size


def f1_score(pred_Y, actual_Y):
    P = calculate_precision(pred_Y, actual_Y)
    R = calculate_recall(pred_Y, actual_Y)

    return 2 * P * R / (P + R)


def sigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    return s


def compute_gradient_of_cost_function(X, Y, W, b):
    m = len(X)

    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    dW = 1 / m * np.dot((A - Y).T, X)
    db = 1 / m * np.sum(A - Y)
    dW = dW.T
    return dW, db


def compute_cost(X, Y, W, b):
    m = len(X)
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    A[A == 1] = 0.99999
    A[A == 0] = 0.00001
    cost = -1/m * np.sum(np.multiply(Y, np.log(A)) + np.multiply((1-Y), np.log(1-A)))
    return cost


def optimize_weights_using_gradient_descent(X, Y, W, b, class_label, learning_rate, limit):
    #ones = np.ones(len(X))
    #X = np.insert(X, 0, ones, axis=1)
    previous_iter_cost = 0
    iter_no = 0
    while True:
        iter_no += 1
        dW, db = compute_gradient_of_cost_function(X, Y, W, b)
        W = W - learning_rate * dW
        b = b - learning_rate * db
        cost = compute_cost(X, Y, W, b)
        if iter_no % 10000 == 0:
            print(iter_no, cost, class_label)

        if abs(previous_iter_cost - cost) < limit:
            print(iter_no, cost, class_label)
            break

        previous_iter_cost = cost
    return W, b


def train_model(X, Y, class_label, learning_rate, limit):
    #X = np.insert(X, 0, 1, axis = 1)
    X, Y = get_train_data_for_class(X, Y, class_label)
    Y = Y.reshape(len(X), 1)
    W = np.zeros((X.shape[1],1))
    b = 1
    W, b = optimize_weights_using_gradient_descent(X, Y, W, b, class_label, learning_rate, limit)
    # pred_Y = predict_labels(X, W, b, threshold_values[class_label])
    #
    #
    # print("accuracy "+str(calculate_accuracy(pred_Y, Y)))
    # print("precision "+str(calculate_precision(pred_Y, Y)))
    # print("recall "+ str(calculate_recall(pred_Y, Y)))
    # print("f1-score "+ str(f1_score(pred_Y, Y))+"\n\n\n")
    return np.vstack((np.array([[b]]), W))


def save_model(weights, weights_file_name):
    weights = weights.reshape(4,21)
    with open(weights_file_name, 'w', newline='') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()


if __name__ == '__main__':
    X, Y = import_data()
    #weights = np.ones((4, X.shape[1]+1, 1))
    learning_rate = [0.007, 0.007, 0.007, 0.007]
    limit = [0.00000001, 0.00000001, 0.00000001, 0.00000001]
    class_label = [0, 1, 2, 3]
    X = [X,X,X,X]
    Y = [Y,Y,Y,Y]
    # for class_label in range(4):
    #     print("class:", class_label)
    #     weights[class_label] = train_model(X, Y, class_label, learning_rate[class_label])
    #
    # save_model(weights, "WEIGHTS_FILE.csv")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        weights = list(executor.map(train_model, X, Y, class_label, learning_rate, limit))

    weights = np.array(weights)
    save_model(weights, "WEIGHTS_FILE.csv")
