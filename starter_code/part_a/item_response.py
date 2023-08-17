from matplotlib import pyplot as plt
import numpy as np
import math
import sys
sys.path.append('../')
from utils import *


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                p = sigmoid(theta[i] - beta[j])
                log_lklihood += (data[i, j] * np.log(p)) - ((1 - data[i, j]) * np.log(1 - p))
        
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    n, m = data.shape
    theta_tile = np.tile(np.reshape(theta, (n, 1)), (1, m))
    beta_tile = np.tile(beta, (n, 1))
    z = theta_tile - beta_tile
    sigmoid_all = np.vectorize(sigmoid)
    y = sigmoid_all(z)
    theta = theta + (lr * np.nansum(data - y, axis=1).T)
    beta = beta + (lr * np.nansum(y - data, axis=0))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.randn(data.shape[0])
    beta = np.random.randn(data.shape[1])

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, score


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q])
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################

    iteration = 60
    learn = 0.005
    theta, beta, acc = irt (sparse_matrix, val_data, learn, iteration)
    test_acc = evaluate (test_data, theta, beta)

    print("Final Validation accuracy: " + str(acc))
    print("Final Test Accuracy: " + str(test_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    j_s = train_data.get('question_id')[3:6]

    p_j = np.zeros((3, len(theta)))

    # sorted_theta_idx = np.argsort(theta)
    # sorted_theta = theta[sorted_theta_idx]
    # students_idx = np.argsort(train_data.get('user_id'))
    # students = train_data.get('user_id')[students_idx]
    
    for j in range(3):
        for i in range(sparse_matrix.shape[0]):
            p_j[j][i] = sigmoid(theta[i] - beta[j_s[j]])
    for i in range(3):
        plt.plot(range(sparse_matrix.shape[0]), p_j[i], label='Question ' + str(j_s[i]))

    plt.xlabel('Student ID')
    plt.ylabel('Probability of Correct Response')
    plt.title('Probability of Correct Response vs. Theta for Three Questions')
    plt.legend()
    plt.grid(True)
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
