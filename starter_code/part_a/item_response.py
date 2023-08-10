import random

from matplotlib import pyplot as plt
import numpy as np
import sys

sys.path.append('../')
from starter_code.utils import *
import math


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

    # for i in range(len(data['user_id'])):
    #     if data['is_correct'] == 1:
    #         log_lklihood += np.log(sigmoid(theta[data['user_id'][i]] - beta[data['question_id'][i]]))
    #     else:
    #         # print(theta[data['user_id'][i])
    #         log_lklihood += np.log(1 - sigmoid(theta[data['user_id'][i]] - beta[data['question_id'][i]]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            log_lklihood += (theta[i] - beta[j]) - np.log(1 + np.exp(theta[i] - beta[j]))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def grd_wrt_theta_helper(cij, p_cij):
    if math.isnan(cij):  # we don't want the model to focus on empty data
        return 0
    return (cij * (1 - p_cij)) + ((1 - cij) * p_cij)


def grd_wrt_beta_helper(cij, p_cij):
    if math.isnan(cij):
        return 0
    return - (cij * (1 - p_cij)) - ((1 - cij) * p_cij)

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
    n2 = n//2
    m2 = m//2
    lam = 0.0001
    reg = 0
    for i in range(n):
        gr_theta_i = 0.0
        for j in random.sample(range(m), m2):
            p_cij = sigmoid(theta[i] - beta[j])
            gr_theta_i += grd_wrt_theta_helper(data[i, j], p_cij)
        reg = lam * (theta[i] - beta).sum() * -1
        theta[i] += (lr * gr_theta_i + reg)

    for j in range(m):
        gr_beta_j = 0.0
        for i in random.sample(range(n), n2):
            p_cij = sigmoid(theta[i] - beta[j])
            gr_beta_j += grd_wrt_beta_helper(data[i, j], p_cij)
        reg = lam * (theta - beta[j]).sum()
        beta[j] += (lr * gr_beta_j + reg)
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
    return theta, beta, val_acc_lst


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
    print(np.array(pred).sum() / len(pred))
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    # print(train_data.shape)
    iteration = 8
    learn = 0.1
    theta, beta, acc = irt(sparse_matrix, val_data, learn, iteration)
    test_acc = evaluate(test_data, theta, beta)
    print("Final Validation accuracy: " + str(acc))
    print("Final Test Accuracy: " + str(test_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    j_s = train_data.get('question_id')[0:3]
    print(j_s)

    p_j = np.zeros((3, len(theta)))

    sorted_theta_idx = np.argsort(theta)
    sorted_theta = theta[sorted_theta_idx]

    for j in range(3):
        for i in range(len(sorted_theta)):
            p_j[j][i] = sigmoid(sorted_theta[i] - beta[j_s[j]])
    for i in range(3):
        plt.plot(sorted_theta, p_j[i], label='Question ' + str(j_s[i]))

    plt.xlabel('Theta')
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
