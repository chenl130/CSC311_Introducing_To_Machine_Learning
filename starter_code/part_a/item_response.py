from utils import *

import numpy as np
import copy
import matplotlib.pyplot as plt


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
    N, M = 542, 1774
    temp_matrix = copy.deepcopy(data).toarray()
    temp_matrix = np.nan_to_num(temp_matrix, nan=0)
    m1 = theta@np.ones((1, M))
    m2 = (beta@np.ones((1, N))).T
    m3 = m1-m2

    m4 = np.log1p(np.exp(m3))
    temp_coeff = copy.deepcopy(data).toarray()
    temp_coeff[temp_coeff == 0] = 1.
    temp_coeff = np.nan_to_num(temp_coeff)
    temp_coeff[temp_coeff != 0] = 1
    m5 = np.multiply(temp_matrix, m3) - np.multiply(temp_coeff, m4)
    nllk = -1*m5.sum()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return nllk


def update_theta_beta_alpha(data, lr, theta, beta, alpha):
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
    N, M = 542, 1774
    temp_matrix = copy.deepcopy(data).toarray()
    temp_matrix = np.nan_to_num(temp_matrix, nan=0)

    temp_coeff = copy.deepcopy(data).toarray()
    temp_coeff[temp_coeff == 0] = 1.
    temp_coeff = np.nan_to_num(temp_coeff)

    theta_mat = theta @ np.ones((1, M))
    beta_mat = (beta @ np.ones((1, N))).T
    alpha_mat = (alpha @ np.ones((1, N))).T

    alpha_mask = np.multiply(temp_matrix, alpha_mat)
    diff_mask = np.multiply(temp_matrix, theta_mat-beta_mat)

    alpha_sum_user = alpha_mask.sum(axis=1).reshape(N, 1)

    combine_coef = np.multiply(alpha_mat, theta_mat-beta_mat)
    sig = sigmoid(combine_coef)
    second_part = np.multiply(temp_coeff, sig)
    sum_log_user = second_part.sum(axis=1).reshape(N, 1)

    alpha_sum_item = alpha_mask.sum(axis=0).reshape(M, 1)
    sum_log_item = second_part.sum(axis=0).reshape(M, 1)

    diff_sum_item = diff_mask.sum(axis=0).reshape(M, 1)
    diff_log = np.multiply(sig, theta_mat-beta_mat)
    sum_diff_log = np.multiply(temp_coeff, diff_log).sum(axis=0).reshape(M, 1)

    theta = theta - lr*(sum_log_user - alpha_sum_user)
    beta = beta - lr*(alpha_sum_item - sum_log_item)
    alpha = alpha - lr*(sum_diff_log - diff_sum_item)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, alpha


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
    N, M = 542, 1774
    temp_matrix = copy.deepcopy(data).toarray()
    temp_matrix = np.nan_to_num(temp_matrix, nan=0)

    temp_coeff = copy.deepcopy(data).toarray()
    temp_coeff[temp_coeff == 0] = 1.
    temp_coeff = np.nan_to_num(temp_coeff)

    theta_mat = theta @ np.ones((1, M))
    beta_mat = (beta @ np.ones((1, N))).T

    sum_user = temp_matrix.sum(axis=1).reshape(N, 1)

    combine_coef = theta_mat-beta_mat
    sig = sigmoid(combine_coef)
    second_part = np.multiply(temp_coeff, sig)
    sum_log_user = second_part.sum(axis=1).reshape(N, 1)

    sum_item = temp_matrix.sum(axis=0).reshape(M, 1)
    sum_log_item = second_part.sum(axis=0).reshape(M, 1)

    theta = theta - lr*(sum_log_user - sum_user)
    beta = beta - lr*(sum_item - sum_log_item)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt_with_alpha(data, val_data, lr, iterations):
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
    N = 542
    M = 1774
    theta = np.ones((N, 1))
    beta = np.ones((M, 1))
    alpha = np.ones((M, 1))
    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, alpha = update_theta_beta_alpha(data, lr, theta, beta, alpha)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, alpha, val_acc_lst


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
    N = 542
    M = 1774
    theta = np.ones((N, 1))
    beta = np.ones((M, 1))
    alpha = np.ones((M, 1))
    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta, alpha):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (alpha[q] * (theta[u] - beta[q])).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


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

    # Experiment 1: tune learning rate
    # val_acc_lst_lowrate = irt(sparse_matrix, val_data, 0.0005, 500)[2]
    # train_acc_lst_lowrate = irt(sparse_matrix, train_data, 0.0005, 500)[2]
    # plt.figure()
    # plt.plot(np.arange(500), val_acc_lst_lowrate)
    # plt.plot(np.arange(500), train_acc_lst_lowrate)
    # plt.xlabel('number of iterations')
    # plt.ylabel('accuracy')
    # plt.legend(["validation accuracy", "training accuracy"])
    # plt.title('learning rate experiment: lr = 0.0005')
    # plt.show()

    # val_acc_lst_midrate = irt(sparse_matrix, val_data, 0.001, 500)[2]
    # train_acc_lst_midrate = irt(sparse_matrix, train_data, 0.001, 500)[2]
    # plt.figure()
    # plt.plot(np.arange(500), val_acc_lst_midrate)
    # plt.plot(np.arange(500), train_acc_lst_midrate)
    # plt.xlabel('number of iterations')
    # plt.ylabel('accuracy')
    # plt.legend(["validation accuracy", "training accuracy"])
    # plt.title('learning rate experiment: lr = 0.001')
    # plt.show()

    # val_acc_lst_highrate = irt(sparse_matrix, val_data, 0.01, 500)[2]
    # train_acc_lst_highrate = irt(sparse_matrix, train_data, 0.01, 500)[2]
    # plt.figure()
    # plt.plot(np.arange(500), val_acc_lst_highrate)
    # plt.plot(np.arange(500), train_acc_lst_highrate)
    # plt.xlabel('number of iterations')
    # plt.ylabel('accuracy')
    # plt.legend(["validation accuracy", "training accuracy"])
    # plt.title('learning rate experiment: lr = 0.01')
    # plt.show()

    # Experiment 2: number of iterations
    # val_acc_lst_lownumi = irt(sparse_matrix, val_data, 0.001, 100)[2]
    # train_acc_lst_lownumi = irt(sparse_matrix, train_data, 0.001, 100)[2]
    # plt.figure()
    # plt.plot(np.arange(100), val_acc_lst_lownumi)
    # plt.plot(np.arange(100), train_acc_lst_lownumi)
    # plt.xlabel('number of iterations')
    # plt.ylabel('accuracy')
    # plt.legend(["validation accuracy", "training accuracy"])
    # plt.title('number of iteration experiment: n = 100')
    # plt.show()

    #val_acc_lst_midrate = irt(sparse_matrix, val_data, 0.001, 1000)[2]
    #train_acc_lst_midrate = irt(sparse_matrix, train_data, 0.001, 1000)[2]
    # plt.figure()
    # plt.plot(np.arange(300), val_acc_lst_midnumi)
    # plt.plot(np.arange(300), train_acc_lst_midhnumi)
    # plt.xlabel('number of iterations')
    # plt.ylabel('accuracy')
    # plt.legend(["validation accuracy", "training accuracy"])
    # plt.title('number of iteration experiment: n = 300')
    # plt.show()

    #val_acc_lst_highnumi = irt(sparse_matrix, val_data, 0.001, 1000)[2]
    #train_acc_lst_highnumi = irt(sparse_matrix, train_data, 0.001, 1000)[2]
    # plt.figure()
    # plt.plot(np.arange(1000), val_acc_lst_highnumi)
    # plt.plot(np.arange(1000), train_acc_lst_highnumi)
    # plt.xlabel('number of iterations')
    # plt.ylabel('accuracy')
    # plt.legend(["validation accuracy", "training accuracy"])
    # plt.title('number of iteration experiment: n = 1000')
    # plt.show()

    #plot training curve of log-likelihood vs. iterations of training and
    #validation sets
    #lld_train_lst = []
    #lld_valid_lst = []
    #theta = np.ones((542, 1))
    #beta = np.ones((1774, 1))
   # for i in range(300):
        #lld_train = -1*neg_log_likelihood(sparse_matrix, theta=theta, beta=beta)
        #lld_valid = 0
        #for i, q in enumerate(val_data["question_id"]):
           # u = val_data["user_id"][i]
           # m = val_data["is_correct"][i]
            #lld_valid += m*(theta[u] - beta[q]) - np.log1p(np.exp(theta[u] - beta[q]))
       # lld_train_lst.append(lld_train)
       # lld_valid_lst.append(lld_valid)
       # theta, beta = update_theta_beta(sparse_matrix, 0.001, theta, beta)

    #plt.figure()
   # plt.plot(np.arange(300), lld_train_lst)
    #plt.plot(np.arange(300), lld_valid_lst)
    #plt.xlabel('number of iterations')
    #plt.ylabel('log-likelihood')
    #plt.legend(["training set", "validation set"])
    #plt.title('log-likelihood vs. number of iterations')
    #plt.show()


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)
    #####################################################################

    private_test = load_private_test_csv("C:/Users/16475/Desktop/CSC311/csc311_final_project/starter_code/data")

    N = 542
    M = 1774
    theta = np.ones((N, 1))
    beta = np.ones((M, 1))
    alpha = np.ones((M, 1))

    for i in range(76):
        theta, beta = update_theta_beta(sparse_matrix, 0.001, theta, beta)
    final_acc_valid = evaluate(val_data, theta, beta, alpha)
    final_acc_test = evaluate(test_data, theta, beta, alpha)

    print("final accuracy of validation set is " + str(final_acc_valid))
    print("final accuracy of test set is " + str(final_acc_test))
    threshold = 0.5
    predictions = []
    for i in range(len(private_test["question_id"])):
        u = private_test["user_id"][i]
        q = private_test["question_id"][i]
        x = (alpha[q] * (theta[u] - beta[q])).sum()
        p_a = sigmoid(x)
        predictions.append(p_a >= 0.5)
    private_test["is_correct"] = predictions
    save_private_test_csv(private_test)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    """
    # Implement part (d)
    js = np.random.choice(1774, 5, replace=False)
    betas = []
    for j in js:
        betas.append(beta[j])

    theta.sort(axis=0)

    plt.figure()
    for i in range(5):
        prob = sigmoid(np.subtract(theta, betas[i]))
        plt.plot(theta, prob)
    plt.xlabel('theta')
    plt.ylabel('probabilitycorrect response')
    plt.legend(["question" + str(js[0]), "question" + str(js[1]),
                "question" + str(js[2]), "question" + str(js[3]),
                "question" + str(js[4])])
    plt.title('influence of learning rate on cross entropy - training set')
    plt.show()

    print("selected five questions has the following difficulty level: " +
          "question "+ str(js[0]) + "has difficulty" + str(betas[0]) +
          ",question "+ str(js[1]) + "has difficulty" + str(betas[1]) +
          ",question "+ str(js[2]) + "has difficulty" + str(betas[2]) +
          ",question "+ str(js[3]) + "has difficulty" + str(betas[3]) +
          ",question "+ str(js[4]) + "has difficulty" + str(betas[4]))

    """


if __name__ == "__main__":
    main()
