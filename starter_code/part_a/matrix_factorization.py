from utils import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import numpy as np


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def squared_error_loss_reg(data, u, z, beta):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        uid = data["user_id"][i]
        loss += ((data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.) + beta * (np.sum(u[uid])**2 + np.sum(z[q])**2)
    return 0.5 * loss


def squared_error_loss_2bias(data, u, z, user_bias, question_bias, beta, mean):
    loss = 0
    for i, q in enumerate(data["question_id"]):
        uid = data["user_id"][i]
        mse = (data["is_correct"][i] - (np.sum(u[uid] * z[q]) +
                                        user_bias[uid] + question_bias[q] + mean)) ** 2.
        regularization = beta * (np.sum(u[uid])**2 + np.sum(z[q])**2 +
                                 user_bias[uid]**2 + question_bias[q]**2)
        loss += mse + regularization
    return 0.5 * loss


def squared_error_loss_3bias(data, u, z, user_bias, question_bias, male_bias,
                             female_bias, tran_bias, beta, gender_mat, mean):
    loss = 0
    for i, q in enumerate(data["question_id"]):
        uid = data["user_id"][i]
        if gender_mat[uid, q] == 1:
            gender_bias = male_bias
        elif gender_mat[uid, q] == 2:
            gender_bias = female_bias
        else:
            gender_bias = tran_bias
        mse = (data["is_correct"][i] - (np.sum(u[uid] * z[q]) +
                                        user_bias[uid] + question_bias[q] + mean + gender_bias)) ** 2.
        regularization = beta * (np.sum(u[uid])**2 + np.sum(z[q])**2 +
                                 user_bias[uid]**2 + question_bias[q]**2 + gender_bias**2)
        loss += mse + regularization
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]
    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]
    # update u
    u[n] = u[n] + lr * ((c - np.dot(u[n], z[q].T)) * z[q])

    # update z
    z[q] = z[q] + lr * ((c - np.dot(u[n], z[q].T)) * u[n])
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def update_u_z_reg(train_data, lr, u, z, beta):
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]
    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]
    # update u
    u[n] = u[n] + lr * ((c - np.dot(u[n], z[q].T)) * z[q] - beta * u[n])

    # update z
    z[q] = z[q] + lr * ((c - np.dot(u[n], z[q].T)) * u[n] - beta * z[q])
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def update_u_z_with_bias(train_data, lr, u, z, user_bias, question_bias, beta, mean):
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]
    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # update u
    u[n] = u[n] + lr * ((c - (np.dot(u[n], z[q].T) + mean + user_bias[n] + question_bias[q])) * z[q] - beta * u[n])

    user_bias[n] = user_bias[n] + lr * ((c - (np.dot(u[n], z[q].T) + mean + user_bias[n] +
                                              question_bias[q])) - beta * user_bias[n])
    # update z
    z[q] = z[q] + lr * ((c - (np.dot(u[n], z[q].T) + mean + user_bias[n] + question_bias[q])) * u[n] - beta * z[q])

    question_bias[q] = question_bias[q] + lr * ((c - (np.dot(u[n], z[q].T) + mean + user_bias[n] +
                                                      question_bias[q])) - beta * question_bias[q])
    return u, z, user_bias, question_bias


def update_u_z_with_bias_and_gender(train_data, lr, u, z, user_bias, question_bias,
                                    male_bias, female_bias, tran_bias, beta, gender_mat, mean):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :param user_bias: user_bias
    :param question_bias: question_bias
    :param male_bias: male_bias
    :param female_bias: female_bias
    :param tran_bias: tran_bias
    :param beta: Regularization term
    :param gender_mat: gender_mat
    :param mean: data mean
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]
    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]
    gid = gender_mat[n, q]
    y = np.dot(u[n], z[q].T)

    # update gender bias
    if gid == 1:
        gender_copy = male_bias
        male_bias = male_bias + lr * ((c - (y + mean + user_bias[n] + question_bias[q] + male_bias)) - beta * male_bias)
    elif gid == 2:
        gender_copy = female_bias
        female_bias = female_bias + lr * (
                (c - (y + mean + user_bias[n] + question_bias[q] + female_bias)) - beta * female_bias)
    else:
        gender_copy = tran_bias
        tran_bias = tran_bias + lr * ((c - (y + mean + user_bias[n] + question_bias[q] + tran_bias)) - beta * tran_bias)

    # update u
    u[n] = u[n] + lr * ((c - (y + mean + user_bias[n] + question_bias[q] + gender_copy)) * z[q] - beta * u[n])
    # update user_bias
    user_bias[n] = user_bias[n] + lr * (
            (c - (y + mean + user_bias[n] + question_bias[q] + gender_copy)) - beta * user_bias[n])
    # update z
    z[q] = z[q] + lr * ((c - (y + mean + user_bias[n] + question_bias[q] + gender_copy)) * u[n] - beta * z[q])
    # update question_bias
    question_bias[q] = question_bias[q] + lr * ((c - (y + mean + user_bias[n] +
                                                      question_bias[q] + gender_copy)) - beta * question_bias[q])
    return u, z, user_bias, question_bias, male_bias, female_bias, tran_bias


def sigmoid(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


def update_u_z_logistic(train_data, lr, u, z, beta):
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]
    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # update u
    y = sigmoid(np.dot(u[n], z[q].T))
    u[n] = u[n] + lr * ((c - y) * z[q] - beta * u[n])
    # update z
    z[q] = z[q] + lr * ((c - y) * u[n] - beta * z[q])
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als_logistic(train_data, k, lr, num_iteration, beta):
    u = np.random.uniform(low=-1 / np.sqrt(k), high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=-1 / np.sqrt(k), high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    for i in range(num_iteration):
        u, z = update_u_z_logistic(train_data, lr, u, z, beta)

    mat = sigmoid(u.dot(z.T))
    return mat


def als(train_data, k, lr, num_iteration):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    val_data = load_valid_csv("../data")
    mse_train = []
    mse_valid = []
    acc_valid = []
    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        if not i % 100:
            mse_train.append(squared_error_loss(train_data, u, z))
            mse_valid.append(squared_error_loss(val_data, u, z))
            mat = u.dot(z.T)
            acc_valid.append(sparse_matrix_evaluate(val_data, mat))
    mat = u.dot(z.T)

    figure, ar = plt.subplots(1, 3, figsize=(15, 6))
    ar[0].set_title("Vanilla MF on Training ")
    ar[0].set_xlabel('Iteration * 100')
    ar[0].set_ylabel('Squared Error Loss')
    ar[0].plot(mse_train)
    ar[1].set_title("Vanilla MF on Validation")
    ar[1].set_xlabel('Iteration * 1000')
    ar[1].set_ylabel('Squared Error Loss')
    ar[1].plot(mse_valid)
    ar[2].set_title("Accuracy on Validation")
    ar[2].set_xlabel('Iteration * 100')
    ar[2].set_ylabel('Accuracy')
    ar[2].plot(acc_valid)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def als_reg(train_data, k, lr, num_iteration, beta):
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    val_data = load_valid_csv("../data")
    mse_train = []
    mse_valid = []
    acc_valid = []
    for i in range(num_iteration):
        u, z = update_u_z_reg(train_data, lr, u, z, beta)
        if not i % 100:
            mse_train.append(squared_error_loss_reg(train_data, u, z, beta))
            mse_valid.append(squared_error_loss_reg(val_data, u, z, beta))
            mat = u.dot(z.T)
            acc_valid.append(sparse_matrix_evaluate(val_data, mat))
    mat = u.dot(z.T)

    figure, ar = plt.subplots(1, 3, figsize=(15, 6))
    ar[0].set_title("Vanilla MF on Training ")
    ar[0].set_xlabel('Iteration * 100')
    ar[0].set_ylabel('Squared Error Loss')
    ar[0].plot(mse_train)
    ar[1].set_title("Vanilla MF on Validation")
    ar[1].set_xlabel('Iteration * 1000')
    ar[1].set_ylabel('Squared Error Loss')
    ar[1].plot(mse_valid)
    ar[2].set_title("Accuracy on Validation")
    ar[2].set_xlabel('Iteration * 100')
    ar[2].set_ylabel('Accuracy')
    ar[2].plot(acc_valid)
    return mat


def als_with_bias(train_data, k, lr, num_iteration, beta):
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    user_bias = np.zeros((len(set(train_data["user_id"])), 1))

    question_bias = np.zeros((len(set(train_data["question_id"])), 1))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    val_data = load_valid_csv("../data")
    mse_train = []
    mse_valid = []
    valid_acc = []
    N, M = 542, 1774
    correct_count = 0
    for i in range(len(train_data["question_id"])):
        if train_data["is_correct"][i] == 1:
            correct_count += 1
    mean = correct_count / len(train_data["question_id"])
    for i in range(num_iteration):
        u, z, user_bias, question_bias = update_u_z_with_bias(train_data,
                                                              lr, u, z, user_bias, question_bias, beta, mean)

        if not i % 1000:
            mse_train.append(squared_error_loss_2bias(train_data, u, z, user_bias, question_bias, beta, mean))
            mse_valid.append(squared_error_loss_2bias(val_data, u, z, user_bias, question_bias, beta, mean))
            user_bias_mat = user_bias @ np.ones((1, M))
            question_bias_mat = (question_bias @ np.ones((1, N))).T
            mat = u.dot(z.T) + mean + user_bias_mat + question_bias_mat
            valid_acc.append(sparse_matrix_evaluate(val_data, mat))

    user_bias_mat = user_bias @ np.ones((1, M))
    question_bias_mat = (question_bias @ np.ones((1, N))).T
    mat = u.dot(z.T) + user_bias_mat + question_bias_mat + mean

    figure, ar = plt.subplots(1, 3, figsize=(15, 6))
    ar[0].set_title("MF with 2 bias with regularization on Training")
    ar[0].set_xlabel('Iteration * 1000')
    ar[0].set_ylabel('Squared Error Loss')
    ar[0].plot(mse_train)
    ar[1].set_title("MF with 2 bias with regularization on Validation")
    ar[1].set_xlabel('Iteration * 1000')
    ar[1].set_ylabel('Squared Error Loss')
    ar[1].plot(mse_valid)
    ar[2].set_title("Accuracy on Validation")
    ar[2].set_xlabel('Iteration * 1000')
    ar[2].set_ylabel('Accuracy')
    ar[2].plot(valid_acc)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def als_with_bias_and_gender(train_data, k, lr, num_iteration, beta, gender_mat):
    """ Performs ALS algorithm. Return reconstructed matrix.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :param beta: float
    :param gender_mat: Matrix
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    user_bias = np.zeros((len(set(train_data["user_id"])), 1))

    question_bias = np.zeros((len(set(train_data["question_id"])), 1))

    male_bias = 0
    female_bias = 0
    tran_bias = 0

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    val_data = load_valid_csv("../data")
    mse_train = []
    mse_valid = []
    valid_acc = []
    N, M = 542, 1774
    correct_count = 0
    for i in range(len(train_data["question_id"])):
        if train_data["is_correct"][i] == 1:
            correct_count += 1
    mean = correct_count / len(train_data["question_id"])
    for i in range(num_iteration):
        u, z, user_bias, question_bias, male_bias, female_bias, tran_bias = \
            update_u_z_with_bias_and_gender(train_data, lr, u, z, user_bias,
                                            question_bias, male_bias, female_bias, tran_bias, beta, gender_mat, mean)
        if not i % 2000:
            mse_train.append(squared_error_loss_3bias(train_data, u, z, user_bias, question_bias,
                                                      male_bias, female_bias, tran_bias, beta, gender_mat, mean))
            mse_valid.append(squared_error_loss_3bias(val_data, u, z, user_bias, question_bias,
                                                      male_bias, female_bias, tran_bias, beta, gender_mat, mean))
            gend_mask = np.copy(gender_mat)
            gend_mask[gend_mask == 1] = male_bias
            gend_mask[gend_mask == 2] = female_bias
            gend_mask[gend_mask == 3] = tran_bias
            user_bias_mat = user_bias @ np.ones((1, M))
            question_bias_mat = (question_bias @ np.ones((1, N))).T
            mat = u.dot(z.T) + mean + user_bias_mat + question_bias_mat + gend_mask
            valid_acc.append(sparse_matrix_evaluate(val_data, mat))

    gend_mask = np.copy(gender_mat)
    gend_mask[gend_mask == 1] = male_bias
    gend_mask[gend_mask == 2] = female_bias
    gend_mask[gend_mask == 3] = tran_bias
    user_bias_mat = user_bias @ np.ones((1, M))
    question_bias_mat = (question_bias @ np.ones((1, N))).T
    mat = u.dot(z.T) + mean + user_bias_mat + question_bias_mat + gend_mask

    figure, ar = plt.subplots(1, 3, figsize=(15, 6))
    ar[0].set_title("MF with 3 bias with regularization on Training")
    ar[0].set_xlabel('Iteration * 2000')
    ar[0].set_ylabel('Squared Error Loss')
    ar[0].plot(mse_train)
    ar[1].set_title("MF with 3 bias with regularization on Validation")
    ar[1].set_xlabel('Iteration * 2000')
    ar[1].set_ylabel('Squared Error Loss')
    ar[1].plot(mse_valid)
    ar[2].set_title("Accuracy on Validation")
    ar[2].set_xlabel('Iteration * 2000')
    ar[2].set_ylabel('Accuracy')
    ar[2].plot(valid_acc)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def _load_gender_csv(path):
    # A helper function to load the meta_data.
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    data = {
        "user_id": [],
        "gender": [],
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["user_id"].append(int(row[0]))
                data["gender"].append(int(row[1]))
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data


def load_gender_meta_csv(root_dir="/data"):
    path = os.path.join(root_dir, "student_meta.csv")
    return _load_gender_csv(path)


def get_gender_matrix(gender_meta):
    N, M = 542, 1774
    gender_matrix = np.zeros((N, M))
    for i in range(len(gender_meta["user_id"])):
        uid = gender_meta["user_id"][i]
        gid = gender_meta["gender"][i]
        if gid == 0:
            gender_matrix[uid, :] = 1
        elif gid == 1:
            gender_matrix[uid, :] = 2
        else:
            gender_matrix[uid, :] = 3
    return gender_matrix


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    gender_meta = load_gender_meta_csv("../data")
    gender_matrix = get_gender_matrix(gender_meta)

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    """
    SVD_k_list = [i for i in range(1, 21)]
    max_k = 0
    max_acc = 0
    max_recon_mat = None
    for k in SVD_k_list:
        recon_matrix = svd_reconstruct(train_matrix, k)
        accuracy = sparse_matrix_evaluate(val_data, recon_matrix)
        if accuracy > max_acc:
            max_k = k
            max_acc = accuracy
            max_recon_mat = recon_matrix
        print("Evaluating SVD with k = {}"
              "   Current max accuracy is {} with k = {}"
              .format(k, round(max_acc, 4), max_k))
    acc_test = sparse_matrix_evaluate(test_data, max_recon_mat)
    print("\nMaximum accuracy achieved on validation set = {} "
          " with k = {}.".format(round(max_acc, 4), max_k))
    print("\nAccuracy achieved on test set = {} with our selected k = {}"
          ".".format(round(acc_test, 4), max_k))
    """
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    """
    als_k_list = [5, 10, 20, 40, 70, 100, 130, 160, 190]
    max_acc = 0
    best_k = 0
    best_mat = None
    for k in als_k_list:
        print("Evaluating k = {}, "
              "current max accuracy achieved on k = {} with {}"
              .format(k, best_k, round(max_acc, 4)))
        mat = als(train_data, k, 0.08, 70000)
        val_acc = sparse_matrix_evaluate(val_data, mat)
        if val_acc > max_acc:
            max_acc = val_acc
            best_k = k
            best_mat = mat
    acc_test = sparse_matrix_evaluate(test_data, best_mat)
    print("\nMaximum accuracy achieved on validation set = {} "
          " with k = {}.".format(round(max_acc, 4), best_k))
    print("\nAccuracy achieved on test set = {} with our selected k = {}"
          ".".format(round(acc_test, 4), best_k))
    """
    mat = als_with_bias(train_data, 120, 0.08, 70000, 0.05)
    train_acc = sparse_matrix_evaluate(train_data, mat)
    val_acc = sparse_matrix_evaluate(val_data, mat)
    acc_test = sparse_matrix_evaluate(test_data, mat)
    print("\nMaximum accuracy achieved on train set = {}."
          .format(round(train_acc, 4)))
    print("\nMaximum accuracy achieved on validation set = {}."
          .format(round(val_acc, 4)))
    print("\nAccuracy achieved on test set = {}"
          .format(round(acc_test, 4)))
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
