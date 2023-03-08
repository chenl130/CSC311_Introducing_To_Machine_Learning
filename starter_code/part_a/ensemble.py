# TODO: complete this file.
from part_a.matrix_factorization import *
from part_a.item_response import *
from utils import *
from sklearn.impute import KNNImputer
import numpy as np
from scipy import sparse


def resample(data, m):
    size = len(data["question_id"])
    bags_of_data = []
    num_bags = 0
    while num_bags < m:
        bag = {'user_id': [], 'question_id': [],
               'is_correct': []}
        print("Creating bag number {}". format(num_bags + 1))
        while len(bag["question_id"]) < size:
            index = np.random.choice(size, 1)[0]
            uid = data['user_id'][index]
            qid = data['question_id'][index]
            aid = data['is_correct'][index]
            bag['user_id'].append(uid)
            bag['question_id'].append(qid)
            bag['is_correct'].append(aid)
        bags_of_data.append(bag)
        num_bags += 1
    return bags_of_data


def resampled_data_to_matrix(data):
    mat = np.empty((542, 1774))
    mat[:] = np.NaN
    for i in range(len(data["question_id"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        mat[cur_user_id, cur_question_id] = data["is_correct"][i]
    return mat


def get_pred_matrix_IRT(theta, beta, alpha):
    N = 542
    M = 1774
    theta_mat = theta @ np.ones((1, M))
    beta_mat = (beta @ np.ones((1, N))).T
    alpha_mat = (alpha @ np.ones((1, N))).T
    diff_mat = np.multiply(alpha_mat, theta_mat - beta_mat)
    confidence_mat = sigmoid(diff_mat)
    return confidence_mat


def get_pred_matrix_no_alpha_IRT(theta, beta):
    N = 542
    M = 1774
    theta_mat = theta @ np.ones((1, M))
    beta_mat = (beta @ np.ones((1, N))).T
    diff_mat = theta_mat - beta_mat
    confidence_mat = sigmoid(diff_mat)
    return confidence_mat


def add_data_to_matrix(sparse_mat, data):
    for i in range(len(data["question_id"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        sparse_mat[cur_user_id, cur_question_id] = data['is_correct']
    return sparse_mat


def main():
    train_matrix = load_train_sparse("../data")
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    private_test = load_private_test_csv("C:/Users/16475/Desktop/CSC311/csc311_final_project/starter_code/data")
    gender_meta = load_gender_meta_csv("../data")
    gender_matrix = get_gender_matrix(gender_meta)

    # Specify number of bags to train on each model
    ########################################################
    vanilla_mf = 1
    knn_item = 1
    knn_user = 1
    vanilla_irt = 1
    irt_with_bias = 1
    mf_with_bias = 1
    mf_with_bias_gender = 1
    mf_with_CE_loss = 1

    NUM_OF_BAGS = (knn_item + knn_user + vanilla_irt + irt_with_bias +
                   vanilla_mf + mf_with_bias + mf_with_bias_gender + mf_with_CE_loss)
    ########################################################
    # Generate bootstrapped data
    bags = resample(train_data, NUM_OF_BAGS)
    N = 542
    M = 1774
    sum_conf_mat = np.zeros((N, M))
    # Ensemble of each model specified above
    for i in range(len(bags)):
        if i < NUM_OF_BAGS - irt_with_bias - vanilla_irt:
            if i < vanilla_mf:
                print("Fitting vanilla Matrix Factorization on data set number: {}.".format(i + 1))
                mat = als(bags[i], 120, 0.08, 140000)

            elif i < vanilla_mf + mf_with_bias_gender:
                print("Fitting Matrix Factorization with user bias, question bias, and gender bias "
                      "on data set number: {}.".format(i + 1))
                mat = als_with_bias_and_gender(bags[i], 70, 0.04, 350000, 0.02, gender_matrix)

            elif i < vanilla_mf + mf_with_bias_gender + mf_with_CE_loss:
                print("Fitting Matrix Factorization with cross entropy loss on data set number: {}.".format(i + 1))
                mat = als_logistic(bags[i], 240, 0.08, 280000, 0.02)

            elif i < vanilla_mf + mf_with_bias_gender + mf_with_CE_loss + knn_item:
                print("Fitting Nearest neighbor by item on data set number: {}.".format(i + 1))
                nbrs = KNNImputer(n_neighbors=21)
                new_mat = resampled_data_to_matrix(bags[i])
                mat = nbrs.fit_transform(np.transpose(new_mat)).T
            elif i < vanilla_mf + mf_with_bias_gender + mf_with_CE_loss + knn_item + knn_user:
                print("Fitting Nearest neighbor by user data set number: {}.".format(i + 1))
                nbrs = KNNImputer(n_neighbors=11)
                new_mat = resampled_data_to_matrix(bags[i])
                mat = nbrs.fit_transform(new_mat)
            else:
                mat = als_with_bias(bags[i], 120, 0.08, 230000, 0.001)
         
            sum_conf_mat += mat
        else:
            if i < NUM_OF_BAGS - vanilla_irt:
                print("Fitting Item Response Model with bias on data set number: {}.".format(i + 1))
                num_iteration = 232
                theta = np.ones((N, 1))
                beta = np.ones((M, 1))
                alpha = np.ones((M, 1))
                for _ in range(num_iteration):
                    new_mat = sparse.csc_matrix(resampled_data_to_matrix(bags[i]))
                    theta, beta, alpha = update_theta_beta_alpha(new_mat, 0.001, theta, beta, alpha)
                sum_conf_mat += get_pred_matrix_IRT(theta, beta, alpha)
            else:
                print("Fitting vanilla Item Response Model on data set number: {}.".format(i + 1))
                num_iteration = 76
                theta = np.ones((N, 1))
                beta = np.ones((M, 1))
                for _ in range(num_iteration):
                    new_mat = sparse.csc_matrix(resampled_data_to_matrix(bags[i]))
                    theta, beta = update_theta_beta(new_mat, 0.001, theta, beta)
                sum_conf_mat += get_pred_matrix_no_alpha_IRT(theta, beta)

    avg_mat = sum_conf_mat/NUM_OF_BAGS
    final_acc_valid = sparse_matrix_evaluate(val_data, avg_mat)
    final_acc_test = sparse_matrix_evaluate(test_data, avg_mat)

    print("\nMaximum accuracy of mf achieved on validation set = {}."
          .format(round(final_acc_valid, 4)))
    print("\nAccuracy of mf achieved on test set = {}"
          .format(round(final_acc_test, 4)))

    # Save predictions
    matrix = avg_mat
    threshold = 0.5
    predictions = []
    for i in range(len(private_test["question_id"])):
        cur_user_id = private_test["user_id"][i]
        cur_question_id = private_test["question_id"][i]
        if matrix[cur_user_id, cur_question_id] >= threshold:
            predictions.append(1.)
        else:
            predictions.append(0.)
    private_test["is_correct"] = predictions
    save_private_test_csv(private_test)


if __name__ == '__main__':
    main()
