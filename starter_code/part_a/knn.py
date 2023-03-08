from sklearn.impute import KNNImputer
from utils import *
from matplotlib import pyplot


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(np.transpose(matrix))
    acc = sparse_matrix_evaluate(valid_data, np.transpose(mat))
    print("Validation Accuracy: {}".format(acc))

    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    # code for parts a) and b)
    k_values = {1, 6, 11, 16, 21, 26}

    # dictionary that maps k values to their accuracies
    k_value_accuracies = {}

    for k in k_values:
        k_value_accuracies[k] = knn_impute_by_user(sparse_matrix, val_data, k)

    # plot the results
    pyplot.xlabel("k")
    pyplot.ylabel("Accuracy")
    pyplot.title("Validation Accuracy With Different k Values")
    pyplot.scatter(k_value_accuracies.keys(), k_value_accuracies.values(),
                label="Validation")
    pyplot.legend()
    pyplot.show()

    # report the final test accuracy with k^* = 11
    knn_impute_by_user(sparse_matrix, test_data, 11)

    # ----------------------------------------------------------------
    # code for part c) below

    k_values = {1, 6, 11, 16, 21, 26}

    # dictionary that maps k values to their accuracies
    k_value_accuracies = {}

    for k in k_values:
        k_value_accuracies[k] = knn_impute_by_item(sparse_matrix, val_data, k)

    # plot the results
    pyplot.xlabel("k")
    pyplot.ylabel("Accuracy")
    pyplot.title("Validation Accuracy With Different k Values")
    pyplot.scatter(k_value_accuracies.keys(), k_value_accuracies.values(),
                   label="Validation")
    pyplot.legend()
    pyplot.show()

    # report the final test accuracy with k^* = 21
    knn_impute_by_item(sparse_matrix, test_data, 21)

if __name__ == "__main__":
    main()
