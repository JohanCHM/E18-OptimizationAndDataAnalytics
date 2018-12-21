# **********************************************************************
#  Project           : Optimization and Data Analytics Project
#
#  Program name      : optimization_and_data_analytics_project.py
#
#  Description       : Main module with the implementation of the different
#                       algorithms from the project.
#
#  Author            : Carlos Hansen
#
#  Studienr          : 201803181
#
# **********************************************************************

# For matrix manipulation:
import numpy as np

# For PCA generation:
from sklearn.decomposition import PCA

# For ploting
import matplotlib.pyplot as plt
# ---------------------------------------------------------------------------------------------------
# Processing data
# ---------------------------------------------------------------------------------------------------


def random_vectors(vectors, labels, samples):
    """
    Return a splited vectors matrix and its corresponding labels in two, random orgnized matrix each.
    Samples are the number of vectors in the first splited matrix and it correspondant labels.
    """
    # check if the samples are not bigger
    if vectors.shape[1] < samples:
        print('Vector too small to take such a sample')

    # concatenate vectors and labels
    tmp_vtrs = np.concatenate((vectors, [labels]), axis=0)

    # mix it
    tmp_vtrs = np.random.permutation(tmp_vtrs.T).T
    # print(tmp_vtrs)

    # cut the array
    train_vtrs = tmp_vtrs[:-1, 0:samples]
    train_lbls = tmp_vtrs[-1:, 0:samples]
    test_vtrs = tmp_vtrs[:-1, samples:]
    test_lbls = tmp_vtrs[-1:, samples:]

    return train_vtrs, train_lbls[0], test_vtrs, test_lbls[0]


def data_PCA2(Xtrain, Xtest):
    """Return the vectors transform to a PCA with 2 Principal Components"""
    # Scaled data
    # scaled_vctrs = preprocessing.scale(vectors.T)

    # Define and adjust the PCA object
    pca = PCA(n_components=10)
    pca.fit(Xtrain.T)
    pca_data = pca.transform(Xtrain.T)

    # Scree plot of the principal components
    # percentage of variation each PCA
    per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
    labels = ['PC'+str(x) for x in range(1, len(per_var)+1)]

    # Bar plot of each principal component participation
    plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Percentage of Variation each Principal Component accounts for')
    plt.show()

    pca = PCA(n_components=2)
    pca.fit(Xtrain.T)

    return np.transpose(pca.transform(Xtrain.T)), np.transpose(pca.transform(Xtest.T))


def plot_PCA2(pca_data, pca_lbls):
    uq, clusters = lbls_clusters(pca_data, pca_lbls)

    x = range(100)
    y = range(100, 200)

    # ax1.scatter(x[:4], y[:4], s=10, c='b', marker="s", label='first')
    # ax1.scatter(x[40:],y[40:], s=10, c='r', marker="o", label='second')
    # plt.legend(loc='upper left');
    # plt.show()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # for each cluster
    for k in range(0, len(clusters)):
        pc1 = clusters[k][0, :]
        pc2 = clusters[k][1, :]
        marker = "o"
        color = 'C' + str(k % 9)
        label = 'lbl' + str(k)
        ax1.scatter(x=pc1,
                    y=pc2,
                    color=color,
                    marker=marker,
                    label=label)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

    # label of the axes

    plt.show()


# ---------------------------------------------------------------------------------------------------
# Basic measures
# ---------------------------------------------------------------------------------------------------


def euc_distance_sq(vector1, vector2):
    """return the square eucledean distance between vector1 and vector2"""
    # check if they have a valid dimesion
    if vector1.shape[0] != vector2.shape[0]:
        print("The vectors do not have a valid  dimension")
        return

    # Getting the euclidean distance
    distance = 0
    for d in range(0, vector1.shape[0]):
        dif = vector1[d]-vector2[d]
        distance += dif*dif
    return distance


def cluster_mean(cluster):
    """Return the mean of the cluster"""
    # print(cluster.shape)
    return(1/cluster.shape[1])*np.sum(cluster, axis=1)

# ---------------------------------------------------------------------------------------------------
# Data manipultion
# ---------------------------------------------------------------------------------------------------


def initialize_clusters_w_means(means):
    """Returns a set of clusters with each mean in means as their initial value."""
    clusters = []

    for m in range(0, means.shape[1]):
        clusters.append(means[:, [m]])

    return clusters


def closest_vector_to_vector(vectorTest, vectors):
    """Return the arg to the closest vector between a vectorTest an matrix of vectors"""
    # check if they have the same dimesion
    if vectorTest.shape[0] != vectors.shape[0]:
        print("The vectors and the matriz do not have a compatible dimension")
        return

    # Initialize the a matrix to initialize the distance of the vector to each mean
    vectors_distance = np.zeros(vectors.shape[1])

    # Fill the means_distance matrix
    for m in range(0, vectors.shape[1]):
        vectors_distance[m] = euc_distance_sq(vectorTest, vectors[:, [m]])

    # print('*') # To display while waiting
    # return the argument of the least distance = number corresponding to the cluster
    return np.argmin(vectors_distance)


def closest_mean_to_vector(vector, means):
    """Return the closest mean vector between a vector an matrix of means"""
    return means[:, [closest_vector_to_vector(vector, means)]]


def closest_mean_to_vectors(vectors, means):
    """Return the a  matrix made of the closest mean vector between a each vector an matrix of means"""
    # create a matrix with the same number of columns as the number of vectors
    clo_mns = np.zeros((vectors.shape[0], vectors.shape[1]))
    # get to which mean is closest each vector
    for n in range(0, vectors.shape[1]):

        clo_mns[:, n] = closest_mean_to_vector(vectors[:, [n]], means)[:, 0]
        # print('-') # To display while waiting
    return clo_mns


def lbls_to_address(lbls):
    """return a set of ordered unique lbls and a set of addresses corresponding to the lbls mappedd into the unique vectors """
    address = np.zeros(lbls.shape[0])
    uniques = np.unique(lbls)
    for l in range(0, lbls.shape[0]):
        for u in range(0, uniques.shape[0]):
            if uniques[u] == lbls[l]:
                address[l] = u
                break
    return uniques, address


def lbls_clusters(vectors, lbls):
    """Return a matrix of uniques lbls and a set with the Separated vectors into matrices of clusters according to the labels and in the same order as the unique lbls"""
    uniques, address = lbls_to_address(lbls)
    # Initialize clusters
    clusters = []
    for c in range(0, uniques.shape[0]):
        clusters.append(np.zeros((vectors.shape[0], 1)))

    # Fill the clusters
    for n in range(0, vectors.shape[1]):
        clusters[int(address[n])] = np.concatenate(
            (clusters[int(address[n])], vectors[:, [n]]), axis=1)

    # Remove initizalization zero from clusters
    for c in range(0, len(clusters)):
        clusters[c] = np.delete(clusters[c], 0, 1)
    return uniques, clusters


def means_clusters(vectors, means):
    """Separate the vectors into matrices of clusters sccording to the means"""
    # Initialize clusters with the means as initial values
    klusters = initialize_clusters_w_means(means)

    closest_means = closest_mean_to_vectors(vectors, means)

    # Fill the clusters
    for c in range(0, vectors.shape[1]):
        for k in range(0, len(klusters)):
            if np.array_equal(means[:, [k]], closest_means[:, [c]]):
                klusters[k] = np.concatenate(
                    (klusters[k], vectors[:, [c]]), axis=1)
                break

    # remove means from clusters
    for c in range(0, len(klusters)):
        klusters[c] = np.delete(klusters[c], 0, 1)
    return klusters


def clusters_to_means(clusters):
    """Returns a matrix of means according to an array of clusters"""
    # Initialize the means matrix
    mn = np.zeros((clusters[0].shape[0], 1))
    for k in clusters:
        mn = np.concatenate((mn, cluster_mean(k).reshape(-1, 1)), axis=1)
    # Remove initizalization zero
    mn = np.delete(mn, 0, 1)
    return mn


def k_means(cluster, k):
    # Assign the first means as the firrst k values in the cluster
    mns = cluster[:, 0:k]

    i = 0
    while True:

        subc = means_clusters(cluster, mns)
        nw_mns = clusters_to_means(subc)
        # print(i, end='')

        diff = np.array_equal(mns, nw_mns)
        # print(".", end='')

        mns = nw_mns

        i += 1
        if i > 1000000 or diff:
            break
    subc = means_clusters(cluster, mns)
    # print(".")

    return mns, subc

# ---------------------------------------------------------------------------------------------------
# Error Analysis
# ---------------------------------------------------------------------------------------------------


def amount_different_lbls(tst_lbls, known_lbls):
    """Count the ammoun of difference between the tst_lbls and the known"""
    diff = tst_lbls - known_lbls
    return np.count_nonzero(diff)


def percentage_errrors(tst_data, tst_lbls, known_lbls):

    num_errors = amount_different_lbls(tst_lbls, known_lbls)
    return round((num_errors / tst_data.shape[1])*100, 4)


def print_errors(Xtest, Xtest_lbls, Xresults_lbls):
    num_errors = amount_different_lbls(Xtest_lbls, Xresults_lbls)
    per_errors = percentage_errrors(Xtest, Xtest_lbls, Xresults_lbls)

    print("No errors: ", num_errors)
    print("Percentage of errors: ", per_errors, "%")

# ---------------------------------------------------------------------------------------------------
# Clasifiers
# ---------------------------------------------------------------------------------------------------

# Nearest Centroid Classifier


def nc_classify(Xtrain, Xtest, train_lbls):
    """Return the labels of a test data according to the Nearest Centroid Classifier"""

    # Obtain the different clusters according to the labels
    unique_lbls, klusters = lbls_clusters(Xtrain, train_lbls)
    # print('k', klusters[0])
    # print('u', unique_lbls)

    # Initialize the means matrix
    mn = np.zeros((Xtrain.shape[0], 1))
    for k in klusters:
        mn = np.concatenate((mn, cluster_mean(k).reshape(-1, 1)), axis=1)

    # Remove initizalization zero
    mn = np.delete(mn, 0, 1)

    # Obtain the closest mean for each test value
    clos_mean = closest_mean_to_vectors(Xtest, mn)

    # Initialize the test_lbls
    test_lbls = np.zeros([Xtest.shape[1]])

    # Map the closest mean to each label
    for i in range(0, clos_mean.shape[1]):
        for m in range(0, mn.shape[1]):
            if np.array_equal(clos_mean[:, [i]], mn[:, [m]]):
                test_lbls[i] = unique_lbls[m]
                break

    return test_lbls


# Nearest subclass centroid Classifier {2,3,5}

def nsc_classify(Xtrain, K, Xtest, train_lbls):
    """Return the labels of a test data according to the Nearest Subclass Centroid Classifier"""
    # Obtain the different clusters according to the labels
    unique_lbls, clusters = lbls_clusters(Xtrain, train_lbls)

    # Obtain the means arrays and the means matrix
    means_clusters = []
    mn = np.zeros((Xtrain.shape[0], 1))
    for clus in clusters:
        k_mns = k_means(clus, K)[0]
        means_clusters.append(k_mns)
        mn = np.concatenate((mn, k_mns), axis=1)
    # Remove initizalization zero
    mn = np.delete(mn, 0, 1)

    # Obtain the closest mean for each test value
    clos_mean = closest_mean_to_vectors(Xtest, mn)

    # Initialize the test_lbls
    test_lbls = np.zeros([Xtest.shape[1]])

    # Map the closest mean to each label
    for i in range(0, clos_mean.shape[1]):
        for m in range(0, mn.shape[1]):
            if np.array_equal(clos_mean[:, [i]], mn[:, [m]]):
                test_lbls[i] = unique_lbls[int(m/K)]
                break

    return test_lbls


def nn_classify(Xtrain, Xtest, train_lbls):
    """Return the labels of a test data according to the Nearest Neighbour Classifier"""
    # Initialize the test_lbls
    test_lbls = np.zeros([Xtest.shape[1]])

    # For each test vector
    for n in range(0, Xtest.shape[1]):
        # Assign the label of the closest vector
        test_lbls[n] = train_lbls[closest_vector_to_vector(
            Xtest[:, [n]], Xtrain)]

    return test_lbls

# ---------------------------------------------------------------------------------------------------
# Perceptron CLasifiers
# ---------------------------------------------------------------------------------------------------


class PerceptronBP (object):
    """
    Perceptron Clasifier trained used Backpropagation

    Parameters
    -----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    epochs : int
        Number of passes over the training set

    Attributes
    -----------
    w_ : array
        Weights after fitting.
    errors_ : list
        Number of misclassification in every epoch.
    unq_ : vector
        Unique lbls for the training set
    """

    def __init__(self, eta=0.01, epochs=10):
        self.eta = eta
        self.epochs = epochs

    def fit(self, Xtrain, train_lbls):
        """
        Fit method to training data

        Parameters
        -----------
        Xtrain : {array-like}, shape = [n_features, n_samples]
            Training vectors, where `n_samples` is the number
            of samples and `n_features` is the number of features.
        train_lbls : {array-like}, shape = [n_samples]
            Target values.

        Returns
        --------
        self : object

        """
        self.unq_ = np.unique(train_lbls)
        self.w_ = np.zeros(
            (1 + Xtrain.shape[0], self.unq_.shape[0]))
        self.errors_ = []

        b = self.map_lbls_to_b(train_lbls)

        for u in range(0, self.unq_.shape[0]):
            for ep in range(self.epochs):
                errors = 0  # number of errors in each epoch
                workingX = np.vstack((Xtrain, b[u]))

                for v in range(0, workingX.shape[1]):
                    target = workingX[-1, v]
                    vi = workingX[:-1, v]
                    update = self.eta * (target - self.predict_one(vi, u))
                    self.w_[1:, u] += update * vi
                    self.w_[0, u] += update
                    errors += int(update != 0.0)

                self.errors_.append(errors)
                # print("Vec", u, "Ep:", ep, " errors: ", errors)

        return self

    def net_input(self, X, unq_adrs):
        """
        Calculate the net input.

        Parameters
        -----------
        X : {array-like}, shape = [n_features, n_samples]
            Vectors, where `n_samples` is the number
            of samples and `n_features` is the number of features.
        unq_adrs : int
            Address of the lbl related to the specific Weight
        """
        return (np.dot(X, self.w_[1:, unq_adrs])) + self.w_[0, unq_adrs]

    def predict_one(self, X, unq_adrs):
        """Return class binary label for the specific vector after unit step"""
        return np.where(self.net_input(X, unq_adrs) >= 0.0, 1, 0)

    def predict(self, X):
        """Return class binary labels"""

        bin_lbls = np.zeros((self.w_.shape[1], X.shape[1]))

        for u in range(0, self.w_.shape[1]):
            for v in range(0, X.shape[1]):
                bin_lbls[u, v] = self.predict_one(X[:, v], u)

        return bin_lbls

    def map_lbls_to_b(self, train_lbls):
        """
        Method to map to a binary lbls 1 if the category is correct, 0 if not.
        one row for each train_lbls

        Parameters
        -----------
        train_lbls : {1d-array}, shape = [n_samples]
            Target values.

        Returns
        --------
        b : {array}, shape = [unique_lbls, labeled_vectors]
            Binary lbls 1 if has it,0 if not

        """

        unique_lbls = self.unq_

        b = np.zeros((unique_lbls.shape[0], train_lbls.shape[0]))

        for l in range(0, train_lbls.shape[0]):
            for u in range(0, unique_lbls.shape[0]):
                if train_lbls[l] == unique_lbls[u]:
                    b[u, l] = 1
                    break

        return b

    def map_b_to_lbls(self, b):
        """
        Convert bin lbls to regular lbls
        """
        lbls = np.zeros(b.shape[1])

        for v in range(0, b.shape[1]):
            la = np.argmax(b[:, v])
            lbls[v] = self.unq_[la]
            # print("V:", v, "Lbls:", self.unq_[la])

        return lbls

    def predict_lbls(self, X):
        """Return the labels for a set of data"""
        return self.map_b_to_lbls(self.predict(X))

# Perceptron trained using MSE


class PerceptronMSE (object):
    """
    Perceptron Clasifier trained using MSE

    Attributes
    -----------
    w_ : array
        Weights after fitting.
    errors_ : list
        Number of misclassification in every epoch.
    unq_ : vector
        Unique lbls for the training set
    """

    # def __init__(self):
    #     print('.')

    def fit(self, Xtrain, train_lbls):
        """
        Fit method to training data

        Parameters
        -----------
        Xtrain : {array-like}, shape = [n_features, n_samples]
            Training vectors, where `n_samples` is the number
            of samples and `n_features` is the number of features.
        train_lbls : {array-like}, shape = [n_samples]
            Target values.

        Returns
        --------
        self : object

        """
        self.unq_ = np.unique(train_lbls)

        b = self.map_lbls_to_b(train_lbls)

        # w = inv(X * X.T) * X * b
        self.w_ = np.linalg.inv(Xtrain.dot(Xtrain.T)).dot(Xtrain).dot(b.T)

        print(self.w_.shape)

        return self

    def net_input(self, X, unq_adrs):
        """
        Calculate the net input.

        Parameters
        -----------
        X : {array-like}, shape = [n_features, n_samples]
            Vectors, where `n_samples` is the number
            of samples and `n_features` is the number of features.
        unq_adrs : int
            Address of the lbl related to the specific Weight
        """
        return (np.dot(X, self.w_[:, unq_adrs]))

    def predict_one(self, X, unq_adrs):
        """Return class binary label for the specific vector after unit step"""
        return np.where(self.net_input(X, unq_adrs) >= 0.0, 1, 0)

    def predict(self, X):
        """Return class binary labels"""

        bin_lbls = np.zeros((self.w_.shape[1], X.shape[1]))

        for u in range(0, self.w_.shape[1]):
            for v in range(0, X.shape[1]):
                bin_lbls[u, v] = self.predict_one(X[:, v], u)

        return bin_lbls

    def map_lbls_to_b(self, train_lbls):
        """
        Method to map to a binary lbls 1 if the category is correct, 0 if not.
        one row for each train_lbls

        Parameters
        -----------
        train_lbls : {1d-array}, shape = [n_samples]
            Target values.

        Returns
        --------
        b : {array}, shape = [unique_lbls, labeled_vectors]
            Binary lbls 1 if has it,0 if not

        """

        unique_lbls = self.unq_

        b = np.zeros((unique_lbls.shape[0], train_lbls.shape[0]))

        for l in range(0, train_lbls.shape[0]):
            for u in range(0, unique_lbls.shape[0]):
                if train_lbls[l] == unique_lbls[u]:
                    b[u, l] = 1
                    break

        return b

    def map_b_to_lbls(self, b):
        """
        Convert bin lbls to regular lbls
        """
        lbls = np.zeros(b.shape[1])

        for v in range(0, b.shape[1]):
            la = np.argmax(b[:, v])
            lbls[v] = self.unq_[la]
            # print("V:", v, "Lbls:", self.unq_[la])

        return lbls

    def predict_lbls(self, X):
        """Return the labels for a set of data"""
        return self.map_b_to_lbls(self.predict(X))
