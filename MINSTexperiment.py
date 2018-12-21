# **********************************************************************
#  Project           : Optimization and Data Analytics Project
#
#  Program name      : MINSTexperiment.py
#
#  Description       : Program to classify the MINST dataset according to
#                      different technics
#
#  Author            : Carlos Hansen
#
#  Studienr          : 201803181
#
# **********************************************************************

import numpy as np

import optimization_and_data_analytics_project as odap  # module with the code


# ---------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------

# MNIST
# Generate the Xtrain, Xtest, train_lbls
print('\n')
MNISTtest = np.loadtxt('MNIST_test_data.txt', delimiter=',')
print('Amount of Vectorized image to test:  ', MNISTtest.shape[1])
MNISTtest_lbls = np.loadtxt('MNIST_test_labels.txt', delimiter=',')
print('.')
MNISTtrain = np.loadtxt('MNIST_train_data.txt', delimiter=',')
print('Amount of Vectorized image to train:  ', MNISTtrain.shape[1])
MNISTtrain_lbls = np.loadtxt('MNIST_train_labels.txt', delimiter=',')
print('Amount of unique train labels: ', np.unique(MNISTtrain_lbls).shape[0])

print("train", MNISTtrain.shape)
print("train_lbls", MNISTtrain_lbls.shape)


# PCA of MNIST
MNISTtrainPCA, MNISTtestPCA = odap.data_PCA2(MNISTtrain, MNISTtest)

# Print PCA
odap.plot_PCA2(MNISTtrainPCA, MNISTtrain_lbls)
# -------------------------------------------
# 1. Nearest Centroid Classifier

# MNIST
print('\n-------------------------------------------')
print("Nearest Centroid Classifier:")

NC_MNISTexp_lbls = odap.nc_classify(MNISTtrain, MNISTtest, MNISTtrain_lbls)
odap.print_errors(MNISTtest, MNISTtest_lbls, NC_MNISTexp_lbls)  # Print errors

# PCA of MNIST
print('\n-------------------------------------------')
print("Nearest Centroid Classifier (PCA data):")

NCpca_MNISTexp_lbls = odap.nc_classify(
    MNISTtrainPCA, MNISTtestPCA, MNISTtrain_lbls)
odap.print_errors(MNISTtestPCA, MNISTtest_lbls,
                  NCpca_MNISTexp_lbls)  # Print errors


# -------------------------------------------
# 2.Nearest subclass centroid Classifier {2,3,5}
print('\n-------------------------------------------')
print("Nearest subclass centroid classifier (2 subclasses):")

NSC2_MNISTexp_lbls = odap.nsc_classify(
    MNISTtrain, 2, MNISTtest, MNISTtrain_lbls)
odap.print_errors(MNISTtest, MNISTtest_lbls,
                  NSC2_MNISTexp_lbls)  # Print errors

print('\n-------------------------------------------')
print("Nearest subclass centroid classifier (PCA data) (2 subclasses):")

NSC2pca_MNISTexp_lbls = odap.nsc_classify(
    MNISTtrainPCA, 2, MNISTtestPCA, MNISTtrain_lbls)
odap.print_errors(MNISTtestPCA, MNISTtest_lbls,
                  NSC2pca_MNISTexp_lbls)  # Print errors

print('\n-------------------------------------------')
print("Nearest subclass centroid classifier (3 subclasses):")
NSC3_MNISTexp_lbls = odap.nsc_classify(
    MNISTtrain, 3, MNISTtest, MNISTtrain_lbls)
odap.print_errors(MNISTtest, MNISTtest_lbls,
                  NSC3_MNISTexp_lbls)  # Print errors

print('\n-------------------------------------------')
print("Nearest subclass centroid classifier (PCA data) (3 subclasses):")
NSC3pca_MNISTexp_lbls = odap.nsc_classify(
    MNISTtrainPCA, 3, MNISTtestPCA, MNISTtrain_lbls)
odap.print_errors(MNISTtestPCA, MNISTtest_lbls,
                  NSC3pca_MNISTexp_lbls)  # Print errors

print('\n-------------------------------------------')
print("Nearest subclass centroid classifier (5 subclasses):")
NSC5_MNISTexp_lbls = odap.nsc_classify(
    MNISTtrain, 5, MNISTtest, MNISTtrain_lbls)
odap.print_errors(MNISTtest, MNISTtest_lbls,
                  NSC5_MNISTexp_lbls)  # Print errors

print('\n-------------------------------------------')
print("Nearest subclass centroid classifier (PCA data) (5 subclasses):")
NSC5pca_MNISTexp_lbls = odap.nsc_classify(
    MNISTtrainPCA, 5, MNISTtestPCA, MNISTtrain_lbls)
odap.print_errors(MNISTtestPCA, MNISTtest_lbls,
                  NSC5pca_MNISTexp_lbls)  # Print errors

# -------------------------------------------
# 3. Nearest neighbour classifier
print('\n-------------------------------------------')
print("Nearest neighbour classifier:")
NNC_MNISTexp_lbls = odap.nn_classify(MNISTtrain, MNISTtest, MNISTtrain_lbls)
odap.print_errors(MNISTtest, MNISTtest_lbls, NNC_MNISTexp_lbls)  # Print errors

print('\n-------------------------------------------')
print("Nearest neighbour classifier (PCA data):")
NNCpca_MNISTexp_lbls = odap.nn_classify(
    MNISTtrainPCA, MNISTtestPCA, MNISTtrain_lbls)
odap.print_errors(MNISTtestPCA, MNISTtest_lbls,
                  NNCpca_MNISTexp_lbls)  # Print errors

# -------------------------------------------
# 4. Perceptron trained using Backpropagation

print('\n-------------------------------------------')
print("Perceptron Clasifier trained using Backpropagation")

eta = 0.1
epoch = 20

pbp = odap.PerceptronBP(eta=eta, epochs=epoch)
pbp.fit(MNISTtrain, MNISTtrain_lbls)



PBB_MNISTexp_lbls = pbp.predict_lbls(MNISTtest)
odap.print_errors(MNISTtest, MNISTtest_lbls, PBB_MNISTexp_lbls)  # Print errors

print('\n-------------------------------------------')
print("Perceptron Clasifier trained using Backpropagation (PCA data)")
pbp_pca = odap.PerceptronBP(eta=eta, epochs=epoch)
pbp_pca.fit(MNISTtrainPCA, MNISTtrain_lbls)

PBBpca_MNISTexp_lbls = pbp_pca.predict_lbls(MNISTtestPCA)
odap.print_errors(MNISTtestPCA, MNISTtest_lbls,
                  PBBpca_MNISTexp_lbls)  # Print errors

# -------------------------------------------
# 5. Nearest neighbour classifier

print('\n-------------------------------------------')
print("Perceptron Clasifier trained using MSE")

pmse = odap.PerceptronMSE()
pmse.fit(MNISTtrain, MNISTtrain_lbls)

PMSE_MNISTexp_lbls = pmse.predict_lbls(MNISTtest)
odap.print_errors(MNISTtest, MNISTtest_lbls,
                  PMSE_MNISTexp_lbls)  # Print errors

print('\n-------------------------------------------')
print("Perceptron Clasifier trained using MSE (PCA data)")

pmse_pca = odap.PerceptronMSE()
pmse_pca.fit(MNISTtrainPCA, MNISTtrain_lbls)

PMSEpca_MNISTexp_lbls = pmse_pca.predict_lbls(MNISTtestPCA)
odap.print_errors(MNISTtestPCA, MNISTtest_lbls,
                  PMSEpca_MNISTexp_lbls)  # Print errors
