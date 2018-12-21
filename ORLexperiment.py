# **********************************************************************
#  Project           : Optimization and Data Analytics Project
#
#  Program name      : ORLexperiment.py
#
#  Description       : Program to classify the ORL dataset according to
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

# ORL
# Generate the Xtrain, Xtest, train_lbls
orl_data = np.loadtxt('orl_data.txt')
orl_lbls = np.loadtxt('orl_lbls.txt')

# From the complete dataset of 400 vectorized images,  it will take 70% to train and 30% to test
# 70% of 400 images are 280 images
ORLtrain, ORLtrain_lbls, ORLtest, ORLtest_lbls = odap.random_vectors(
    orl_data, orl_lbls, 280)
print('\n')
print('Amount of Vectorized image to train:  ', ORLtrain.shape[1])
print('Amount of Vectorized image to test:  ', ORLtest.shape[1])
print('Amount of unique train labels: ', np.unique(ORLtrain_lbls).shape[0])


# PCA of ORL
ORLtrainPCA, ORLtestPCA = odap.data_PCA2(ORLtrain, ORLtest)

# Print PCA
odap.plot_PCA2(ORLtrainPCA, ORLtrain_lbls)
# -------------------------------------------
# 1. Nearest Centroid Classifier

# ORL
print('\n-------------------------------------------')
print("Nearest Centroid Classifier:")

NC_ORLexp_lbls = odap.nc_classify(ORLtrain, ORLtest, ORLtrain_lbls)
odap.print_errors(ORLtest, ORLtest_lbls, NC_ORLexp_lbls)  # Print errors

# PCA of ORL
print('\n-------------------------------------------')
print("Nearest Centroid Classifier (PCA data):")

NCpca_ORLexp_lbls = odap.nc_classify(ORLtrainPCA, ORLtestPCA, ORLtrain_lbls)
odap.print_errors(ORLtestPCA, ORLtest_lbls, NCpca_ORLexp_lbls)  # Print errors


# -------------------------------------------
# 2.Nearest subclass centroid Classifier {2,3,5}
print('\n-------------------------------------------')
print("Nearest subclass centroid classifier (2 subclasses):")

NSC2_ORLexp_lbls = odap.nsc_classify(ORLtrain, 2, ORLtest, ORLtrain_lbls)
odap.print_errors(ORLtest, ORLtest_lbls, NSC2_ORLexp_lbls)  # Print errors

print('\n-------------------------------------------')
print("Nearest subclass centroid classifier (PCA data) (2 subclasses):")

NSC2pca_ORLexp_lbls = odap.nsc_classify(
    ORLtrainPCA, 2, ORLtestPCA, ORLtrain_lbls)
odap.print_errors(ORLtestPCA, ORLtest_lbls,
                  NSC2pca_ORLexp_lbls)  # Print errors

print('\n-------------------------------------------')
print("Nearest subclass centroid classifier (3 subclasses):")
NSC3_ORLexp_lbls = odap.nsc_classify(ORLtrain, 3, ORLtest, ORLtrain_lbls)
odap.print_errors(ORLtest, ORLtest_lbls, NSC3_ORLexp_lbls)  # Print errors

print('\n-------------------------------------------')
print("Nearest subclass centroid classifier (PCA data) (3 subclasses):")
NSC3pca_ORLexp_lbls = odap.nsc_classify(
    ORLtrainPCA, 3, ORLtestPCA, ORLtrain_lbls)
odap.print_errors(ORLtestPCA, ORLtest_lbls,
                  NSC3pca_ORLexp_lbls)  # Print errors

print('\n-------------------------------------------')
print("Nearest subclass centroid classifier (5 subclasses):")
NSC5_ORLexp_lbls = odap.nsc_classify(ORLtrain, 5, ORLtest, ORLtrain_lbls)
odap.print_errors(ORLtest, ORLtest_lbls, NSC5_ORLexp_lbls)  # Print errors

print('\n-------------------------------------------')
print("Nearest subclass centroid classifier (PCA data) (5 subclasses):")
NSC5pca_ORLexp_lbls = odap.nsc_classify(
    ORLtrainPCA, 5, ORLtestPCA, ORLtrain_lbls)
odap.print_errors(ORLtestPCA, ORLtest_lbls,
                  NSC5pca_ORLexp_lbls)  # Print errors

# -------------------------------------------
# 3. Nearest neighbour classifier
print('\n-------------------------------------------')
print("Nearest neighbour classifier:")
NNC_ORLexp_lbls = odap.nn_classify(ORLtrain, ORLtest, ORLtrain_lbls)
odap.print_errors(ORLtest, ORLtest_lbls, NNC_ORLexp_lbls)  # Print errors

print('\n-------------------------------------------')
print("Nearest neighbour classifier (PCA data):")
NNCpca_ORLexp_lbls = odap.nn_classify(ORLtrainPCA, ORLtestPCA, ORLtrain_lbls)
odap.print_errors(ORLtestPCA, ORLtest_lbls, NNCpca_ORLexp_lbls)  # Print errors

# -------------------------------------------
# 4. Perceptron trained using Backpropagation

print('\n-------------------------------------------')
print("Perceptron Clasifier trained using Backpropagation")

eta = 0.1
epoch = 20

pbp = odap.PerceptronBP(eta=eta, epochs=epoch)
pbp.fit(ORLtrain, ORLtrain_lbls)

PBB_ORLexp_lbls = pbp.predict_lbls(ORLtest)
odap.print_errors(ORLtest, ORLtest_lbls, PBB_ORLexp_lbls)  # Print errors

print('\n-------------------------------------------')
print("Perceptron Clasifier trained using Backpropagation (PCA data)")
pbp_pca = odap.PerceptronBP(eta=eta, epochs=epoch)
pbp_pca.fit(ORLtrainPCA, ORLtrain_lbls)

PBBpca_ORLexp_lbls = pbp_pca.predict_lbls(ORLtestPCA)
odap.print_errors(ORLtestPCA, ORLtest_lbls, PBBpca_ORLexp_lbls)  # Print errors

# -------------------------------------------
# 5. Nearest neighbour classifier

print('\n-------------------------------------------')
print("Perceptron Clasifier trained using MSE")

pmse = odap.PerceptronMSE()
pmse.fit(ORLtrain, ORLtrain_lbls)

PMSE_ORLexp_lbls = pmse.predict_lbls(ORLtest)
odap.print_errors(ORLtest, ORLtest_lbls, PMSE_ORLexp_lbls)  # Print errors

print('\n-------------------------------------------')
print("Perceptron Clasifier trained using MSE (PCA data)")

pmse_pca = odap.PerceptronMSE()
pmse_pca.fit(ORLtrainPCA, ORLtrain_lbls)

PMSEpca_ORLexp_lbls = pmse_pca.predict_lbls(ORLtestPCA)
odap.print_errors(ORLtestPCA, ORLtest_lbls,
                  PMSEpca_ORLexp_lbls)  # Print errors
