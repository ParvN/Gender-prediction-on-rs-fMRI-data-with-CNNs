import os
import numpy as np
import h5py as hp
import scipy
from Code.utils import *


def combine_data(data_dir, sub_list):
    """

    Combines all the subject data in a directory to one large matrix
    input:
    data_dir : Directory containing the data
    sub_list: list of subjects

    """
    data = []
    for i in sub_list:
        C = hp.File(data_dir + str(i) + '.mat', 'r')['CorrMatrix'][()]
        data.append(C)
    data = np.asarray(data)
    return data


def Euclidean_mean(data):
    """
    Calculate the euclidean mean of the all subjects data.
    """

    avg = np.average(data, axis=0)
    return avg


def inverse_squareroot(C_mean):
    """
    Finding the inverse square root of the mean data by diagonalization
    """
    eigvals, eigvects = scipy.linalg.eigh(C_mean)
    inv_sq_eigvals = np.diag(1 / np.sqrt(eigvals))
    diag_matrix = np.dot(np.dot(eigvects, inv_sq_eigvals), eigvects.T)
    return diag_matrix

def make_symmetric(vec):
    """
    make a symmetric matrix from a vector
    vec: input vector
    """
    n = int(np.sqrt(len(vec)*2))+1
    symm = np.zeros((n,n))
    r,c = np.triu_indices(n,k=1)
    symm[r,c] = vec
    symm[c,r] = vec
    return symm


def check_symmetric(M, rtol=1e-05, atol=1e-08):
    """
    Check if a matrix is symmetric.
    input :
    M : Matrix
    returns:
    flag: True if Matrix is symmetric else False
    """
    if not np.allclose(M, M.T, rtol=rtol, atol=atol):
        flag = False
    else:
        flag = True

    return flag


def check_pos_def(M):
    """
    Check if a matrix is positive definite.
    input :
    M : Matrix
    returns:
    flag: True if Matrix is Pos def else False
    least_eig_val : the least eigen value
    """

    eigvals, eigvecs = np.linalg.eigh(M)
    least_eig_val = eigvals.min()
    flag = least_eig_val > 0
    return [flag, least_eig_val]


def tangent_space(data_dir, sub_list, dest_dir):
    """
    Transform the data to tangent space
    """
    data = combine_data(data_dir, sub_list)
    C_mean = Euclidean_mean(data)
    C_inv = inverse_squareroot(C_mean)
    for i in sub_list:
        C = hp.File(data_dir + str(i) + '.mat', 'r')['CorrMatrix'][()]
        val = C_inv @ C @ C_inv
        C_T = np.log(val)
        save_as_numpy_array(dest_dir, str(i), np.abs(C_T))
    print("Converted data to tangent space")