import numpy as np
from numpy.linalg import eigvalsh
from scipy.linalg import schur, eigh, eig, hessenberg,eigvals
from scipy.optimize import linear_sum_assignment


def randdiag(U):
    H = (U+U.conj().T) / 2; S = (U-U.conj().T) / 2
    mu = np.random.normal(0,1,2)
    A_mu = mu[0] * H + mu[1] * 1j*S #+ mu[2]*1j * H + mu[3]* 1j * 1j * S
    _, Q = eigh(A_mu,overwrite_a=True,overwrite_b = True,check_finite=False)
    return Q

def eigenvalue_unitary_angle(U):
    H = (U+U.conj().T) / 2; S = (U-U.conj().T) / 2
    mu = np.random.normal(0,1,2)
    A_mu = mu[0] * H + mu[1] * 1j*S
    D1 = eigvalsh(A_mu)
    D2 = eigvalsh(H)
    D2 = np.arccos(np.clip(D2,-1,1))
    D2 = np.concatenate([D2,-D2])

    angle = np.angle( mu[0]-1j*mu[1]); radius = np.absolute(mu[0]-1j*mu[1])
    D1 = np.arccos(D1 / radius)
    D1_plus = angle+D1; D1_plus = D1_plus + (D1_plus > np.pi) * (- 2*np.pi) 
    D1_minus = angle-D1; D1_minus = D1_minus  + (D1_minus < -np.pi) * (2*np.pi)
    condition = np.array([ True if np.min(np.abs(D1_plus[x] - D2)) < np.min(np.abs(D1_minus[x] - D2)) \
                      else False for x in range(D1.size)],dtype=bool)
    D1 = np.where(condition,D1_plus,D1_minus)
    return D1

def random_normal_matrix(n:int):
    A = np.random.randn(n,n) + 1j*np.random.randn(n,n)
    U,_ = np.linalg.qr(A)
    d = (np.random.randn(n) / np.sqrt(2) ) + (1j*np.random.randn(n) / np.sqrt(2) )
    return U @ np.diag(d) @ U.conj().T, d

def compare_eigenvalues(d1,d2):
    n = d1.shape[0]
    distances = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            distances[i,j] = np.linalg.norm(d1[i]- d2[j]) ** 2
    row_ind, col_ind = linear_sum_assignment(distances)
    return col_ind, np.sqrt(distances[row_ind,col_ind].sum()) / np.linalg.norm(d1)

def report_stats(error_array):
    return np.mean(error_array), np.std(error_array), np.min(error_array), np.max(error_array)