import numpy as np
from scipy.linalg import eig, inv


def stochastique(P):
    return np.allclose(np.sum(P, axis=1), 1) and np.all(P >= 0)


def puits(P, i):
    return np.all(P[i, :] == np.eye(len(P))[i])


def stationnaire(P):
    n = P.shape[0]
    A = np.eye(n) - P.T
    A[-1, :] = 1
    b = np.zeros(n)
    b[-1] = 1
    return np.linalg.solve(A, b)


def simulation(P, pi0, t0, tf):
    t = np.arange(t0, tf + 1)

    if P.shape[0] != P.shape[1] or len(pi0) != P.shape[0]:
        raise ValueError("Dimensions incorrectes")
    elif not stochastique(P):
        raise ValueError("La matrice n'est pas stochastique")

    pi = np.zeros((tf + 1, P.shape[0]))
    pi[0, :] = pi0

    for i in range(1, tf + 1):
        pi[i, :] = pi[i - 1, :].dot(P)

    return t, pi


def Diagonalisation(P):
    D, V = eig(P)
    D_matrix = np.diag(D)
    P_reconstruit = V.dot(D_matrix).dot(inv(V))
    return np.allclose(P, P_reconstruit), V, D_matrix


def Exemple(n):
    if n == 1:
        print("shadock")
        P = np.array(
            [[5 / 6, 1 / 12, 1 / 12], [1 / 4, 1 / 2, 1 / 4], [1 / 4, 0, 3 / 4]]
        )
        pi0 = [1, 0, 0]
    elif n == 2:
        print("imprimante")
        P = np.array([[0.2, 0.8, 0], [0.04, 0.95, 0.01], [0.3, 0, 0.7]])
        pi0 = [1, 0, 0]

        #  ajouter d'autres exemples

    return P, pi0
