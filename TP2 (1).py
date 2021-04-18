import numpy as np
import matplotlib.pyplot as plt
import time

def ResolutionLLT(L, B):
    # On décompose A*X = B donc L*L.T*X = B , on pose L.T*X = Y
    # Ainsi on obtient L*Y = B

    Y = ResolutionSysTriInf(L, B)

    # print(L)
    # Ainsi on cherche donc L.T*X = Y

    Aaug = np.column_stack([L.T, Y])
    X = ResolutionSystTriSup(Aaug)

    # print(X)
    return (X)

A = np.array([[4,2,0, 0],[2,4,2, 0], [0, 2, 4, 2], [0, 0, 2, 4]])

L = np.linalg.cholesky(A)

B = np.array([[1], [2], [3], [4]])


def Gauss(A, B):

    Aaug = np.concatenate((A, B), axis=1)
    print("matrice augmentée: ", Aaug)

    Taug = ReductionGauss(Aaug)
    print("Triangulaire augmentée: ", Taug)

    X = ResolutionSystTriSup(Taug)
    print("solution: ", X,"\n----------------\nfin de procédure\n----------------\n")

    return X

def ReductionGauss(Aaug):

    n, m = np.shape(Aaug)

    for i in range (0, n - 1):

        for j in range (i + 1, n):

            g = Aaug[j, i]

            for l in range (i, m):

                Aaug[j, l] = Aaug[j, l] - (g*Aaug[i, l]/Aaug[i, i])
    return Aaug

def ResolutionSysTriInf(L, B):

    Taug = np.column_stack([L, B])
    n = len(Taug)
    Y = np.zeros((n, 1))
    for i in range(n):
        Y[i] = B[i]
        for j in range(0, i):
            Y[i] -= Y[j]*Taug[i, j]
        Y[i] = Y[i]/float(Taug[i, i])

    return Y


def ResolutionSystTriSup(Taug):

    n, m = np.shape(Taug)
    X = np.zeros((n,1))
    X[n-1, 0] = Taug[n-1, n]/Taug[n-1, n-1]

    for i in range(n-2, -1, -1):

        X[i] = Taug[i, n]

        for j in range(i+1, n):

            X[i] = X[i]-Taug[i, j]*X[j]

        X[i] = X[i]/Taug[i, i]

    return X


def ResolutionLU(L, U, B):

    Y = []
    n, m = B.shape
    for i in range(n):
        Y.append(B[i])
        for j in range(i):
            Y[i] = Y[i] - (L[i, j] * Y[j])
        Y[i] = Y[i] / L[i, i]
    X = np.zeros(n)
    for i in range(n, 0, - 1):
        X[i - 1] = (Y[i - 1] - np.dot(U[i - 1, i:], X[i:])) / U[i - 1, i - 1]
    return X


def DecompositionLU(A):

    U = np.array(A)
    L = np.zeros((np.size(A[0]),np.size(A[0])))
    np.fill_diagonal(L, 1)
    for j in range(len(U[0])):
        for i in range(len(U)):
            if j < i and U[i, j] != 0:
                pivot = U[j, j]
                g = U[i, j] / pivot
                for k in range(len(U[i])):
                    U[i, k] = U[i, k] - g * U[j, k]
                L[i, j] = g
    return L, U


def cholesky_decomposition(A):

    n = len(A)
    L = np.zeros((n,n))

    for i in range(n):
        somme1 = 0.
        for k in range(i):
            somme1 += pow(L[i, k], 2)
        L[i, i] = pow(A[i, i]-somme1, 1/2)
        for j in range(i + 1, n):
            somme2 = 0.
            for l in range(j):
                somme2 += L[j, l]*L[i, l]
            L[j, i] = (A[i, j] - somme2)/L[i, i]

    return L

def precision(A, X, B):

    n = len(B)
    B = np.reshape(B, (1, n))
    X = np.ravel(X)
    print("précision", X)
    B = np.ravel(B)
    print("précision", B)
    a = np.dot(A, X) - B
    return np.linalg.norm(a)


def etude_prec_global2():

    """Cette fonction trace l'ensemble des courbes
    de précision étudiées sur un seul et même graphique (comparaison)"""

    nb_matrice = list()
    end = 1000
    pas = 100

    l_p1 = list()
    l_p2 = list()
    l_p3 = list()
    l_p4 = list()
    l_p5 = list()

    for i in range(3, end + pas, pas):

        nb_matrice.append(i)

        A = np.random.rand(i, i)
        A = A@A.T
        B = np.random.rand(i, 1)

        print("A = ", A)
        print("B = ", B)

        X = Gauss(A, B)
        P = precision(A, X, B)
        print(X)
        print("---------------------\nETAPE:" + str(i) + "," + str(100 * i / end) + "\n---------------------")
        l_p1.append(P)

        L, U = DecompositionLU(A)
        X = ResolutionLU(L, U, B)
        P = precision(A, X, B)
        print(X)
        l_p2.append(P)

        X = np.linalg.solve(A, B)
        P = precision(A, X, B)
        print(X)
        l_p3.append(P)

        L = cholesky_decomposition(A)
        X = ResolutionLLT(L, B)
        P = precision(A, X, B)
        print(X)
        l_p4.append(P)

        L = np.linalg.cholesky(A)
        X = ResolutionLLT(L, B)
        P = precision(A, X, B)
        print(X)
        l_p5.append(P)

    print("-----------------\nfin de procédure\n-----------------\n")


    p1 = plt.plot(nb_matrice, l_p1, color="blue", label="Gauss")
    p2 = plt.plot(nb_matrice, l_p2, color="red", label="LU")
    p3 = plt.plot(nb_matrice, l_p3, color="green", label="Python")
    p4 = plt.plot(nb_matrice, l_p4, color="yellow", label="cholesky")
    p5 = plt.plot(nb_matrice, l_p5, color="black", label="linalgcho")


    plt.xlabel("taille des matrices - pas de " + str(pas))
    plt.ylabel("|Ax - b|")
    plt.title("Comparaison des méthodes étudiées - précision")
    plt.legend([p1, p2, p3, p4, p5], ["Algo Gauss", "LU", "Python", "cholesky", "linalgcho"])

    plt.savefig("graph précision - comparaison" + str(end))
    plt.show()



etude_prec_global2()