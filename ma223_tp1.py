import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math

def ReductionGauss(Aaug):

    n, m = np.shape(Aaug)

    for i in range (0, n-1):

        for j in range (i+1, n):

            g = Aaug[j, i]

            for l in range (i, m):

                Aaug[j, l] = Aaug[j, l] - (g*Aaug[i, l]/Aaug[i, i])
    return Aaug


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


def ReducGauss_PivotTotal(Aaug):
    A = np.copy(Aaug)
    n, m = np.shape(A)
    for k in range(0, n - 1):
        value_max = 0
        for i in range(k, n):
            for j in range(k, n):
                if (abs(Aaug[i][j]) > value_max):
                    value_max = abs(Aaug[i][j])
                    ligne_value_max = i
                    colonne_value_max = j
        for j in range(k, n):
            value = A[k][j]
            A[k][j] = A[ligne_value_max][j]
            A[ligne_value_max][j] = value

        for i in range(k, n):
            value = A[i][k]
            A[i][k] = A[i][colonne_value_max]
            A[i][colonne_value_max] = value
        pivot = A[k, k]
        if (pivot == 0):
            print("Le pivot est nul")

        elif (pivot != 0):

            for i in range(k + 1, n):
                A[i, :] = A[i, :] - (A[i, k] / pivot) * A[k, :]

    return (A)


def Gauss_PivotTotal(A, B):
    stack = np.column_stack([A, B])
    reduc = ReducGauss_PivotTotal(stack)
    solution = ResolutionSystTriSup(reduc)
    return (solution)


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


def Gauss(A, B):

    Aaug = np.concatenate((A, B), axis=1)
    print("matrice augmentée: ", Aaug)

    Taug = ReductionGauss(Aaug)
    print("Triangulaire augmentée: ", Taug)

    X = ResolutionSystTriSup(Taug)
    print("solution: ", X,"\n----------------\nfin de procédure\n----------------\n")

    return X


def random_list():

    L = list()
    
    n = random.randint(0, 999)
    
    for i in range(0, n):
        a = random.randint(-99, 99)
        L.append(a)

    return L

# -------- Pivot partiel
def PivotPartiel (A, k, n):
    value = k
    for p in range (k, n):
        if abs(A[p][k]) > abs(A[value][k]):
            value = p
    return(value)

def Transposition(A, i, p):

    copy = A[i].copy()
    A[i] = A[p]
    A[p] = copy
    return(A)

def ReducGauss_PivotPartiel(Aaug):

    A = np.copy(Aaug)
    n, m = np.shape(A)
    for k in range (0, n - 1):
        p = PivotPartiel(A, k, n)
        pivot = A[p, k]
        A = Transposition(A, k, p)
        if pivot == 0:
            print("Le pivot est nul")
        else :
            for i in range (k + 1, n):
                G = A[i, k] / pivot
                A[i, :] = A[i, :] - G * A[k, :]
    return A

def Gauss_PivotPartiel(A, B):
    stack = np.column_stack([A, B])
    reduc = ReducGauss_PivotPartiel(stack)
    solution = ResolutionSystTriSup(reduc)
    return(solution)


def precision(A, X, B):

    n = len(B)
    B = np.reshape(B,(1,n))
    X = np.ravel(X)
    print("précision", X)
    B = np.ravel(B)
    print("précision", B)
    a = np.dot(A, X) - B
    return np.linalg.norm(a)


def etude_prec_global():

    """Cette fonction trace l'ensemble des courbes
    de temps étudiées sur un seul et même graphique (comparaison)"""

    nb_matrice = list()
    end = 1100
    pas = 100

    l_p1 = list()

    for i in range(2, end, pas):

        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        print("A = ", A)
        print("B = ", B)
        X = Gauss(A, B)
        P = precision(A, X, B)
        print(X)
        print("---------------------\nETAPE:" + str(i) + "," + str(100 * i / end) + "\n---------------------")
        l_p1.append(P)
        nb_matrice.append(i)

    l_p2 = list()

    for i in range(2, end, pas):

        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        print("A = ", A)
        print("B = ", B)
        L, U = DecompositionLU(A)
        X = ResolutionLU(L, U, B)
        P = precision(A, X, B)
        print(X)
        print("---------------------\nETAPE:" + str(i) + "," + str(100 * i / end) + "\n---------------------")
        l_p2.append(P)

    l_p3 = list()

    for i in range(2, end, pas):

        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        print("A = ", A)
        print("B = ", B)
        X = np.linalg.solve(A, B)
        P = precision(A, X, B)
        print(X)
        print("---------------------\nETAPE:" + str(i) + "," + str(100 * i / end) + "\n---------------------")
        l_p3.append(P)

    l_p4 = list()

    for i in range(2, end, pas):

        A = np.random.rand(i, i)
        B = np.random.rand(i, 1)

        print("A = ", A)
        print("B = ", B)
        X = Gauss_PivotPartiel(A, B)
        P = precision(A, X, B)
        print(X)
        print("---------------------\nETAPE:" + str(i) + "," + str(100 * i / end) + "\n---------------------")
        l_p4.append(P)


    p1 = plt.plot(nb_matrice, l_p1, color="blue", label="Gauss")
    p2 = plt.plot(nb_matrice, l_p2, color="red", label="LU")
    p3 = plt.plot(nb_matrice, l_p3, color="green", label="Python")
    p4 = plt.plot(nb_matrice, l_p4, color="yellow", label="Partiel")


    plt.xlabel("taille des matrices - pas de " + str(pas))
    plt.ylabel("|Ax - b|")
    plt.title("Comparaison des méthodes étudiées - précision")
    plt.legend([p1, p2, p3, p4], ["Algo Gauss", "LU", "Python", "Partiel"])

    plt.savefig("graph précision - comparaison" + str(end))
    plt.show()

etude_prec_global()