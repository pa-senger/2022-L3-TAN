import numpy as np
import networkx as nx
import scipy.linalg as sc

TOL = 1e-8


def CSR(M):
    """CSR storage for sparses matrices.

    param M : ndarray, matrix to store.

    return (V,C,R): 3-tuple of ndarray,
            V = nnz coeffs,
            C = column indexes of M,
            R = indexes of the first nnz in V for each row and R[-1] = nb of nnz+1.
    """
    n, m = np.shape(M)
    nb_nnz = np.count_nonzero(M)  # we need the number of nnz to init ndarrays

    if nb_nnz == 0:
        print("The matrix is the zero matrix, there is nothing to do")
        return None

    V = np.zeros(nb_nnz, dtype=float)  # 1D ndarrays
    C = np.zeros(nb_nnz, dtype=int)
    R = np.zeros(n + 1, dtype=int)
    R[-1] = nb_nnz + 1

    v_idx = 0  # ndarray cant be appended, so we keep track of indexes to use
    r_idx = 0

    for i in range(n):
        first_nnz = False  # we keep track of the first nnz per row

        for j in range(m):
            if M[i, j] != 0:
                V[v_idx] = M[i, j]  # array of nnz
                C[v_idx] = j  # array of nnz col indexes
                v_idx += 1  # found 1 nnz so we ++idx

                if first_nnz == False:
                    R[r_idx] = v_idx - 1  # index in V of the first nnz of the row
                    first_nnz = True  # we found the first nnz for this row
                    r_idx += 1
    return V, C, R


def SKS(M):
    """SKS storage for sparse skyline format matrices.

    param M : ndarray, matrix to store.

    return (I,V) : 2-tuple of ndarray,
                I[i] = first nnz of the i-th column (1st col doesnt count and last coeff is number of nnz+1)
                V = nnz of the upper triangular part read col by col, left to right.
    """
    n, m = np.shape(M)
    nb_nnz_total = np.count_nonzero(M)

    if nb_nnz_total == 0:
        print("The matrix is the zero matrix, there is nothing to do")
        return None

    v_idx = 1
    V = [M[0, 0]]
    I = []

    for j in range(1, m):  # column by column
        first_nnz = True  # we start with having found the 1st nnz in each column

        for i in range(j + 1):
            if M[i, j] != 0:  # we look for the 1st nnz of the column
                if first_nnz:
                    I.append(v_idx)  # store the first nnz idx of the column
                    first_nnz = False

                    for k in range(i, j + 1):
                        V.append(M[k, j])
                        v_idx += 1
    I.append(len(V) + 1)
    return np.array(I), np.array(V)


def multiply_SKS(I, V, b):
    """Multiply a symmetric skyline matrix by a vector.

    param I : ndarray, from SKS decomposition,
            I[i] = first nnz of the i-th column (1st col doesnt count and last coeff is number of nnz+1)
    param V : ndarray, from SKS decomposition,
            V = nnz of the upper triangular part read col by col, left to right.
    param b: ndarray, vector of values.

    return Ab: ndarray, result of the product.
    """
    n = len(I)
    m = len(b)
    if n != m or n == 0:
        raise ValueError("Error: matrix and vector size are incompatible")

    Ab = np.zeros(n, dtype=float)

    # nb_nnz_col[i] = nb of nnz on the i-th col
    nb_nnz_col = np.zeros(n, dtype=int)

    # diag_start[i] = idx of where the diag of the i-th col starts in V
    diag_start = np.zeros(n, dtype=int)

    nb_nnz_col[0] = 1  # diag is full, A[0,0] is a nnz
    diag_start[0] = 0
    diag_start[-1] = V.shape[0] - 1  # = I[-1] - 2 also works
    for i in range(1, n - 1):
        nb_nnz_col[i] = I[i] - I[i - 1]
        diag_start[i] = I[i] - 1

    nnz = sum(nb_nnz_col)
    nb_nnz_col[-1] = I[-1] - nnz - 1
    # print(nb_nnz_col)
    # print(diag_start)

    Ab[0] += V[0] * b[0]
    for col in range(1, n):
        size_col = nb_nnz_col[col]
        for j in range(size_col):
            d = diag_start[col]
            row = col - j

            # print(f"Ab[{col}] += b[{row}] * {V[d - j]}")
            Ab[col] += b[row] * V[d - j]

            # we dont want to count the diagonal twice
            if j != 0:
                # print(f"Ab[{row}] += b[{row+j}] * {V[d - j]}")
                Ab[row] += b[row + j] * V[d - j]
    return Ab


# Exercice 4 - factorisation LU avec pivot partiel
def LU(M):
    """PM=LU factoriation with partial pivot.

    param M : ndarray, square matrix to factorize, all main minors should be non zero,

    return (L,U,P) : 3-tuple of ndarrays,
                    L = lower triangulare matrix with diag elem = 1,
                    U = upper triangulare matrix,
                    P = permutation matrix used to keep track of the permutations.
    """
    n, m = np.shape(M)
    if n != m or n == 0:
        raise ValueError("Error: M must be a square matrix")

    L = np.zeros((n, n), dtype=float)
    P = np.eye(n)
    U = np.copy(M)

    for i in range(n):
        # we search the abs max value of the i-th col
        # argmax gives us the index of this val
        max_row = np.argmax(np.abs(U[i:, i])) + i  # +i to correct indexes in the view
        # [[]] is used to swap rows
        L[[i, max_row]] = L[[max_row, i]]
        P[[i, max_row]] = P[[max_row, i]]
        U[[i, max_row]] = U[[max_row, i]]
        L[i, i] = 1

        for k in range(i + 1, n):
            if abs(U[i, i]) < TOL:
                raise ValueError("Error: pivot is too close to zero")
            L[k, i] = U[k, i] / U[i, i]
            U[k, 0:n] = U[k, 0:n] - U[k, i] / U[i, i] * U[i, 0:n]

    return L, U, P


# Exercice 5 - résolution d'un système linéaire avec/sans pivot
def Descente(L, b):
    n = np.size(b)
    y = np.zeros(n, dtype=float)

    y[0] = b[0] / L[0, 0]
    for i in range(1, n):
        S = np.sum(L[i, 0:i] * y[0:i])
        y[i] = (b[i] - S) / L[i, i]

    return y


def Remontee(U, y):
    n = np.size(y)
    x = np.zeros(n, dtype=float)

    x[n - 1] = y[n - 1] / U[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        S = np.sum(U[i, i + 1 : n] * x[i + 1 : n])  # operation terme a terme
        x[i] = (y[i] - S) / U[i, i]

    return x


def solve(L, U, b, P=None):
    """Solve linear system with/without partial pivot PAx=Pb, P is optional.

    param L : ndarray, lower triangular matrix from PA=LU factorization.
    param U : ndarray, upper triangular matrix from PA=LU factorization.
    param b : ndarray.
    param P : ndarray, optional pivot matrix from PA=LU factorization.

    return x : ndarray, solution of the system.
    """
    if P is not None:
        b = P @ b  # not in place, b is preserved
    # PA = LU
    # PAx = Pb
    y = Descente(L, b)  # we solve Ly=b
    x = Remontee(U, y)  # we solve Ux=y

    return x


def profile_SKS(I):
    """Compute the profile of a skyline matrix decomposed like in exo2.

    param I : ndarray, from SKS decomposition,
            I[i] = first nnz of the i-th column (1st col doesnt count and last coeff is number of nnz+1)

    return p : int, profile of the skyline matrix.
    """
    n = len(I)  # size of the original matrix
    p = n  # initialize the profile sum with n

    nb_nnz_col = np.zeros(n, dtype=int)
    nb_nnz_col[0] = 1  # diag is full, A[0,0] is a nnz
    for i in range(1, n - 1):
        nb_nnz_col[i] = I[i] - I[i - 1]

    nnz = sum(nb_nnz_col)
    nb_nnz_col[-1] = I[-1] - nnz - 1  # I[-1] is nb nnz +1

    for i in range(1, n):
        h = nb_nnz_col[i] - 1  # -1 because we dont count the diag
        p += i - (i - h)  # = h ...

    return p
