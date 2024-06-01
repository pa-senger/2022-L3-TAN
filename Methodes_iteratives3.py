import numpy as np
import scipy.sparse as sp
import ilupp
import warnings

warnings.filterwarnings("ignore", message="splu converted its input to CSC format")
warnings.filterwarnings(
    "ignore",
    message="spsolve is more efficient when sparse b is in the CSC matrix format",
)


def jacobi(A, b, tol=1e-10, prec="jacobi", max_iter=100):
    """Solves Ax=b via an iterative method.

    Args:
        A (ndarray or sparray): matrix of the system
        b (ndarray): rhs, dense array of shape (n,1) or (n,)
        tol (int, optional): tolerence. Defaults to 1e-10.
        prec (str, optional): preconditioner {jacobi, gs, ilu}. Defaults to "jacobi".
        max_iter (int, optional): maximum iterations. Defaults to 100.

    Raises:
        ValueError: if prec not supported

    Returns:
        ndarray: vector solution of shape (n,)
    """
    A = sp.csc_matrix(A)

    # checking if A is really inversible is long ...
    # try:
    #     sp.linalg.inv(A)
    # except RuntimeError:
    #     print("Error: A is exactly singular")
    #     return None, 0

    n = len(b)
    assert n == A.shape[0]
    b = b.squeeze()

    # if the diagonal is bad
    reg_param = 1e-6
    A += reg_param * sp.eye(n)

    if prec == "jacobi":
        M = sp.diags(A.diagonal(), format="csr")  # M = D

    elif prec == "gs":
        M = sp.tril(A, format="csr")  # M = D - E, i.e., the lower triangular part of A

    elif prec == "ilu":
        M = sp.linalg.spilu(A, drop_tol=1e-10, fill_factor=20)  # returns an operator
        Mh = M.solve

    else:
        raise ValueError(f"Unsupported preconditioner: {prec}")

    x = np.random.rand(n)  # start with a random vector of solution
    r = A @ x - b  # initialize residu
    norm_b = np.linalg.norm(b)  # used in relative residu

    niter = 0  # number of iterations
    while (np.linalg.norm(r) / norm_b) > tol and niter < max_iter:
        if prec == "ilu":  # only 1 comparaison
            # y = Minv * r <=> My = r, so we dont have to use the inverse of M
            x = x - Mh(r)

        else:  # no comparaison, prec == gs or jacobi
            # y = Minv * r <=> My = r, so we dont have to use the inverse of M
            x = x - sp.linalg.spsolve_triangular(M, r, lower=True)

        r = A @ x - b  # actualize residu
        niter += 1

    if niter == max_iter:
        cond = np.linalg.cond(A.toarray())
        message = f"the method didn't converge in {niter} iterations with tol={tol} and cond(A)={cond}"
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    return x, niter


def SOR(A, b, w, tol=1e-10, max_iter=100):
    """Solve Ax=b via the SOR iterative method.

    Args:
        A (ndarray or sparray): matrix of the system
        b (ndarray): rhs, dense ndarray of shape (n,) or (n,1)
        w (float): omega parameter of SOR
        tol (int, optional): tolerence. Defaults to 1e-10.
        max_iter (int, optional): maximum iterations. Defaults to 100.

    Raises:
        ValueError: if not (0 < w < 2)

    Returns:
        ndarray: vector solution of shape (n,)
    """
    if w == 1:
        x, niter = exo1(A, b, prec="gs")
        return x, niter

    # try:
    #     sp.linalg.inv(A)
    # except RuntimeError:
    #     print("Error: A is exactly singular")
    #     return None, 0

    if not (0 < w < 2):
        raise ValueError(
            f"the method cannot converge if w={w} is not in the open interval (0, 2)"
        )

    A = sp.csc_matrix(A)

    n = len(b)
    assert n == A.shape[0]
    b = b.squeeze()

    # if the diagonal is bad
    reg_param = 1e-6
    A += reg_param * sp.eye(n)

    D = sp.diags(A.diagonal(), format="csr")
    L = sp.tril(A, format="csr")
    M = D / w + L - D  # -D since its already in L so we dont count it twice

    x = np.random.rand(n)  # start with a random vector of solution
    r = A @ x - b  # initialize residu
    norm_b = np.linalg.norm(b)  # for the relative residu

    niter = 0
    while (np.linalg.norm(r) / norm_b) > tol and niter < max_iter:
        x = x - sp.linalg.spsolve_triangular(M, r, lower=True)  # <=> x = x-M_inv*r
        r = A @ x - b  # actualize residu
        niter += 1

    if niter == max_iter:
        cond = np.linalg.cond(A.toarray())
        message = f"the method didn't converge in {niter} iterations with tol={tol} and cond(A)={cond}"
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    return x, niter


def solve(A, b, tol=1e-10, prec="jacobi", max_iter=100):
    """Solve Ax=b via the precondionned conjugated gradient method.

    Args:
        A (ndarray or sparray): matrix of the system
        b (ndarray): rhs, dense vector of shape (n,) or (n,1)
        tol (float, optional): tolerence. Defaults to 1e-10.
        prec (str, optional): precontioner {jacobi, ichol}. Defaults to "jacobi".
        max_iter (int, optional): maximum iterations. Defaults to 100.

    Raises:
        ValueError: if prec is not supported

    Returns:
        ndarray: vector of solution of shape (n,)
    """
    A = sp.csc_matrix(A)

    n = len(b)
    assert n == A.shape[0]
    b = b.squeeze()

    # if the diagonal is bad
    reg_param = 1e-6
    A += reg_param * sp.eye(n)

    if prec == "jacobi":
        M = sp.diags(A.diagonal(), format="csr")  # M = D

    elif prec == "ichol":
        M = ilupp.ichol0(A)

    else:
        raise ValueError(f"Unsupported preconditioner: {prec}")

    x = np.random.rand(n)  # start from random vector of solutions
    r = b - A @ x  # init residu
    z = sp.linalg.spsolve_triangular(M, r, lower=True)  # init z = M_inv*r
    d = z  # init d the descent direction
    norm_b = np.linalg.norm(b)  # used for relative residu

    niter = 0
    while (np.linalg.norm(r) / norm_b) > tol and niter < max_iter:
        Ad = A * d  # we need it twice so better to compute only once
        s = np.dot(r, r) / np.dot(Ad, d)
        x = x + s * d  # current sol
        zr = np.dot(z, r)  # saves the past <z,r> for beta
        r = r - s * Ad  # actualize residu
        z = sp.linalg.spsolve_triangular(M, r, lower=True)  # <=> z = M_inv * r
        beta = -np.dot(z, r) / zr
        d = z + beta * d  # actualize descent direction

        niter += 1

    if niter == max_iter:
        cond = np.linalg.cond(A.toarray())
        message = f"the method didn't converge in {niter} iterations with tol={tol} and cond(A)={cond}"
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    return x, niter
