"""
Based on fbpca, which facebook archived a while ago, but released
under BSD license: https://github.com/facebookarchive/fbpca

---------------------------------------------------------------------

This module contains only the fast pca:

pca
    principal component analysis (singular value decomposition)
"""

import math
import numpy as np
from scipy.linalg import cholesky, eigh, lu, qr, svd, norm, solve
from scipy.sparse import coo_matrix, issparse, spdiags

def mult(A, B):
    """
    default matrix multiplication.

    Multiplies A and B together via the "dot" method.

    Parameters
    ----------
    A : array_like
        first matrix in the product A*B being calculated
    B : array_like
        second matrix in the product A*B being calculated

    Returns
    -------
    array_like
        product of the inputs A and B

    Examples
    --------
    >>> from randomised_pca import mult
    >>> from numpy import array
    >>> from numpy.linalg import norm
    >>>
    >>> A = array([[1., 2.], [3., 4.]])
    >>> B = array([[5., 6.], [7., 8.]])
    >>> norm(mult(A, B) - A.dot(B))

    This example multiplies two matrices two ways -- once with mult,
    and once with the usual "dot" method -- and then calculates the
    (Frobenius) norm of the difference (which should be near 0).
    """

    if issparse(B) and not issparse(A):
        # dense.dot(sparse) is not available in scipy.
        return B.T.dot(A.T).T
    else:
        return A.dot(B)

def pca(A, k=6, raw=False, n_iter=2, l=None):
    """
    Principal component analysis.

    Constructs a nearly optimal rank-k approximation U diag(s) Va to A,
    centering the columns of A first when raw is False, using n_iter
    normalized power iterations, with block size l, started with a
    min(m,n) x l random matrix, when A is m x n; the reference PCA_
    below explains "nearly optimal." k must be a positive integer <=
    the smaller dimension of A, n_iter must be a nonnegative integer,
    and l must be a positive integer >= k.

    The rank-k approximation U diag(s) Va comes in the form of a
    singular value decomposition (SVD) -- the columns of U are
    orthonormal, as are the rows of Va, and the entries of s are all
    nonnegative and nonincreasing. U is m x k, Va is k x n, and
    len(s)=k, when A is m x n.

    Increasing n_iter or l improves the accuracy of the approximation
    U diag(s) Va; the reference PCA_ below describes how the accuracy
    depends on n_iter and l. Please note that even n_iter=1 guarantees
    superb accuracy, whether or not there is any gap in the singular
    values of the matrix A being approximated, at least when measuring
    accuracy as the spectral norm || A - U diag(s) Va || of the matrix
    A - U diag(s) Va (relative to the spectral norm ||A|| of A, and
    accounting for centering when raw is False).

    Notes
    -----
    To obtain repeatable results, reset the seed for the pseudorandom
    number generator.

    The user may ascertain the accuracy of the approximation
    U diag(s) Va to A by invoking diffsnorm(A, U, s, Va), when raw is
    True. The user may ascertain the accuracy of the approximation
    U diag(s) Va to C(A), where C(A) refers to A after centering its
    columns, by invoking diffsnormc(A, U, s, Va), when raw is False.

    Parameters
    ----------
    A : array_like, shape (m, n)
        matrix being approximated
    k : int, optional
        rank of the approximation being constructed;
        k must be a positive integer <= the smaller dimension of A,
        and defaults to 6
    raw : bool, optional
        centers A when raw is False but does not when raw is True;
        raw must be a Boolean and defaults to False
    n_iter : int, optional
        number of normalized power iterations to conduct;
        n_iter must be a nonnegative integer, and defaults to 2
    l : int, optional
        block size of the normalized power iterations;
        l must be a positive integer >= k, and defaults to k+2

    Returns
    -------
    U : ndarray, shape (m, k)
        m x k matrix in the rank-k approximation U diag(s) Va to A or
        C(A), where A is m x n, and C(A) refers to A after centering
        its columns; the columns of U are orthonormal
    s : ndarray, shape (k,)
        vector of length k in the rank-k approximation U diag(s) Va to
        A or C(A), where A is m x n, and C(A) refers to A after
        centering its columns; the entries of s are all nonnegative and
        nonincreasing
    Va : ndarray, shape (k, n)
        k x n matrix in the rank-k approximation U diag(s) Va to A or
        C(A), where A is m x n, and C(A) refers to A after centering
        its columns; the rows of Va are orthonormal

    Examples
    --------
    >>> from numpy.random import uniform
    >>> A = uniform(low=-1.0, high=1.0, size=(100, 2))
    >>> A = A.dot(uniform(low=-1.0, high=1.0, size=(2, 100)))

    Then we can either do:
    >>> from scipy.linalg import svd
    >>> (U, s, Va) = svd(A, full_matrices=False)
    >>> A = A / s[0]

    Or, as provided by this file:
    >>> from randomised_pca pca
    >>> (U, s, Va) = pca(A, 2, True)

    This example produces a rank-2 approximation U diag(s) Va to A such
    that the columns of U are orthonormal, as are the rows of Va, and
    the entries of s are all nonnegative and are nonincreasing.

    References
    ----------
    .. [PCA] Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp,
             Finding structure with randomness: probabilistic
             algorithms for constructing approximate matrix
             decompositions, arXiv:0909.4061 [math.NA; math.PR], 2009
             (available at `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """

    if l is None:
        l = k + 2

    (m, n) = A.shape

    assert k > 0
    assert k <= min(m, n)
    assert n_iter >= 0
    assert l >= k

    if np.isrealobj(A):
        isreal = True
    else:
        isreal = False

    # Promote the types of integer data to float data.
    dtype = (A * 1.0).dtype

    if raw:

        #
        # SVD A directly if l >= m/1.25 or l >= n/1.25.
        #
        if l >= m / 1.25 or l >= n / 1.25:
            (U, s, Va) = svd(
                A.todense() if issparse(A) else A, full_matrices=False)
            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]

        if m >= n:

            #
            # Apply A to a random matrix, obtaining Q.
            #
            if isreal:
                Q = np.random.uniform(low=-1.0, high=1.0, size=(n, l)) \
                    .astype(dtype)
                Q = mult(A, Q)
            if not isreal:
                Q = np.random.uniform(low=-1.0, high=1.0, size=(n, l)) \
                    .astype(dtype)
                Q += 1j * np.random.uniform(low=-1.0, high=1.0, size=(n, l)) \
                    .astype(dtype)
                Q = mult(A, Q)

            #
            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            #
            if n_iter == 0:
                (Q, _) = qr(Q, mode='economic')
            if n_iter > 0:
                (Q, _) = lu(Q, permute_l=True)

            #
            # Conduct normalized power iterations.
            #
            for it in range(n_iter):

                Q = mult(Q.conj().T, A).conj().T

                (Q, _) = lu(Q, permute_l=True)

                Q = mult(A, Q)

                if it + 1 < n_iter:
                    (Q, _) = lu(Q, permute_l=True)
                else:
                    (Q, _) = qr(Q, mode='economic')

            #
            # SVD Q'*A to obtain approximations to the singular values
            # and right singular vectors of A; adjust the left singular
            # vectors of Q'*A to approximate the left singular vectors
            # of A.
            #
            QA = mult(Q.conj().T, A)
            (R, s, Va) = svd(QA, full_matrices=False)
            U = Q.dot(R)

            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]

        if m < n:

            #
            # Apply A' to a random matrix, obtaining Q.
            #
            if isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(l, m)) \
                    .astype(dtype)
            if not isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(l, m)) \
                    .astype(dtype)
                R += 1j * np.random.uniform(low=-1.0, high=1.0, size=(l, m)) \
                    .astype(dtype)

            Q = mult(R, A).conj().T

            #
            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            #
            if n_iter == 0:
                (Q, _) = qr(Q, mode='economic')
            if n_iter > 0:
                (Q, _) = lu(Q, permute_l=True)

            #
            # Conduct normalized power iterations.
            #
            for it in range(n_iter):

                Q = mult(A, Q)
                (Q, _) = lu(Q, permute_l=True)

                Q = mult(Q.conj().T, A).conj().T

                if it + 1 < n_iter:
                    (Q, _) = lu(Q, permute_l=True)
                else:
                    (Q, _) = qr(Q, mode='economic')

            #
            # SVD A*Q to obtain approximations to the singular values
            # and left singular vectors of A; adjust the right singular
            # vectors of A*Q to approximate the right singular vectors
            # of A.
            #
            (U, s, Ra) = svd(mult(A, Q), full_matrices=False)
            Va = Ra.dot(Q.conj().T)

            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]

    if not raw:

        #
        # Calculate the average of the entries in every column.
        #
        c = A.sum(axis=0) / m
        c = c.reshape((1, n))

        #
        # SVD the centered A directly if l >= m/1.25 or l >= n/1.25.
        #
        if l >= m / 1.25 or l >= n / 1.25:
            (U, s, Va) = svd(
                (A.todense() if issparse(A) else A) -
                np.ones((m, 1), dtype=dtype).dot(c), full_matrices=False)
            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]

        if m >= n:

            #
            # Apply the centered A to a random matrix, obtaining Q.
            #
            if isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(n, l)) \
                    .astype(dtype)
            if not isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(n, l)) \
                    .astype(dtype)
                R += 1j * np.random.uniform(low=-1.0, high=1.0, size=(n, l)) \
                    .astype(dtype)

            Q = mult(A, R) - np.ones((m, 1), dtype=dtype).dot(c.dot(R))

            #
            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            #
            if n_iter == 0:
                (Q, _) = qr(Q, mode='economic')
            if n_iter > 0:
                (Q, _) = lu(Q, permute_l=True)

            #
            # Conduct normalized power iterations.
            #
            for it in range(n_iter):

                Q = mult(Q.conj().T, A) \
                    - (Q.conj().T.dot(np.ones((m, 1), dtype=dtype))).dot(c)
                Q = Q.conj().T
                (Q, _) = lu(Q, permute_l=True)

                Q = mult(A, Q) - np.ones((m, 1), dtype=dtype).dot(c.dot(Q))

                if it + 1 < n_iter:
                    (Q, _) = lu(Q, permute_l=True)
                else:
                    (Q, _) = qr(Q, mode='economic')

            #
            # SVD Q' applied to the centered A to obtain
            # approximations to the singular values and right singular
            # vectors of the centered A; adjust the left singular
            # vectors to approximate the left singular vectors of the
            # centered A.
            #
            QA = mult(Q.conj().T, A) \
                - (Q.conj().T.dot(np.ones((m, 1), dtype=dtype))).dot(c)
            (R, s, Va) = svd(QA, full_matrices=False)
            U = Q.dot(R)

            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]

        if m < n:

            #
            # Apply the adjoint of the centered A to a random matrix,
            # obtaining Q.
            #
            if isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(l, m)) \
                    .astype(dtype)
            if not isreal:
                R = np.random.uniform(low=-1.0, high=1.0, size=(l, m)) \
                    .astype(dtype)
                R += 1j * np.random.uniform(low=-1.0, high=1.0, size=(l, m)) \
                    .astype(dtype)

            Q = mult(R, A) - (R.dot(np.ones((m, 1), dtype=dtype))).dot(c)
            Q = Q.conj().T

            #
            # Form a matrix Q whose columns constitute a
            # well-conditioned basis for the columns of the earlier Q.
            #
            if n_iter == 0:
                (Q, _) = qr(Q, mode='economic')
            if n_iter > 0:
                (Q, _) = lu(Q, permute_l=True)

            #
            # Conduct normalized power iterations.
            #
            for it in range(n_iter):

                Q = mult(A, Q) - np.ones((m, 1), dtype=dtype).dot(c.dot(Q))
                (Q, _) = lu(Q, permute_l=True)

                Q = mult(Q.conj().T, A) \
                    - (Q.conj().T.dot(np.ones((m, 1), dtype=dtype))).dot(c)
                Q = Q.conj().T

                if it + 1 < n_iter:
                    (Q, _) = lu(Q, permute_l=True)
                else:
                    (Q, _) = qr(Q, mode='economic')

            #
            # SVD the centered A applied to Q to obtain approximations
            # to the singular values and left singular vectors of the
            # centered A; adjust the right singular vectors to
            # approximate the right singular vectors of the centered A.
            #
            (U, s, Ra) = svd(
                mult(A, Q) - np.ones((m, 1), dtype=dtype).dot(c.dot(Q)),
                full_matrices=False)
            Va = Ra.dot(Q.conj().T)

            #
            # Retain only the leftmost k columns of U, the uppermost
            # k rows of Va, and the first k entries of s.
            #
            return U[:, :k], s[:k], Va[:k, :]

if __name__=="__main__":
    from numpy.random import uniform
    #show we can accurately recover the rank of a noisy square matrix
    for k in range(15,35):
        A = uniform(low=-1.,high=1.,size=(10000,k))
        A = A@(uniform(low=-1.,high=1.,size=(10000,k)).T)
        _,s,_ = pca(A,2*k,True)
        s = s.round(2)
        sm = int(round(sum(np.where(s>.5,1,0),start=0.),0))
        assert sm==k