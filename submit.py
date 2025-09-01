import numpy as np
from sklearn.linear_model import LogisticRegression

# Feature mapping function φ(C) -> R^D
def my_map(C):

    C = 2 * C - 1  # Convert from {0,1} to {-1,+1}
    N, k = C.shape
    features = [np.ones((N, 1))]  # Bias term
    features.append(C)  # Linear terms

    # Quadratic terms
    for i in range(k):
        for j in range(i + 1, k):
            features.append((C[:, i] * C[:, j])[:, None])

    return np.hstack(features)  # Shape (N, D)

def my_fit(C_train, R_train):

    X_train = my_map(C_train)

    # Logistic regression with L2 regularization strength
    clf = LogisticRegression(penalty='l2', solver='lbfgs', fit_intercept=True, max_iter=1000, C=100.0)
    clf.fit(X_train, R_train)

    W = clf.coef_.flatten()
    b = clf.intercept_[0]
    return W, b

def my_decode(w):
    # Separate weights and bias
    w = np.array(w[:-1])
    b = w[-1]

    # Number of elements (should be 64)
    length = w.shape[0]

    # Allocate output delay arrays
    p = np.zeros(length)
    q = np.zeros(length)
    r = np.zeros(length)
    s = np.zeros(length)

    alpha_vec = w
    beta_vec = np.zeros_like(alpha_vec)
    beta_vec[-1] = b

    # Compute all four delay vectors using max(., 0)
    p = np.maximum(alpha_vec + beta_vec, 0)
    q = np.maximum(-(alpha_vec + beta_vec), 0)
    r = np.maximum(alpha_vec - beta_vec, 0)
    s = np.maximum(-(alpha_vec - beta_vec), 0)

    return p, q, r, s

