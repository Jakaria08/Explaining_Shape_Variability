import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def sap_regression_holdout(train_factors, train_codes, eval_factors, eval_codes):
    """Compute regression SAP with train/eval split.

    Args:
        train_factors: (n_train, n_factors)
        train_codes: (n_train, n_codes)
        eval_factors: (n_eval, n_factors)
        eval_codes: (n_eval, n_codes)

    Returns:
        sap_score: float
        s_matrix: (n_factors, n_codes) clipped non-negative R2 values on eval set
        pred_matrix: (n_factors, n_codes, n_eval) predictions on eval set
    """
    train_factors = np.asarray(train_factors)
    train_codes = np.asarray(train_codes)
    eval_factors = np.asarray(eval_factors)
    eval_codes = np.asarray(eval_codes)

    if train_factors.ndim != 2 or train_codes.ndim != 2:
        raise ValueError('train_factors and train_codes must be 2D arrays.')
    if eval_factors.ndim != 2 or eval_codes.ndim != 2:
        raise ValueError('eval_factors and eval_codes must be 2D arrays.')
    if train_factors.shape[0] != train_codes.shape[0]:
        raise ValueError('train_factors and train_codes must have the same number of rows.')
    if eval_factors.shape[0] != eval_codes.shape[0]:
        raise ValueError('eval_factors and eval_codes must have the same number of rows.')
    if train_factors.shape[1] != eval_factors.shape[1]:
        raise ValueError('train_factors and eval_factors must have the same number of factors.')

    nb_factors = train_factors.shape[1]
    nb_codes = train_codes.shape[1]
    n_eval = eval_codes.shape[0]

    if nb_factors == 0 or nb_codes == 0:
        return 0.0, np.zeros((nb_factors, nb_codes), dtype=np.float64), np.zeros(
            (nb_factors, nb_codes, n_eval), dtype=np.float64
        )

    s_matrix = np.zeros((nb_factors, nb_codes), dtype=np.float64)
    pred_matrix = np.zeros((nb_factors, nb_codes, n_eval), dtype=np.float64)

    for f in range(nb_factors):
        y_train = train_factors[:, f].reshape(-1, 1)
        y_eval = eval_factors[:, f].reshape(-1)

        for c in range(nb_codes):
            x_train = train_codes[:, c].reshape(-1, 1)
            x_eval = eval_codes[:, c].reshape(-1, 1)

            reg = LinearRegression()
            reg.fit(x_train, y_train)
            y_pred = reg.predict(x_eval).reshape(-1)

            pred_matrix[f, c, :] = y_pred

            r2 = r2_score(y_eval, y_pred)
            if not np.isfinite(r2):
                r2 = 0.0
            s_matrix[f, c] = max(0.0, float(r2))

    sum_gap = 0.0
    for f in range(nb_factors):
        row = s_matrix[f, :]
        if nb_codes == 1:
            gap = float(row[0])
        else:
            top_two = np.sort(row)[-2:]
            gap = float(top_two[1] - top_two[0])
        sum_gap += gap

    sap_score = float(sum_gap / nb_factors)
    return sap_score, s_matrix, pred_matrix
