import numpy as np


def update_adam(t, w, b, dw, db, m_dw, v_dw, m_db, v_db, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m_dw = beta1 * m_dw + (1 - beta1) * dw
    m_db = beta1 * m_db + (1 - beta1) * db
    v_dw = beta2 * v_dw + (1 - beta2) * (dw ** 2)
    v_db = beta2 * v_db + (1 - beta2) * db

    # bias correction
    m_dw_corr = m_dw / (1 - beta1 ** t)
    m_db_corr = m_db / (1 - beta1 ** t)
    v_dw_corr = v_dw / (1 - beta2 ** t)
    v_db_corr = v_db / (1 - beta2 ** t)

    # update weights and biases
    w = w - eta * (m_dw_corr / (np.sqrt(v_dw_corr) + epsilon))
    b = b - eta * (m_db_corr / (np.sqrt(v_db_corr) + epsilon))

    return w, b, m_dw, v_dw, m_db, v_db

