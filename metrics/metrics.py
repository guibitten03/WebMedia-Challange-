import torch
import numpy as np

# https://towardsdatascience.com/demystifying-ndcg-bee3be58cfe0#:~:text=NDCG%20is%20calculated%20by%20dividing,higher%20values%20indicating%20better%20performance.


def ndcg_score(y_true, y_pred):
    if len(y_true) != len(y_pred): return
    n = len(y_true)

    def calculate_dcg(scores):
        dcg = 0
        for i in range(n):
            dcg += (2 ** scores[i] - 1) / np.log2(i + 2)
        return dcg

    dcg_real = calculate_dcg(y_true)
    dcg_predicted = calculate_dcg(y_pred)

    # Cálculo do IDCG usando as relevâncias reais ordenadas
    idcg_real = calculate_dcg(sorted(y_true, reverse=True))

    # Cálculo do NDCG
    ndcg = dcg_predicted / idcg_real

    return ndcg.item()