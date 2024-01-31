import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

def evaluate_kpi(model, test, train=None, k=10):
    valid_user = 0.0
    NDCG = 0.0
    HT = 0.0

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)
        predictions = predictions.argsort()

        if train is not None:
            rated = set(train[user_id].indices)
        else:
            rated = []

        predictions = [p for p in predictions if p not in rated]

        targets = row.indices
        pred = predictions[:k]
        valid_user += 1
        for i, p in enumerate(pred):
            if p in targets and p == predictions[i]:
                NDCG += 1 / np.log2(i + 2)
                HT += 1
    
    return NDCG / valid_user, HT / valid_user
















