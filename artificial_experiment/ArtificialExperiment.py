import numpy as np
import pandas as pd

import os
import statistics
import multiprocessing

from tqdm import tqdm

def CC(test, thr=0.5):
    result = np.sum(test >= thr) / len(test)
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)

def ACC(test, TprFpr, thr=0.5):
    dC = CC(test)

    tpr_fpr_row = TprFpr[TprFpr[:, 0] == thr, 1:3].astype(float)
    if tpr_fpr_row.size == 0:
        raise ValueError("Threshold value not found in TprFpr.")

    tpr, fpr = tpr_fpr_row[0]

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)


def DyS(p_score, n_score, test, measure="topsoe", bins=np.arange(2, 22, 2), err=1e-5):
    results = []

    for b_size in bins:
        Sty_1 = getHist(p_score, b_size)
        Sty_2 = getHist(n_score, b_size)
        Uy = getHist(test, b_size)

        def f(x):
            return DySyn_distance(np.vstack([(Sty_1 * x) + (Sty_2 * (1 - x)), Uy]), method=measure)

        best_alpha = TernarySearch(0, 1, f, err)
        results.append(best_alpha)

    result = statistics.median(results)
    result = max(0, min(result, 1))

    return np.array([result, 1 - result])

def DySyn(ts, measure, MF=np.arange(0.1, 1.0, 0.2)):
    MF = np.round(MF, 2)

    results = []
    distances = []

    for mf in MF:
        scores = MoSS(1000, 0.5, mf)
        test_p = scores[scores[:, 2] == 1, 0]
        test_n = scores[scores[:, 2] == 2, 0]

        if measure == "sord":
            rQnt = DySyn_SORD(test_p, test_n, ts)
        else:
            rQnt = DySyn_DyS(test_p, test_n, ts, measure, [10])

        distances.append(rQnt[1])
        results.append(rQnt[0][0])

    best_result = round(results[np.argmin(distances)], 2)
    return [np.array([best_result, 1 - best_result]), min(distances), MF[np.argmin(distances)]]

def DySyn_DyS(p_score, n_score, test, measure="hellinger", b_sizes = list(range(2, 21, 2)) + [30]):
    results = []
    vDistAll = []

    for b_size in b_sizes:
        Sty_1 = getHist(p_score, b_size)
        Sty_2 = getHist(n_score, b_size)
        Uy = getHist(test, b_size)

        def f(x):
            return DySyn_distance(np.vstack([(Sty_1 * x) + (Sty_2 * (1 - x)), Uy]), method=measure)

        best_alpha = TernarySearch(0, 1, f, 1e-2)
        results.append(best_alpha)
        vDistAll.append(f(best_alpha))

    median_result = np.median(results)
    return [np.array([round(median_result, 2), 1 - round(median_result, 2)]), min(vDistAll)]

def PNTDiff(pos, neg, test, pos_prop):
    p_w = pos_prop / len(pos)
    n_w = (1 - pos_prop) / len(neg)
    t_w = -1 / len(test)

    p = np.column_stack((pos, np.full(len(pos), p_w)))
    n = np.column_stack((neg, np.full(len(neg), n_w)))
    t = np.column_stack((test, np.full(len(test), t_w)))

    v = np.vstack((p, n, t))
    v = v[v[:, 0].argsort()]

    acc = v[0, 1]
    total_cost = 0

    for i in range(1, len(v)):
        cost_mul = v[i, 0] - v[i - 1, 0]
        total_cost += abs(cost_mul * acc)
        acc += v[i, 1]

    return total_cost

def DySyn_SORD(p_score, n_score, test):
    def f(x):
        return PNTDiff(p_score, n_score, test, x)

    best_alpha = TernarySearch(0, 1, f, 1e-5)
    vDist = f(best_alpha)

    return [np.array([round(best_alpha, 2), 1 - round(best_alpha, 2)]), vDist]


# Utils
def getTPRandFPRbyThreshold(validation_scores):
    unique_scores = np.arange(0.01, 1.00, 0.01)
    arrayOfTPRandFPRByTr = []

    total_positive = np.sum(validation_scores[:, 2] == 1)
    total_negative = np.sum(validation_scores[:, 2] == 2)

    for threshold in unique_scores:
        fp = np.sum((validation_scores[:, 0] > threshold) & (validation_scores[:, 2] == 2))
        tp = np.sum((validation_scores[:, 0] > threshold) & (validation_scores[:, 2] == 1))
        tpr = tp / total_positive if total_positive > 0 else 0
        fpr = fp / total_negative if total_negative > 0 else 0

        arrayOfTPRandFPRByTr.append([round(threshold, 2), tpr, fpr])

    return np.array(arrayOfTPRandFPRByTr, dtype=object)


def DySyn_distance(x, method="hellinger"):
    if method == "ord":
        x_dif = x[0, :] - x[1, :]
        acum = 0
        aux = 0
        for val in x_dif:
            aux += val
            acum += aux
        return abs(acum)

    if method == "topsoe":
        return sum(x[0, i] * np.log((2 * x[0, i]) / (x[0, i] + x[1, i])) +
                   x[1, i] * np.log((2 * x[1, i]) / (x[1, i] + x[0, i])) for i in range(x.shape[1]))

    if method == "jensen_difference":
        return sum(((x[0, i] * np.log(x[0, i]) + x[1, i] * np.log(x[1, i])) / 2) -
                   ((x[0, i] + x[1, i]) / 2) * np.log((x[0, i] + x[1, i]) / 2) for i in range(x.shape[1]))

    if method == "taneja":
        return sum(((x[0, i] + x[1, i]) / 2) * np.log((x[0, i] + x[1, i]) / (2 * np.sqrt(x[0, i] * x[1, i])))
                   for i in range(x.shape[1]))

    if method == "hellinger":
        return 2 * np.sqrt(1 - sum(np.sqrt(x[0, i] * x[1, i]) for i in range(x.shape[1])))

    if method == "prob_symm":
        return 2 * sum(((x[0, i] - x[1, i]) ** 2) / (x[0, i] + x[1, i]) for i in range(x.shape[1]))

    raise ValueError("measure argument must be a valid option")

def getHist(scores, nbins):
    breaks = np.linspace(0, 1, nbins + 1)
    breaks[-1] = 1.1
    re = np.full(len(breaks) - 1, 1 / (len(breaks) - 1))

    for i in range(1, len(breaks)):
        re[i - 1] = (re[i - 1] + np.sum((scores >= breaks[i - 1]) & (scores < breaks[i]))) / (len(scores) + 1)

    return re

def TernarySearch(left, right, f, eps=1e-4):
    while True:
        if abs(left - right) < eps:
            return (left + right) / 2

        leftThird = left + (right - left) / 3
        rightThird = right - (right - left) / 3

        if f(leftThird) > f(rightThird):
            left = leftThird
        else:
            right = rightThird


def MoSS(n, alpha, m):
    p_score = np.random.uniform(size=int(n * alpha)) ** m
    n_score = 1 - (np.random.uniform(size=int(round(n * (1 - alpha), 0))) ** m)
    scores = np.column_stack((np.concatenate((p_score, n_score)), np.concatenate((p_score, n_score)), np.concatenate((np.ones(len(p_score)), np.full(len(n_score), 2)))))

    return scores


def apply_qntMethod(qntMethod, p_score, n_score, test, TprFpr=None, thr=None, measure="hellinger", MF_dysyn=np.arange(0.1, 1.0, 0.2)):
    if qntMethod == "CC":
        return CC(test=test, thr=thr)

    if qntMethod == "ACCSyn":
        return ACCSyn(ts=test, measure=measure, MF_dysyn=MF_dysyn)

    if qntMethod == "XSyn":
        return XSyn(ts=test, measure=measure, MF_dysyn=MF_dysyn)

    if qntMethod == "MAXSyn":
        return MAXSyn(ts=test, measure=measure, MF_dysyn=MF_dysyn)

    if qntMethod == "T50Syn":
        return T50Syn(ts=test, measure=measure, MF_dysyn=MF_dysyn)

    if qntMethod == "MSSyn":
        return MSSyn(ts=test, measure=measure, MF_dysyn=MF_dysyn)

    if qntMethod == "MS2Syn":
        return MS2Syn(ts=test, measure=measure, MF_dysyn=MF_dysyn)

    if qntMethod == "SMMSyn":
        return SMMSyn(ts=test, measure=measure, MF_dysyn=MF_dysyn)
    
    if qntMethod == "PACCSyn":
        return PACCSyn(ts=test, measure=measure, MF_dysyn=MF_dysyn)

    if qntMethod == "ACC":
        return ACC(test=test, TprFpr=TprFpr, thr=thr)

    if qntMethod == "T50":
        return T50(ts=test, TprFpr=TprFpr)

    if qntMethod == "X":
        return X(ts=test, TprFpr=TprFpr)

    if qntMethod == "MAX":
        return MAX(ts=test, TprFpr=TprFpr)

    if qntMethod == "PCC":
        return PCC(ts=test)

    if qntMethod == "PACC":
        return PACC(ts=test, TprFpr=TprFpr, thr=thr)

    if qntMethod == "DySyn":
        return DySyn(ts=test, measure=measure, MF=MF_dysyn)

    if qntMethod == "HDy":
        return HDy(p_score=p_score, n_score=n_score, test=test)
    
    if qntMethod == "HDySyn":
        return HDySyn(ts=test, MF=MF_dysyn)

    if qntMethod == "DyS":
        return DyS(p_score=p_score, n_score=n_score, test=test, measure=measure)

    if qntMethod == "MS":
        return MS(ts=test, TprFpr=TprFpr)

    if qntMethod == "MS2":
        return MS2(ts=test, TprFpr=TprFpr)

    if qntMethod == "SMM":
        return SMM(p_scores=p_score, n_scores=n_score, t_scores=test)

    print("ERROR - Quantification method was not applied!")

    return None


# Proposal
def ACCSyn(ts, measure, MF_dysyn):
    rQnt = DySyn(ts, measure, MF_dysyn)
    TprFpr = np.array(getTPRandFPRbyThreshold(MoSS(1000, 0.5, rQnt[2]))).astype(float)  # Implement getTPRandFPRbyThreshold

    dC = CC(ts)  # Implement CC function separately

    tpr_fpr_row = TprFpr[TprFpr[:, 0] == 0.5, 1:3].astype(float)
    if tpr_fpr_row.size == 0:
        raise ValueError("Threshold value not found in TprFpr.")

    tpr, fpr = tpr_fpr_row[0]

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)

def X(ts, TprFpr):
    min_index = abs((1 - TprFpr[:, 1]) - TprFpr[:, 2])
    min_index = np.argmin(min_index)

    tpr_fpr_row = TprFpr[min_index, 0:3].astype(float)

    if tpr_fpr_row.size == 0:
        raise ValueError("Threshold value not found in TprFpr.")

    thr, tpr, fpr = tpr_fpr_row
    dC = CC(ts, thr)  # Implement CC function separately

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)

def XSyn(ts, measure, MF_dysyn):
    rQnt = DySyn(ts, measure, MF_dysyn)
    TprFpr = np.array(getTPRandFPRbyThreshold(MoSS(1000, 0.5, rQnt[2]))).astype(float)
    
    min_index = abs((1 - TprFpr[:, 1]) - TprFpr[:, 2])
    min_index = np.argmin(min_index)

    tpr_fpr_row = TprFpr[min_index, 0:3].astype(float)

    thr, tpr, fpr = tpr_fpr_row
    dC = CC(ts, thr)

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)

def MAX(ts, TprFpr):
    max_index = abs(TprFpr[:, 1] - TprFpr[:, 2])
    max_index = np.argmax(max_index)

    tpr_fpr_row = TprFpr[max_index, 0:3].astype(float)

    thr, tpr, fpr = tpr_fpr_row
    dC = CC(ts, thr)  # Implement CC function separately

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)

def MAXSyn(ts, measure, MF_dysyn):
    rQnt = DySyn(ts, measure, MF_dysyn)
    TprFpr = np.array(getTPRandFPRbyThreshold(MoSS(1000, 0.5, rQnt[2]))).astype(float)

    # Usual MAX implementation
    max_index = abs(TprFpr[:, 1] - TprFpr[:, 2])
    max_index = np.argmax(max_index)

    tpr_fpr_row = TprFpr[max_index, 0:3].astype(float)

    thr, tpr, fpr = tpr_fpr_row
    dC = CC(ts, thr)  # Implement CC function separately

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)


def T50(ts, TprFpr):
    min_index = abs(TprFpr[:, 1] - 0.5)
    min_index = np.argmin(min_index)

    tpr_fpr_row = TprFpr[min_index, 0:3].astype(float)

    thr, tpr, fpr = tpr_fpr_row
    dC = CC(ts, thr)

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)


def T50Syn(ts, measure, MF_dysyn):
    rQnt = DySyn(ts, measure, MF_dysyn)
    TprFpr = np.array(getTPRandFPRbyThreshold(MoSS(1000, 0.5, rQnt[2]))).astype(float)

    return T50(ts, TprFpr)


def MS(ts, TprFpr):
  results = []

  threshold_set = np.arange(0.01, 1.00, 0.01)

  for threshold in threshold_set:
    threshold = round(threshold, 2)
    tpr, fpr = TprFpr[TprFpr[:, 0] == threshold, 1:3].astype(float)[0]
    dC = CC(ts, threshold)

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    results.append(result)

    result = np.median(results)
  return np.array([result, 1 - result], dtype=float)


def MS2(ts, TprFpr):
    results = []

    index = np.where(abs(TprFpr[:,1]-TprFpr[:,2]) > (1/4))[0].tolist()
    threshold_set = TprFpr[index,0]
    if threshold_set.shape[0] == 0:
      threshold_set = np.arange(0.01, 1.00, 0.01)

    for threshold in threshold_set:
      threshold = round(threshold, 2)
      tpr, fpr = TprFpr[TprFpr[:, 0] == threshold, 1:3].astype(float)[0]
      dC = CC(ts, threshold)

      result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
      result = max(0, min(result, 1))

      results.append(result)

    result = np.median(results)
    return np.array([result, 1 - result], dtype=float)


def MSSyn(ts, measure, MF_dysyn):
    results = []

    rQnt = DySyn(ts, measure, MF_dysyn)
    TprFpr = np.array(getTPRandFPRbyThreshold(MoSS(1000, 0.5, rQnt[2]))).astype(float)

    threshold_set = np.arange(0.01, 1.00, 0.01)

    for threshold in threshold_set:
      threshold = round(threshold, 2)
      tpr, fpr = TprFpr[TprFpr[:, 0] == threshold, 1:3].astype(float)[0]
      dC = CC(ts, threshold)

      result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
      result = max(0, min(result, 1))

      results.append(result)

    result = np.median(results)
    return np.array([result, 1 - result], dtype=float)


def MS2Syn(ts, measure, MF_dysyn):
    results = []

    rQnt = DySyn(ts, measure, MF_dysyn)
    TprFpr = np.array(getTPRandFPRbyThreshold(MoSS(1000, 0.5, rQnt[2]))).astype(float)
    
    index = np.where(abs(TprFpr[:,1]-TprFpr[:,2]) > (1/4))[0].tolist()
    threshold_set = TprFpr[index,0]
    if threshold_set.shape[0] == 0:
      threshold_set = np.arange(0.01, 1.00, 0.01)

    for threshold in threshold_set:
      threshold = round(threshold, 2)
      tpr, fpr = TprFpr[TprFpr[:, 0] == threshold, 1:3].astype(float)[0]
      dC = CC(ts, threshold)

      result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
      result = max(0, min(result, 1))

      results.append(result)

    result = np.median(results)
    return np.array([result, 1 - result], dtype=float)


def PCC(ts):
    pos_score_mean = ts.mean()
    result = max(0, min(pos_score_mean, 1))

    return np.array([result, 1 - result], dtype=float)


def PACC(ts, TprFpr, thr):
    dC = PCC(ts)

    tpr_fpr_row = TprFpr[TprFpr[:, 0] == thr, 1:3].astype(float)
    if tpr_fpr_row.size == 0:
        raise ValueError("Threshold value not found in TprFpr.")

    tpr, fpr = tpr_fpr_row[0]

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)


def PACCSyn(ts, measure, MF_dysyn):
    rQnt = DySyn(ts, measure, MF_dysyn)
    TprFpr = np.array(getTPRandFPRbyThreshold(MoSS(1000, 0.5, rQnt[2]))).astype(float)

    dC = PCC(ts)

    tpr_fpr_row = TprFpr[TprFpr[:, 0] == 0.5, 1:3].astype(float)
    if tpr_fpr_row.size == 0:
        raise ValueError("Threshold value not found in TprFpr.")

    tpr, fpr = tpr_fpr_row[0]

    result = (dC[0] - fpr) / (tpr - fpr) if (tpr - fpr) != 0 else 0
    result = max(0, min(result, 1))

    return np.array([result, 1 - result], dtype=float)


def SMM(p_scores, n_scores, t_scores):
  mean_p_scores = np.mean(p_scores)
  mean_n_scores = np.mean(n_scores)
  mean_t_scores = np.mean(t_scores)

  alpha = (mean_t_scores - mean_n_scores) / (mean_p_scores - mean_n_scores)
  alpha = max(0, min(alpha, 1))

  return np.round([alpha, abs(1-alpha)], 2)


def SMMSyn(ts, measure, MF_dysyn):
    rQnt = DySyn(ts, measure, MF_dysyn)
    best_scores = MoSS(1000, 0.5, rQnt[2])

    best_p = best_scores[best_scores[:, 2] == 1, 0]
    best_n = best_scores[best_scores[:, 2] == 2, 0]
    result = SMM(best_p, best_n, ts)

    return result


def HDy(p_score, n_score, test, err=1e-5):
    bins=np.linspace(10, 110, 11).astype(int)
    measure = 'hellinger'

    results = []
    distances = []

    for b_size in bins:
        Sty_1 = getHist(p_score, b_size)
        Sty_2 = getHist(n_score, b_size)
        Uy = getHist(test, b_size)

        def f(x):
            return DySyn_distance(np.vstack([(Sty_1 * x) + (Sty_2 * (1 - x)), Uy]), method=measure)

        best_alpha = TernarySearch(0, 1, f, err)
        results.append(best_alpha)
        distances.append(f(best_alpha))

    result = statistics.median(results)
    result = max(0, min(result, 1))

    return [np.array([result, 1 - result]), min(distances)]


def HDySyn(ts, MF=np.arange(0.1, 1.0, 0.2)):
    MF = np.round(MF, 2)

    results = []
    distances = []

    for mf in MF:
        scores = MoSS(1000, 0.5, mf)
        test_p = scores[scores[:, 2] == 1, 0]
        test_n = scores[scores[:, 2] == 2, 0]

        rQnt = HDy(test_p, test_n, ts)

        distances.append(rQnt[1])
        results.append(rQnt[0][0])

    best_result = round(results[np.argmin(distances)], 2)
    return [np.array([best_result, 1 - best_result]), min(distances), MF[np.argmin(distances)]]


# Main
def exec_eval_complexity_single(mi, MFtr, MF_dysyn):
    vdist = {"TS": "topsoe", "JD": "jensen_difference", "PS": "prob_symm", "ORD": "ord", "SORD": "sord", "TN": "taneja", "HD": "hellinger"}
    var_perc = np.arange(0, 1.01, 0.01)
    var_size = [100]
    n_tests = 10
    MF = np.arange(0.05, 1.0, 0.05)
    MF = np.round(MF, 2)
    qnt = ['CC', 'ACC', 'ACCSyn-TS', 'T50', 'T50Syn-TS', 'PCC', 'PACC', 'PACCSyn-TS', 'X', 'XSyn-TS', 'MAX', 'MAXSyn-TS', 'MS', 'MSSyn-TS', 'MS2', 'MS2Syn-TS', 'DyS-TS', 'DySyn-TS', 'SMM', 'SMMSyn-TS', 'HDy', 'HDySyn']
    results = []

    scores = MoSS(2000, 0.5, MFtr[mi])
    TprFpr = np.array(getTPRandFPRbyThreshold(scores)).astype(float)

    position = mi % 19
    description = f'MFtr: {MFtr[mi]}'

    for k in range(len(var_size)):
        for i in tqdm(range(len(var_perc)), desc=description, position=position, leave=True):
            for j in range(n_tests):
                for ti in range(len(MF)):
                    for qi in qnt:
                        test_set = MoSS(var_size[k], var_perc[i], MF[ti])
                        freq_REAL = pd.Series(test_set[:, 2]).value_counts(normalize=True).reindex([1, 2], fill_value=0)
                        qntMethod = qi.split("-")[0] if "-" in qi else qi

                        if qntMethod != "HDy-LP":
                            try:
                                nk = int(qntMethod.split("_")[0])
                            except ValueError:
                                nk = 1
                            qntMethod = "DySyn" if nk != 1 else qntMethod

                        measure = None
                        if len(qi.split("-")) > 1:
                            measure = vdist.get(qi.split("-")[1])

                        qnt_re = apply_qntMethod(
                            qntMethod=qntMethod,
                            p_score=scores[scores[:, 2] == 1, 0],
                            n_score=scores[scores[:, 2] == 2, 0],
                            test=test_set[:, 0],
                            TprFpr=TprFpr,
                            thr=0.5,
                            measure=measure,
                            MF_dysyn=MF_dysyn,
                        )

                        if qntMethod in ["DySyn", "HDy", "HDySyn"]:
                            freq_PRE = np.round(qnt_re[0][0], 3)
                        else:
                            freq_PRE = np.round(qnt_re[0], 3)

                        results.append([
                            MFtr[mi],
                            MF[ti],
                            freq_REAL.get(1, 0),
                            freq_PRE,
                            np.round(abs(freq_REAL.get(1, 0) - freq_PRE), 2),
                            measure,
                            qi,
                        ])
    return results


def worker(mi_MFtr_MFdysyn):
    mi, MFtr, MF_dysyn = mi_MFtr_MFdysyn

    results = exec_eval_complexity_single(mi, MFtr, MF_dysyn)

    return results

def exec_eval_complexity_parallel(MFtr, MF_dysyn):
    tasks = [(mi, MFtr, MF_dysyn) for mi in range(len(MFtr))]

    file_name = "./results_syn.csv"
    if not os.path.exists(file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        pd.DataFrame(columns=["MFtr", "MFte", "R_1", "P_1", "AE", "Distance", "Qnt"]).to_csv(file_name, index=False)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = list(pool.map(worker, tasks))

    flat_results = [item for sublist in results for item in sublist]
    df = pd.DataFrame(flat_results, columns=["MFtr", "MFte", "R_1", "P_1", "AE", "Distance", "Qnt"])
    df.to_csv(file_name, mode="a", header=False, index=False)

    print("All workers finished.")

if __name__ == "__main__":
    m_Tr = np.arange(0.05, 1.0, 0.05)
    m_Tr = np.round(m_Tr, 2)
    MF_dysyn = np.arange(0.1, 1.0, 0.2)

    exec_eval_complexity_parallel(m_Tr, MF_dysyn)

    print("Synthetic Experiment complete!")