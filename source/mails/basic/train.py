"""Maile, model prosty: TF-IDF (słowa 1-2 + znaki 3-5) + LogisticRegression.
Baseline: ComplementNB, LinearSVC z kalibracją.
Uruchom: python -m source.mails.basic.train
"""
import time
from itertools import product

import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from source.common.config import LABEL_COL, SEED
from source.common.metrics import best_f1_threshold, evaluate
from source.common.results import save_results
from source.common.data import load_emails
from source.common.splits import stratified_split

MAX_CHARS = 3000  # przycięcie bardzo długich maili (szybkość n-gramów znakowych)
VOCAB_SIZES = [50_000, 200_000]
C_GRID = [0.3, 1, 10]


def vectorize(train_text, others, vocab_size):
    """Słownik dopasowany wyłącznie na zbiorze treningowym."""
    word = TfidfVectorizer(ngram_range=(1, 2), max_features=vocab_size, sublinear_tf=True, min_df=2)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=vocab_size,
                           sublinear_tf=True, min_df=3)
    x_train = sp.hstack([word.fit_transform(train_text), char.fit_transform(train_text)]).tocsr()
    xs = [sp.hstack([word.transform(t), char.transform(t)]).tocsr() for t in others]
    return x_train, xs


def log(msg):
    print(msg, flush=True)


def scores_of(model, x):
    return model.predict_proba(x)[:, 1]


def main():
    log("[1/5] Wczytywanie bazy maili i podział train/val/test...")
    train, val, test = stratified_split(load_emails())
    log(f"      train={len(train)} val={len(val)} test={len(test)}")
    text = {k: d["text"].str[:MAX_CHARS] for k, d in dict(train=train, val=val, test=test).items()}
    y_train, y_val, y_test = (d[LABEL_COL].values for d in (train, val, test))

    # strojenie: rozmiar słownika x siła regularyzacji, kryterium: ROC-AUC na walidacji
    best = None
    cache = {}
    log("[2/5] Strojenie LogisticRegression (słownik x C) na walidacji...")
    for vocab in VOCAB_SIZES:
        log(f"      TF-IDF (słowa 1-2 + znaki 3-5), słownik={vocab} (to może potrwać)...")
        x_tr, (x_va, x_te) = vectorize(text["train"], [text["val"], text["test"]], vocab)
        cache[vocab] = (x_tr, x_va, x_te)
        for c in C_GRID:
            lr = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, solver="liblinear")
            lr.fit(x_tr, y_train)
            auc = roc_auc_score(y_val, scores_of(lr, x_va))
            log(f"      LR vocab={vocab} C={c}: val AUC={auc:.5f}")
            if best is None or auc > best[0]:
                best = (auc, vocab, c)
    _, vocab, c = best
    log(f"[3/5] Najlepsze parametry: vocab={vocab}, C={c}")
    x_tr, x_va, x_te = cache[vocab]

    models = {
        "LogisticRegression": LogisticRegression(C=c, class_weight="balanced", max_iter=2000, solver="liblinear"),
        "ComplementNB": ComplementNB(),
        "LinearSVC_calibrated": CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3),
    }
    rows = []
    log("[4/5] Trening modeli, dobór progu na walidacji, ocena na teście...")
    for name, model in models.items():
        log(f"      Trening: {name}...")
        t0 = time.time()
        model.fit(x_tr, y_train)
        fit_s = time.time() - t0
        thr = best_f1_threshold(y_val, scores_of(model, x_va))  # próg z walidacji
        t0 = time.time()
        s_test = scores_of(model, x_te)
        pred_s = time.time() - t0
        res = evaluate(y_test, s_test, thr)
        res.update(task="mail", model=name, threshold=thr, roc_auc=roc_auc_score(y_test, s_test),
                   fit_seconds=round(fit_s, 2), predict_seconds=round(pred_s, 2), seed=SEED)
        rows.append(res)
        log(f"      {name}: F1={res['f1']:.4f} FPR={res['fpr']:.4f} FNR={res['fnr']:.4f} "
            f"próg={thr:.3f} fit={fit_s:.1f}s")
    log("[5/5] Zapis wyników do results/mails_basic.csv")
    save_results("mails_basic", rows)
    log("Gotowe.")


if __name__ == "__main__":
    main()
