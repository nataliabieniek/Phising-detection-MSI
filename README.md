# Detekcja phishingu

## Środowisko (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Maile – TF-IDF + LogisticRegression (`source/mails/basic/train.py`)

Jedna komenda z katalogu głównego repo (sama wczytuje dane z `Dane/emails.csv.gz`, dzieli na train/val/test, trenuje i ocenia):

```powershell
python -m source.mails.basic.train
```

Co robi: łączy temat z treścią → TF-IDF (słowa 1–2 + znaki 3–5, słownik tylko z train) → LogisticRegression
z `class_weight=balanced` → baseline ComplementNB i LinearSVC z kalibracją → strojenie `C` i rozmiaru słownika
na walidacji → próg (max F1) wybrany na walidacji → ocena na teście.
Wyniki zapisują się do `results/mails_basic.csv` (kolejne uruchomienie nadpisuje; trwa ok. 2–3 min).

### Wyniki (zbiór testowy, próg z walidacji)

Najlepsze parametry (walidacja, kryterium ROC-AUC): słownik 200 000, `C=10` (AUC walidacji 0,99975).

| Model | F1 | FPR | FNR | Próg |
|---|---|---|---|---|
| LogisticRegression (model główny) | 0,9936 | 0,94% | 0,41% | 0,47 |
| ComplementNB | 0,9829 | 1,99% | 1,58% | 0,06 |
| LinearSVC (z kalibracją) | **0,9939** | 0,92% | 0,38% | 0,40 |

Najlepszy F1 ma LinearSVC (0,9939), ale różnica względem LogisticRegression to 0,0003, więc w praktyce są równorzędne.
Wyniki są bardzo wysokie, bo podział jest losowy w obrębie tych samych korpusów; wiarygodniejszy będzie test cross-source.

### Jak wyznaczany jest próg i najlepszy F1

Funkcja `best_f1_threshold` (`source/common/metrics.py`):
1. Model jest trenowany na zbiorze treningowym i zwraca prawdopodobieństwo phishingu dla każdego maila walidacyjnego.
2. Kandydaci na próg: 197 wartości z kwantyli 1%–99% wyników walidacyjnych.
3. Dla każdego kandydata maile z wynikiem ≥ próg są klasyfikowane jako phishing i liczone jest F1 na walidacji.
4. Wybierany jest próg z najwyższym F1 na walidacji (`argmax`).
5. Próg jest zamrażany i użyty jeden raz na zbiorze testowym. Raportowane F1 to F1 testowe przy tym progu.
   Próg nigdy nie jest dobierany na teście, żeby wynik nie był zawyżony.

Wzór: F1 = 2·precision·recall / (precision + recall). Model „najlepszy" wybieramy porównując F1 testowe przy progach z walidacji.

### Opis metryk (kolumny w `results/mails_basic.csv`)

| Metryka | Znaczenie |
|---|---|
| `accuracy` | Odsetek poprawnie sklasyfikowanych maili (phishing i legit razem). |
| `f1` | Średnia harmoniczna precision i recall; balansuje wykrywanie phishingu i fałszywe alarmy. |
| `recall` | Jaka część prawdziwego phishingu została wykryta (TP / (TP + FN)). |
| `fpr` | False Positive Rate: odsetek legalnych maili błędnie uznanych za phishing (FP / (FP + TN)). |
| `fnr` | False Negative Rate: odsetek phishingu przeoczonego (FN / (FN + TP)) = 1 − recall. |
| `fpr_at_recall95` | FPR przy takim progu, że recall ≥ 95%; im niżej, tym lepiej (niezależne od wybranego progu). |
| `fnr_at_fpr1` | FNR przy takim progu, że FPR ≤ 1%; im niżej, tym lepiej (niezależne od wybranego progu). |
| `task` | Zadanie (`mail`). |
| `model` | Nazwa modelu. |
| `threshold` | Próg decyzyjny wybrany na walidacji (max F1); wynik ≥ próg = phishing. |
| `roc_auc` | Pole pod krzywą ROC: jakość rankingowania niezależnie od progu (1,0 = idealnie, 0,5 = losowo). |
| `fit_seconds` | Czas treningu modelu (koszt obliczeniowy). |
| `predict_seconds` | Czas predykcji na zbiorze testowym. |
| `seed` | Ziarno losowości (powtarzalność podziału). |

Kod wspólny: `source/common/` (`config.py`, `data.py`, `splits.py`, `metrics.py`, `results.py`).
