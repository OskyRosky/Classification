# Regresión Logística

**¿Qué es?** Clasificador lineal probabilístico: \( P(y=1\mid x)=\sigma(w^\top x+b) \).

**Fundamento**: minimizar log-loss con regularización. Decisión por umbral \(\tau\).

**Implementación (Python)**
- `sklearn.linear_model.LogisticRegression`
  - Parámetros: `C`, `penalty`, `solver`, `class_weight`, `max_iter`, `multi_class`, `fit_intercept`.

**Código mínimo**
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, average_precision_score

X, y = make_classification(n_samples=2000, n_features=20, weights=[0.8,0.2], random_state=42)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25, stratify=y, random_state=42)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
clf.fit(Xtr,ytr)
proba = clf.predict_proba(Xte)[:,1]
print("ROC-AUC=", roc_auc_score(yte, proba))
print("PR-AUC =", average_precision_score(yte, proba))

**Cuándo SÍ / NO usar**  
- SÍ: interpretabilidad, baseline fuerte, datos tabulares.  
- NO: fronteras altamente no lineales sin features adecuadas.

**Referencias**  
- Papers: Cox (1958), McCullagh & Nelder (1989), Friedman-Hastie-Tibshirani (2001, *ESL*).  
- Web: scikit-learn `LogisticRegression`, StatQuest.
