# Temas transversales (previos al modelado)

## 1) Desbalanceo
- Estrategias: `class_weight`, under/over-sampling, **SMOTE/ADASYN**, *focal loss* (NN/boosting).
- Riesgos: overfitting por oversampling, fuga de info si se hace antes del split o fuera del CV.

## 2) Calibración
- **Platt scaling**, **Isotónica**, *temperature scaling*. Métricas: **Brier**, curvas de calibración.

## 3) Selección de umbral
- Umbral fijo vs. dependiente de **costos**/**prevalencia**. Elegir por F1, J(Youden), utilidad esperada.

## 4) Métricas (puente)
- ROC-AUC vs PR-AUC, F1/Fβ, MCC, KS, Log-loss, Brier.  
Para detalle matemático ver `docs/99-evaluation-metrics.md`.
