# Métricas de evaluación (detalle)

- **ROC-AUC**: prob. de ordenar posit. > negat.; estable con balance, engañosa con clase rara.
- **PR-AUC**: precisión-recobrado; preferir con clase positiva rara.
- **F1 / Fβ**: armónica; útil con clases desbalanceadas cuando importa balance P/R.
- **MCC**: correla. de confusión; robusta con desbalance.
- **Log-loss**: evalúa probas; sensible a mala calibración.
- **Brier**: error cuadrático de probas; útil para calibración.
- **KS**: separación de distribuciones de score.
- **Curvas**: ROC, PR y **calibration/reliability**.
- **Umbral óptimo**: depende de la métrica objetivo o matriz de costos.
