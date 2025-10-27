# Classification
Everything about classification

---------------------------------------------

**Repository summary**

1.  **Intro** 🧳


2.  **Tech Stack** 🤖


3.  **Features** 🤳🏽


4.  **Process** 👣

5.  **Learning** 💡

6.  **Improvement** 🔩

7.  **Running the Project** ⚙️

8 .  **More** 🙌🏽




---------------------------------------------

# :computer: Everything about Classification  :computer:

---------------------------------------------

# 1. Let's talk about Classification.

# I. Introduction: What is Classification?

Tipos de clasificación

# II. Core Components of classication analysis.

# III.  Temas transversales


#

#

#




-------------------------------------------------

2) Temas transversales (previos a los modelos)

Propósito: antes de estimar modelos, entender aspectos que cambian decisiones y métricas.

Desbalanceo de clases

Estrategias: class_weight, re-muestreo (Under/Over), SMOTE/ADASYN, focal loss (para NN/boosting).

Cuándo aplicarlas y riesgos (overfitting por oversampling, fuga de información en CV, etc.).

Calibración de probabilidades

Platt scaling, Isotónica, temperature scaling (NN). Evaluar con Brier y curvas de confiabilidad.

Selección de umbral

Umbral fijo vs. dependiente de costos / prevalencia. Maximizar F1, J (Youden), utilidad esperada.

Métricas de evaluación (resumen)

ROC‑AUC vs PR‑AUC (preferir PR‑AUC con clase rara), F1/Fβ, MCC, KS, Log‑loss, Brier.

Curvas ROC/PR y calibración; reporte por clase; matrices de costos cuando aplique.

Estos conceptos aparecerán explícitamente en cada técnica (sección “Evaluación adecuada” y “Buenas prácticas”).

3) Taxonomía (qué cubriremos)

Lineales probabilísticos

Regresión Logística (binaria y multiclase OvR/OvO)

LDA/QDA (Análisis Discriminante Lineal/Cuadrático)

Naive Bayes (Gaussiano/Multinomial/Bernoulli)

Márgenes y planos

Perceptrón

SVM (lineal y kernel: RBF, polinomial)

Basados en instancias

k‑NN

Árboles

Decision Trees (CART, Gini/Entropía)

Ensamblajes

Bagging, Random Forest

AdaBoost, Gradient Boosting

XGBoost, LightGBM, CatBoost

Redes Neuronales (≥ 3 técnicas de clasificación)

MLP (feed‑forward) para tabular/texto vectorizado

CNN simple para imágenes (clasificación básica)

RNN/LSTM/GRU para secuencias (p.ej., texto/tiempo) con salida de clase

4) Plantilla común por técnica (doc)

Cada archivo en docs/ seguirá esta estructura uniforme:

¿Qué es? (definición en 3–5 líneas)

¿Para qué sirve? (casos de uso típicos)

Intuición (figura/visión geométrica o probabilística)

Fundamento matemático (expresión/función de pérdida & decisión)

Algoritmo de entrenamiento (pasos o pseudo‑código)

Supuestos & condiciones (cuando el modelo se comporta mejor)

Hiperparámetros clave (tabla con significado e impacto)

Complejidad (tiempo/espacio, n vs d)

Buenas prácticas (escalado, regularización, validación)

Pitfalls comunes (leakage, overfitting, multicolinealidad, etc.)

Implementación en librerías

Python: librería y clase/función (sklearn.linear_model.LogisticRegression, xgboost.XGBClassifier, lightgbm.LGBMClassifier, catboost.CatBoostClassifier, torch.nn.Module/keras.Model, etc.) + parámetros relevantes y su efecto.

R (opcional): glm, MASS::lda/qda, e1071::svm, randomForest, xgboost, lightgbm, catboost, keras.

Código mínimo (Python y/o R, dataset sintético + métrica principal)

Cuándo SÍ usarla (idónea) y cuándo NO usarla (anti‑patrones)

Referencias

Papers (3): trabajos canónicos/seminales.

Web (2): documentación oficial o tutoriales reputados.

CASO DE LA REGRESIÓN LOGISTICA


6) Métricas (resumen de docs/99-evaluation-metrics.md)

ROC‑AUC vs PR‑AUC (preferir PR‑AUC con clase positiva rara)

F1 / Fβ; MCC; Brier; KS; Log‑loss

Curvas: ROC, PR, calibración; umbral óptimo por métrica/escenario de costos


