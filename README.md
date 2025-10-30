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

Classification is everywhere: even when we don’t notice it.  
Every day, dozens of systems quietly classify information around us:  
Netflix decides what you might like next.  
Gmail separates spam from important messages.  
Spotify predicts the next song that fits your mood.  
A bank model decides whether to approve your credit card.  

These are not coincidences. They are **classification models** — algorithms that learn from data to assign categories, make predictions, and help humans make faster, more consistent decisions.

In the modern world, classification is the foundation of intelligent applications.  
It powers everything from voice assistants and medical diagnosis systems to fraud detection, recommendation engines, and autonomous vehicles.  

This repository is dedicated to understanding these models — from the simplest logistic regression to the most advanced neural networks — explaining not only how they work, but also when and why to use them.

Let’s explore how machines learn to **draw boundaries, make decisions, and see patterns in data**.


 # I. Introduction: what is Classification?

Classification stands as one of the pillars of modern machine learning. At its essence, it is the act of making a decision — determining to which category or class a given observation belongs. When an email service filters a message as spam, when a hospital system predicts whether a patient is at high or low risk, or when a smartphone recognizes a face to unlock the screen, it is performing a classification task. In each of these situations, a model transforms patterns in the data into meaningful, structured outcomes that help automate everyday decisions.

At its core, classification seeks to answer a timeless question: *Given what I have learned from the past, how should I categorize what I see now?*  
This question sits at the heart of predictive intelligence, bridging the gap between observation and action.

### 1. Definition and Essence.

In the language of data science, classification is a supervised learning task in which a model learns from labeled data how to assign new observations to predefined categories. During training, the algorithm is exposed to examples that contain both the input features — the measurable characteristics of each case — and their corresponding labels — the known outcomes. Over time, it captures patterns that describe the relationship between inputs and outputs. When confronted with unseen data, it predicts the most likely class by applying what it has learned.

Unlike regression, which estimates continuous quantities, classification produces discrete outcomes. These outcomes represent states, events, or entities that are distinct and mutually exclusive. This fundamental difference shapes every aspect of modeling: the type of loss functions we use, the interpretation of accuracy, and the criteria we employ to evaluate success. While regression answers “how much,” classification answers “which one.”

### 2. Domains and Data Types.

The power of classification lies in its versatility. It extends far beyond spreadsheets or tabular records to virtually every form of information that can be captured and quantified. In text, it identifies whether a tweet expresses positive or negative sentiment, or whether an email belongs in the spam folder. In images, it distinguishes between cats, dogs, and countless objects of daily life, supporting everything from medical diagnostics to autonomous vehicles. In audio, it allows a voice assistant to recognize who is speaking and what is being said. Even in industrial systems, classification monitors sensor data to detect failures or anomalies before they occur.

If data can be represented numerically — whether as words, pixels, or waveforms — it can be classified. This universality explains why classification has become one of the cornerstones of artificial intelligence. It is the common language that connects different data modalities under a single analytical goal: the pursuit of structured understanding.

### 3. Types of Classification problems.

Not all classification problems are created equal. Some are simple, others profoundly complex. The most basic form, binary classification, distinguishes between two possible outcomes: an email is either spam or not spam; a transaction is fraudulent or legitimate. Multiclass problems expand the scope by allowing more than two categories, as when a model identifies the species of a flower or the digit drawn on a touchscreen. More advanced yet are multilabel problems, in which a single instance can belong to several categories simultaneously — a news article that fits both “politics” and “economy,” for example. At the highest level of abstraction lies hierarchical classification, where classes follow a structured taxonomy, much like the biological hierarchy that organizes species into genera and families.

Understanding which kind of problem one faces is not merely academic. It defines how the data must be prepared, how the algorithm must be chosen, and how the results must be measured. The type of classification shapes the entire analytical journey.

### 4. The spectrum of complexity.

The world of classification algorithms forms a continuum that ranges from elegant simplicity to profound complexity. At one end are interpretable, mathematically transparent models such as Logistic Regression or Linear Discriminant Analysis. These methods rely on linear relationships and serve as powerful baselines, offering insight into how features contribute to decisions. In the middle of the spectrum appear models like Decision Trees, Random Forests, and Support Vector Machines, which are capable of capturing nonlinear interactions and subtle dependencies among variables. At the farthest end, deep neural networks extend classification into unstructured domains — images, text, sound — by learning multiple layers of abstract representation.

This gradual increase in complexity reflects the evolution of classification itself. It has grown from the study of simple linear boundaries into a vast family of methods capable of modeling the intricacies of human perception and reasoning. The same mathematical principle that once separated points on a plane now enables machines to distinguish a pedestrian from a shadow, or a heartbeat from an anomaly.

### 5. Why It Matters?

To understand classification is to understand the foundation of intelligent systems. Almost every predictive model that informs a decision — whether it approves a loan, recommends a movie, or detects a security threat — depends on some form of classification. Knowing how these models work allows data scientists not only to build accurate predictors but also to interpret them responsibly.

Classification teaches us to separate signal from noise, to recognize uncertainty as part of knowledge, and to express probability as a measure of belief rather than certainty. It enables practitioners to calibrate confidence, quantify risk, and explain outcomes in a language that both machines and humans can understand. Its importance extends beyond technical accuracy: it brings rigor and transparency to the very process of decision-making in the digital age.

In disciplines such as healthcare or finance, this understanding becomes ethical as well as practical. A misclassified tumor or a misjudged credit risk can have real consequences. For that reason, mastering classification means mastering both the science and the responsibility behind automated decisions.

### 6. From models to meaning: the human connection.

Every classification model is, in a sense, a reflection of human reasoning. When a doctor recognizes a disease from symptoms, when a teacher evaluates an essay, or when a driver decides whether to stop or continue, they are performing a classification task — grouping observations into categories based on experience and context. Machine learning formalizes this intuition through algorithms and data, allowing it to scale beyond individual judgment to millions of decisions per second. Yet, behind every model, the essence remains human: the desire to understand, to predict, and to act consistently upon patterns in the world.

----

To study classification deeply, one must go beyond algorithms and consider the ingredients that make them reliable. Before building models, we must understand the forces that shape their behavior — how class imbalance distorts performance, how probabilities lose calibration, and how the choice of thresholds and metrics defines what “success” truly means. These foundational ideas are the **core components of classification analysis**, and they set the stage for every method that follows.

----

# II. 🧩 Core Components of classication analysis. 🧩

Classification becomes meaningful when we treat it as a complete process, not just a model choice. A robust analysis begins with a precise understanding of the decision we want to support, continues with careful treatment of data and representation, and matures through thoughtful training, evaluation, interpretation, and monitoring. This section describes that end-to-end path. The goal is simple and ambitious at the same time: build models that perform well, explain themselves clearly, and withstand real-world conditions.

## 1. Understanding the problem and the data.

A solid project starts when the team defines the question with care. The analyst states the decision that the model will support, clarifies who will use the prediction, and documents the consequence of being wrong. A hospital wants to flag patients at high risk, a bank wants to screen transactions for fraud, and a public agency wants to prioritize inspections. Each case carries a different cost for false positives and false negatives, and each cost changes how we should evaluate success.

The analyst also identifies the type of classification at stake. A binary task separates two outcomes, a multiclass task assigns one among many labels, and a multilabel task allows several labels at once. A hierarchical task arranges labels in levels and requires consistent decisions across the hierarchy. The team then audits the sources of data, including tabular records, documents, images, audio, or sensor streams, and writes a short data card that lists provenance, refresh cycles, and known limitations. This early discipline prevents confusion later and aligns the modeling effort with the decisions that matter.

## 2. Data preparation and cleaning.

Clean data creates reliable learning. The analyst handles missing values in a way that respects the data generating process, detects outliers that break assumptions, and fixes inconsistencies that arise when systems merge. Categorical variables receive encodings that preserve information content, numerical variables receive scaling when algorithms require it, and time fields receive careful parsing to avoid hidden leakage. The team draws a clear boundary between training data and future data, because leakage appears the moment information from the future slips into the past. The final step splits the dataset into training, validation, and test partitions, or establishes cross-validation folds that respect grouping or time order. This structure keeps the estimate of generalization honest.

## 3. Feature engineering and representation.

Features shape what a model can learn. The analyst begins with a minimal set that captures the essence of the domain and then improves representation with transformations and interactions. A credit model benefits from ratios that express behavior across time, a medical model benefits from trends rather than single snapshots, and a text model benefits from embeddings that preserve semantics. The team removes redundant or perfectly collinear features when they add noise, and applies dimensionality reduction when the number of variables grows faster than the number of examples. The choice of representation influences linear and nonlinear models in different ways. A linear model rewards informative transformations, while a tree ensemble tolerates raw forms but still benefits from well-designed features. Good representation reduces variance, clarifies signal, and lowers the burden on the learning algorithm.

## 4. Model selection and training.

Model selection follows from problem structure, data regime, and interpretability needs. A team chooses logistic regression or linear discriminant analysis when the relationship appears close to linear and when stakeholders demand transparent coefficients. A team chooses decision trees, random forests, or gradient boosted trees when relationships appear nonlinear and tabular data dominates. A team chooses support vector machines when margins matter and the kernel trick unlocks structure. A team chooses neural networks when images, audio, or language require hierarchical representation.

Training becomes a controlled experiment. The analyst defines an objective function that matches the task, tunes hyperparameters with a method that balances thoroughness and compute budget, and applies regularization to prevent overfitting. The team handles class imbalance with class weights or resampling and reserves synthetic methods for cases where the signal does not reach the minority class without help. The training loop tracks convergence, records seeds for reproducibility, and logs configurations for future audits. A small set of baselines anchors the work, because a strong baseline protects the team from chasing noise.

## 5. Validation and evaluation.

Evaluation turns predictions into judgment. The analyst selects metrics that match the decision, not the dataset convenience. A rare event demands precision and recall, while a balanced screening task tolerates accuracy as a summary. Receiver operating characteristic curves describe ranking power across thresholds, while precision–recall curves reveal performance where positive cases are scarce. The team reads confusion matrices per class and studies error types with the same attention a clinician gives to symptoms. Threshold analysis appears as a natural step, because business value rarely aligns with a fixed probability of 0.5. Calibration closes the loop by aligning predicted probabilities with observed frequencies. A calibrated model gives a reliable measure of confidence, which becomes essential when humans use scores to plan actions.

Cross-validation estimates generalization when data are limited, and temporal splits protect time-dependent structure when events evolve. Robustness checks matter as much as headline scores. The team perturbs inputs, repeats runs with different random seeds, and measures variance across folds. The analyst prefers a model that performs slightly worse on average but more consistently across time and subgroups, because stability sustains trust.

## 6. Interpretation and explainability.

A model serves people when it explains itself. The analyst studies which features drive predictions and which interactions change decisions at the margin. Global importance summarizes the broad picture, while local explanations illuminate individual cases. Tools such as permutation importance, SHAP values, and counterfactual analysis provide complementary views, yet none replaces domain sense. The team validates explanations with subject matter experts and documents how limitations might affect use. The analyst quantifies uncertainty, communicates it with clarity, and avoids false certainty, because confidence without calibration misleads more than it helps. Interpretation does not exist to decorate reports. It exists to support responsible action.

## 7. Deployment and monitoring.

A model becomes real when it leaves the notebook. The team exports artifacts in a portable format, wraps them in an interface that other systems can call, and writes a small contract that states input schema, output schema, and expected latencies. Monitoring begins on day one. The analyst tracks predictive performance and watches for data drift and concept drift, because the world changes slowly and then all at once. When drift appears, the team investigates whether feature distributions moved, whether labels shifted, or whether behavior changed in ways the original training could not anticipate. A feedback loop allows the model to learn again, but only after the team evaluates risks and documents the change.

## 8. Ethical and responsible AI considerations.

A classification system touches people, therefore it carries responsibility. The analyst inspects data for biases that hurt protected groups, measures fairness with metrics that reflect the institution’s values, and records trade-offs that appear when fairness constraints meet accuracy goals. Transparency matters, because users deserve to understand how a system reaches conclusions that affect them. Accountability matters, because a team should be able to trace a decision back to data, code, and configuration. Responsible practice does not slow progress. Responsible practice makes progress sustainable.

## 9. From process to practice.

The classification pipeline behaves like a cycle rather than a line. The analyst learns from errors, revisits data preparation, simplifies features, and resets baselines when the signal does not support complexity. Documentation becomes a habit rather than a chore, because future readers will depend on the record to reproduce results and extend the work. The team treats experiments as evidence, not as decoration, and accepts that small improvements rarely justify fragile systems. Over time, this discipline produces models that travel well from lab to production and remain useful after the initial excitement fades.

---

Before we introduce individual algorithms, we should master the forces that quietly govern their behavior. Class imbalance changes what success looks like, calibration reshapes the meaning of probability, and threshold selection turns ranked scores into real decisions. Metrics complete the picture by defining value in a way the project can defend. These transversal topics represent the grammar of classification. Once we speak that grammar with confidence, every method we study will become clearer, more comparable, and easier to deploy with integrity.

---

# III.  Cross-cutting Topics (Core Analytical Foundations).

Classification is not only about choosing an algorithm.
Before any model can make a fair, reliable, and interpretable decision, we must understand the invisible forces that govern its behavior. These forces—class imbalance, probability calibration, threshold selection, and evaluation metrics—shape how models learn, predict, and fail. They are called cross-cutting topics because they apply to every classifier, regardless of its mathematical form.
Ignoring them leads to models that perform well on paper but fail in reality; mastering them turns simple models into trustworthy tools.

## 1. Introduction: Why Cross-cutting Topics Matter

Every classification model learns patterns, but those patterns emerge inside a context defined by the data and by the decisions we expect to make. These cross-cutting topics form that context. They determine how models interpret uncertainty, how predictions translate into actions, and how performance should be measured in a meaningful way.

For example, a bank that predicts default risk does not simply want to maximize accuracy—it wants to minimize losses while treating clients fairly. A hospital that predicts disease risk must choose a threshold that saves lives without overwhelming resources. In both cases, the success of the model depends less on the algorithm itself and more on how we handle imbalance, probabilities, and evaluation.

Understanding these foundations allows data scientists to move beyond superficial metrics. It builds the capacity to reason statistically, ethically, and operationally at the same time.

## 2. Class Imbalance

Most real-world classification problems are unbalanced. Fraud cases are rare compared to normal transactions, diseases occur in a small fraction of patients, and positive reviews may outnumber negative ones. When the majority class dominates, models learn to favor it, achieving high accuracy while failing to detect the minority class—the one that often matters most.

This imbalance distorts the decision boundary. A model that predicts “no fraud” 99 % of the time might still reach 99 % accuracy, yet it becomes useless when the goal is to detect fraud. The key lies in restoring balance, not by forcing equal counts, but by giving the minority class the attention it deserves.

There are three complementary strategies.
First, algorithmic adjustments, such as using class weights or cost-sensitive learning, tell the model that certain errors are more expensive than others.
Second, data-level techniques, like under-sampling the majority or over-sampling the minority, provide more balanced training examples. Synthetic methods like SMOTE (Synthetic Minority Over-sampling Technique) and ADASYN create plausible new observations to enrich scarce categories.
Third, evaluation-level corrections, such as preferring precision–recall curves or F1-scores instead of plain accuracy, ensure that metrics reflect real performance.

The right choice depends on context. A hospital might tolerate more false alarms if it means catching every positive case. A bank might accept a few missed frauds if it avoids wrongly flagging honest clients. The lesson is simple: imbalance changes what success looks like, and every responsible analyst must recognize that early.

## 3. Probability Calibration

A well-calibrated model not only predicts correctly but also tells the truth about its confidence.
If a model says, “there is a 70 % chance of rain,” it should rain on about 7 out of 10 such days. This alignment between predicted probabilities and observed outcomes defines calibration.

Many algorithms, however, produce distorted probabilities. Decision trees and boosting methods tend to output overconfident scores because they optimize separation rather than probability estimation. Neural networks can also become miscalibrated when trained with aggressive regularization or imbalanced data.

To fix this, analysts apply post-processing calibration techniques.
The most common are Platt scaling, which fits a logistic regression on the model’s outputs; Isotonic regression, which uses nonparametric fitting for greater flexibility; and Temperature scaling, a simple yet effective approach for neural networks.

To evaluate calibration, we can use the Brier score, which measures the mean squared difference between predicted and actual outcomes, or reliability diagrams, which visualize how predicted probabilities align with observed frequencies.

In domains like credit scoring or medical diagnosis, calibration equals trust. A miscalibrated model might assign false confidence to risky loans or overlook uncertain diagnoses. Correcting that bias ensures that probability means probability, not just a score.

## 4. Threshold Selection

Every classifier ultimately faces a single, deceptively simple question:
At what point should we say “yes”?

The conventional threshold of 0.5 is a convenience, not a law of nature.
Different problems require different cutoffs. In a medical triage system, setting a lower threshold may catch more potential patients at risk but also generate more false alarms. In a credit approval system, a higher threshold may reduce default rates but deny opportunities to worthy clients.

Threshold selection transforms continuous model outputs into categorical decisions. It defines the balance between precision (how often we are right when we say “yes”) and recall (how often we find all true positives). Choosing a threshold thus becomes a moral, economic, and operational decision, not just a statistical one.

Analysts use several strategies to find optimal thresholds. The Youden J statistic maximizes the distance between the true positive rate and the false positive rate. The F1-score seeks a harmonic balance between precision and recall. Cost-based thresholds incorporate explicit penalties for different types of errors.
Each method reveals a different aspect of the trade-off.

By treating threshold selection as an intentional act rather than an afterthought, we make models actionable and context-aware. A well-chosen threshold transforms a model from a calculator into a decision aid.

## 5. Evaluation Metrics

Metrics define what “good” means in machine learning.
A model is only as good as the measure we choose to judge it. Selecting the wrong metric can reward the wrong behavior.

For balanced problems, accuracy might suffice, but in unbalanced scenarios, it hides the truth. Analysts instead rely on a suite of metrics that capture performance from multiple angles.
Ranking metrics, such as ROC-AUC (Area Under the Receiver Operating Characteristic Curve) and PR-AUC (Area Under the Precision–Recall Curve), evaluate how well the model ranks positives over negatives.
Class-level metrics, such as precision, recall, and the F1-score, focus on correctness within each class. MCC (Matthews Correlation Coefficient) summarizes overall consistency between predictions and reality.
Probabilistic metrics, such as Log-loss and the Brier score, evaluate not only correctness but also confidence.

These metrics complement each other. A model with high recall but low precision may be useful in emergency screening; a model with high precision but moderate recall may suit fraud prevention. Analysts must also consider calibration curves, confusion matrices, and threshold analysis to fully understand behavior.

Metrics are not neutral—they encode priorities and ethics. Choosing F1 over accuracy, or PR-AUC over ROC-AUC, reflects a statement of values about what outcomes matter most. Understanding this dimension turns evaluation into a deliberate, ethical act.

## 6. Integrative Perspective

The beauty and complexity of classification lie in how these elements interact.
Imbalance changes the meaning of metrics. Calibration reshapes probability interpretation. Thresholds shift the trade-off between risk and reward. Together, they define the operational identity of a model.

Treating these concepts in isolation often leads to contradictions.
An analyst may fix imbalance but forget to recalibrate probabilities; another may adjust thresholds without reconsidering which metric to optimize. The integrative view recognizes that each adjustment influences the others.

Responsible practice, therefore, means iteration. The analyst builds, measures, adjusts, and re-evaluates until the system’s outputs align with both empirical accuracy and practical intent. A good model fits not only data but also purpose.

## 7. Implementing Cross-cutting Topics in Practice

After understanding the theory, the next step is applying it through code.
Modern programming languages provide robust ecosystems to operationalize these analytical foundations.

Python leads the field with libraries such as scikit-learn, which implements sampling strategies, calibration methods, and metric evaluation; imbalanced-learn, which adds advanced resampling algorithms like SMOTE and ADASYN; and visualization tools like matplotlib, seaborn, and plotly for diagnostic plots and calibration curves.
Other libraries, including xgboost, lightgbm, and catboost, incorporate internal mechanisms to manage imbalance and calibration directly during training.

In R, frameworks such as caret, mlr3, and tidymodels provide parallel functionality with a clear statistical foundation. They support cross-validation, weighting schemes, and interpretable evaluation pipelines, often favored by analysts in academia and healthcare.

These languages promote reproducibility and open experimentation. They bridge the analytical insights discussed here with the technical implementations that bring them to life, ensuring that theory becomes measurable, testable, and transparent.


---- 

Having explored the analytical foundations that influence every model, we are now ready to study the models themselves.
The next section presents the Taxonomy of Classification Models—a structured map of the main families of classifiers, from the simplicity of logistic regression to the expressive depth of neural networks.

With the knowledge of imbalance, calibration, thresholds, and metrics, we can now compare models not only by accuracy but by their fit to context, interpretability, and fairness.
This transition marks the moment when understanding becomes craftsmanship, and where classification evolves from an abstract concept into a concrete, reproducible system.

----

# IV. Taxonomy of Classification Models.

Classification is a vast landscape, and each model represents a different way of seeing the world. To classify is not merely to compute; it is to choose a philosophy of learning — a way to interpret data, uncertainty, and structure.
This taxonomy serves as a map through that landscape, tracing how ideas have evolved from simple linear equations to deep architectures capable of discovering meaning in text, vision, and sound.
Here, we focus on understanding the logic behind each family of models, not the formulas or code. We will explore how they were born, what they solve, and why they continue to matter.

## 1. Introduction to Model Taxonomy

Every classification model answers a slightly different question.
Linear and probabilistic models ask, “Can I draw a line that separates these points?” Margin-based models ask, “How far apart can I push them?” Instance-based models respond, “Whom do I resemble most?” Tree-based models inquire, “What sequence of questions best splits this space?” Ensembles reply, “What if we let many weak opinions vote?” And neural networks whisper, “Can the machine learn its own representation?”

This is the logic of progression — from the explicit to the emergent, from geometry to hierarchy.
Linear models begin with human-defined relationships; deep networks end with patterns no human could describe but that still mirror intuition.
Understanding this taxonomy is like standing on a mountain and watching the valleys of Machine Learning unfold below.
We begin where everything started — with linearity and probability.

## 2. Linear and Probabilistic Models

At the dawn of machine learning stood linearity — the belief that the world could be described by a weighted sum of inputs. Logistic Regression, Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), and Naive Bayes emerged from statistical thinking long before the term “machine learning” was coined.

Their philosophy is simple: assume structure, estimate parameters, and predict categories based on explicit mathematical relationships. Logistic regression models the log-odds of belonging to a class, while LDA and QDA model the underlying distributions of each class and the boundary where they meet. Naive Bayes, though based on a strong independence assumption, exploits the same probabilistic foundation.

These models remain powerful because they are transparent and stable. A logistic regression coefficient can be explained to a policymaker; an LDA score can be interpreted by a doctor. They thrive when data are limited and interpretability matters more than raw performance.

Yet their beauty is also their boundary. When relationships are nonlinear or highly interactive, linear separability collapses. That limitation inspired a new generation of models that replaced probability with geometry — the margin-based family.

## 3. Margin-based Models

Margin-based models redefined learning by geometry rather than probability.
The Perceptron, developed in the 1950s, sought a linear boundary that classified all examples correctly. It worked — until data became messy.
Decades later, Support Vector Machines (SVMs) resurrected the idea with mathematical rigor, maximizing the margin between classes and allowing for imperfect separations through soft constraints.

Their central idea is elegant: instead of fitting many points closely, find the boundary that lies farthest from both classes — the safest possible separator. Through kernel functions, SVMs project data into higher-dimensional spaces where complex patterns become linearly separable.

Margin-based models marked the shift from explicit distributional assumptions to optimization and geometry. They balance power and parsimony, generalizing well even on small datasets with many features, such as text classification or genomics.
However, interpretability and scalability suffer as kernels grow complex, making them harder to tune and understand.

Their geometric focus paved the way for the next question: if the best boundary depends on proximity, what if we skip modeling altogether and let neighbors decide directly? Thus emerged instance-based learning.

## 4. Instance-based Models

Instance-based models classify not by abstraction but by resemblance.
The k-Nearest Neighbors (kNN) algorithm embodies the intuition that similarity implies identity. To predict a new observation, kNN looks at the closest existing examples in feature space and assigns the most common label among them.

This approach is lazy — it performs almost no training, deferring all computation to prediction time.
Its simplicity is disarming and its logic profoundly human. We often reason this way ourselves: a new case reminds us of a previous one, and we draw analogies.

Instance-based models excel when decision boundaries are irregular and data are dense. They perform well in small, low-dimensional spaces where local structure matters. Yet they struggle as data scale. Storing and comparing every observation becomes expensive; small changes in distance metrics can flip outcomes dramatically.

Their strengths and weaknesses led researchers to seek hierarchical decisions — a way to learn compact, interpretable structures that could still model nonlinearity. The answer arrived through trees.

##  5. Tree-based Models

Decision Trees changed how we think about classification.
Instead of fitting equations or comparing points, trees ask questions:
“Is income greater than 50,000?” “Is the patient’s age above 45?”
At each node, a split divides the data into purer groups, recursively creating a tree of logical decisions that ends with predicted classes.

Models such as CART (Classification and Regression Trees) and ID3 introduced this principle of recursive partitioning.
The result is intuitive: one can read a tree top-down like a decision manual. Trees handle numeric and categorical features, capture interactions, and require minimal preprocessing.

Their main drawback lies in instability. A small change in data can yield an entirely different tree. They also tend to overfit, learning patterns too specific to the training sample. But trees reintroduced interpretability at a time when models were becoming increasingly opaque.
They set the stage for one of the most transformative ideas in machine learning — combining many weak trees into one strong forest.

## 6. Ensemble Models

The idea of ensembles revolutionized predictive modeling: the collective opinion of many weak learners is often stronger than the judgment of one.
Methods like Bagging (bootstrap aggregating) and Random Forests reduce variance by averaging over many diverse trees, while Boosting methods such as AdaBoost, XGBoost, LightGBM, and CatBoost sequentially correct errors made by previous models.

Ensembles embody the principle of collective intelligence. Each individual tree may be inaccurate, but together they form a robust, flexible predictor.
These models achieve state-of-the-art performance across domains, from credit risk to recommendation systems and beyond.

However, their strength introduces opacity. Hundreds of trees make it difficult to explain why a specific decision was made. Moreover, ensembles demand more computational resources and tuning.
Despite these challenges, they remain the workhorses of modern ML — balancing accuracy, generalization, and practical usability.

As data grew in size and complexity, however, even ensembles struggled to capture abstract relationships in images, audio, and text. The next frontier required models that could learn representations themselves — leading to the era of neural networks.

## 7. Neural Networks for Classification

Neural networks transformed machine learning by introducing representation learning: the ability to discover features automatically through layers of computation.
Early Multilayer Perceptrons (MLPs) extended the logic of linear models with nonlinear activations, capturing complex relationships in tabular data.
Convolutional Neural Networks (CNNs) revolutionized computer vision by learning spatial hierarchies of features: edges, shapes, objects.
Recurrent Neural Networks (RNNs) and later LSTMs and Transformers enabled sequence understanding in language, speech, and time series.

Their conceptual shift was profound: rather than designing features manually, analysts now design architectures that learn features. The model becomes both a learner and a feature engineer.
Neural networks can approximate virtually any function, achieving extraordinary results in domains once thought unreachable by algorithms.

But power comes with cost. These systems require vast data, hardware acceleration, and careful tuning. Interpretability becomes difficult — decisions arise from thousands of parameters interacting in opaque ways.
Yet their capacity for abstraction has expanded what classification can mean: not just labeling, but understanding.

This complexity closes the taxonomy loop — from the transparent logic of equations to the emergent reasoning of layered representations.

## 8. Comparative Reflection: Complexity vs Interpretability

The evolution of classification models is not a ladder of superiority but a landscape of trade-offs.
Linear models are transparent and grounded, ideal when interpretability and causality matter.
Margin-based and instance-based models add geometric sophistication at the expense of simplicity.
Trees and ensembles democratize nonlinearity, achieving strong predictive power with diminishing transparency.
Neural networks push performance further but challenge our ability to explain and trust.

Maturity in data science means choosing the simplest model that solves the problem while respecting ethical and practical boundaries.
A logistic regression that performs nearly as well as a deep network is often the better choice when clarity, speed, or accountability matter.
Each family contributes a different philosophy of learning; progress lies not in abandoning the old but in understanding when to use each wisely.

---- 

In the next section, we will leave the map and enter the terrain.
Having explored how these families connect conceptually, we will now study their mathematical foundations, estimation methods, and implementations — the craft of making classification models work in practice.
 
----

# V. Estimation Methods and Model-Specific Analysis

## Purpose

Classification models are more than algorithms; they are expressions of how we understand uncertainty and structure decision boundaries in data.
This section forms the core analytical backbone of the repository.
Its goal is to explain, model by model, the logic of estimation—how each technique learns, adapts, and makes predictions based on evidence.

In the previous section, we explored the taxonomy of classification models — a conceptual map of the families and their philosophical foundations.
Now, we move from that map to the machinery: we will look inside each model, understanding how it transforms inputs into decisions.

Every model we study here represents a distinct way of thinking about the world: linearity and probability, margins and geometry, similarity and distance, hierarchy and ensemble collaboration, and, ultimately, deep representation.

This part remains purely theoretical and conceptual.
We will not write code or derive formulas step-by-step.
Instead, the goal is to achieve clarity and intuition — to make the mathematics and training principles explain themselves.

All practical implementations, visualizations, and experiments will be developed later in the Practical Annexes (Sections VI–VII), where we will connect theory to execution.

## Guiding Framework

To ensure consistency across all techniques, every model follows the same explanatory skeleton.
This uniform structure allows fair comparison between algorithms and helps build intuition layer by layer.

Each model will therefore be described through the following lenses:

**1. What is it?**
A short conceptual definition and its historical or disciplinary origin.

**2. Why use it?**
The typical scenarios, problems, or data structures where it excels.

**3. Intuition.**
The geometric, probabilistic, or algorithmic “mental picture” of how it learns.

**4. Mathematical foundation.**
The key principle behind its estimation — its objective function or decision rule — explained in plain language and minimal notation.

**5. Training logic.**
A conceptual description of how parameters are adjusted to minimize loss or maximize separation.

**6. Assumptions and limitations.**
The data conditions that must hold for the model to perform well, and where it tends to fail.

**7. Key hyperparameters (conceptual view).**
Explanation of the main parameters that govern flexibility, bias–variance balance, and generalization capacity.

**8. Evaluation focus.**
The most relevant metrics and diagnostic strategies to assess performance (linked to Section III on cross-cutting topics).

**9. When to use / When not to use.**
Guidance on appropriate contexts and common misuses.

**10. References.**
Three foundational academic sources and two reputable web resources for further study.

## Transition to the Model Families

Before diving into individual techniques, it is important to recognize that no model exists in isolation.
Each family of classifiers represents a historical and conceptual response to the limitations of the previous ones.
From linear equations to deep neural networks, the evolution of classification has been a continuous dialogue between simplicity and capacity, interpretability and flexibility, data scarcity and abundance.

We will now follow that natural order — the same logic established in the previous section’s taxonomy — beginning with the most fundamental of all: Linear and Probabilistic Models.

These methods form the bedrock of statistical learning.
They taught machines to make decisions not by memorizing data, but by estimating probabilities and drawing boundaries that separate uncertainty into meaning.

## Canon of Models Covered (by Family)

The diversity of classification models can seem overwhelming at first glance.
Each algorithm has its own philosophy, mathematical logic, and trade-off between interpretability and predictive power.
To make this exploration coherent, we will organize all techniques into families that share a common foundation — linear reasoning, geometric margins, similarity, hierarchical splitting, collective learning, and deep representation.

This “canon of models” mirrors the conceptual order established in the Taxonomy (Section IV), so that we move naturally from the simplest to the most advanced ideas, from probability to representation, from equations we can visualize to networks we can only approximate.

Every model, regardless of its family, will be developed using the same analytical lens:

•	What is it?  → conceptual definition and short historical note.
	
•	Why use it?  → main use cases and decision context.
	
•	Intuition.  → mental model of how the classifier separates or estimates.
	
•	Mathematical foundation.  → principle of estimation or loss function explained in plain terms.
	
•	Training logic.  → how the algorithm learns from data conceptually.
	
•	Assumptions & limitations.  → when the model’s logic holds, and when it breaks.
	
•	Key hyperparameters.  → parameters that shape its flexibility and generalization.
	
•	Evaluation focus.  → metrics that best reflect success for this model.
	
•	When to use / When not to use.  → practical guidance.
	
•	References.  → canonical academic and web sources.
	

This uniform structure ensures that readers can compare models fairly, understand their conceptual genealogy, and recognize that there is no single “best” classifier — only the right one for a given problem and context.

### A. Linear & Probabilistic Models

Linear and probabilistic models represent the foundation of supervised classification.
They are rooted in the idea that decision boundaries can be expressed as linear functions of the input variables, and that uncertainty can be modeled probabilistically.

These models emerged from the intersection of statistics and early pattern recognition in the mid-20th century.
Rather than memorizing examples, they learn relationships between features and outcomes by estimating parameters that maximize the likelihood of observing the data — a statistical view of learning that precedes modern machine learning.

Their key strength lies in interpretability and mathematical elegance.
Each coefficient tells a story about how a variable contributes to the final decision, and probabilities offer a natural way to express confidence.
They perform remarkably well when the relationships in the data are approximately linear and the signal-to-noise ratio is moderate.

However, their simplicity is also their main limitation.
When decision boundaries are nonlinear or interactions between features are complex, linear and probabilistic models may struggle — paving the way for more flexible approaches such as margin-based and tree-based learners.

In this family, we will explore five major techniques that together define the statistical heart of classification:

1.	Logistic Regression (binary, multinomial) – the cornerstone of probabilistic classification, modeling the log-odds of class membership.

2.	Regularized Logistic Regression (L1, L2, Elastic Net) – adding control over model complexity through penalty terms.

3.	Linear Discriminant Analysis (LDA) – separating classes by maximizing between-class variance.

4.	Quadratic Discriminant Analysis (QDA) – extending LDA with distinct covariance structures for each class.

5.	Naive Bayes (Gaussian, Multinomial, Bernoulli, Complement) – a probabilistic learner based on independence assumptions and Bayes’ theorem.

Together, these models form the analytical backbone of classical classification — the bridge between pure statistics and the more flexible, data-driven methods that followed.

#### ** 1.Logistic Regression (binary, multinomial)**

**What is it?**

Logistic Regression is the most fundamental probabilistic model for classification.
Despite its name, it is not a regression method in the traditional sense but a predictive model for categorical outcomes.
It estimates the probability that an observation belongs to a particular class by modeling the relationship between the input features and the log-odds of the event.

The model originated in the early 20th century through the work of David Cox (1958), who extended earlier logit models from statistics and epidemiology to binary outcomes.
It later became a cornerstone of statistical learning theory, serving as the conceptual bridge between linear regression and modern classification algorithms.

⸻

**Why use it?**

Logistic Regression is used when the goal is to classify observations into discrete categories (e.g., “yes/no”, “fraud/not fraud”, “disease/healthy”) while also quantifying the confidence of those classifications.
It is especially valuable in applications where interpretability and probability calibration matter as much as accuracy — such as medicine, finance, and social sciences.

Typical use cases include:
	•	Predicting whether a patient has a disease given test results.
	•	Determining if a transaction is fraudulent.
	•	Estimating whether a customer will churn.
	•	Modeling the probability of an event (success/failure, default/no default).

Its transparency makes it ideal for regulated industries, where decision-making must be explainable and auditable.

⸻

**Intuition**

At its heart, Logistic Regression asks a simple question:

“Given this input, how likely is it that the outcome is 1 instead of 0?”

Imagine a straight line (in one dimension) or a plane (in multiple dimensions) dividing two classes.
Instead of producing a hard decision, the model computes a smooth curve — the logistic function — that transforms any linear combination of inputs into a probability between 0 and 1.

In geometric terms, Logistic Regression learns a linear boundary in the feature space — the point where the model is equally uncertain about both classes (probability = 0.5).
In probabilistic terms, it models the log-odds of belonging to one class as a linear function of the inputs:

$$
\log \left( \frac{P(y = 1 \mid x)}{1 - P(y = 1 \mid x)} \right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p
$$

The logistic (sigmoid) transformation then converts these log-odds back into probabilities:

$$
P(y = 1 \mid x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_p x_p)}}
$$

This combination of linearity and nonlinearity — a linear function wrapped by a nonlinear sigmoid — is what makes Logistic Regression both simple and powerful.

💡 Note:
A closely related model, the Probit Regression, replaces the logistic (sigmoid) function with the cumulative normal distribution.
Both aim to map linear predictors into probabilities, differing mainly in their link functions.
The logit is preferred for interpretability and computational simplicity, but the probit offers similar results and can be used as an alternative in classification contexts.

⸻

**Mathematical foundation**

The core principle is Maximum Likelihood Estimation (MLE).
Given a dataset with binary outcomes y_i \in \{0,1\} and predictors x_i,
the model estimates coefficients \beta that maximize the likelihood of observing the data:

$$
L(\beta) = \prod_{i=1}^{n} P(y_i \mid x_i) = \prod_{i=1}^{n} [p_i]^{y_i} [1 - p_i]^{1 - y_i}
$$

where

$$
p_i = \frac{1}{1 + e^{-(\beta_0 + \beta^T x_i)}}
$$

The optimization is typically performed by minimizing the log-loss (negative log-likelihood):

$$
\text{Loss}(\beta) = - \sum_{i=1}^{n} \Big[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \Big]
$$

This formulation naturally penalizes confident but wrong predictions more heavily than uncertain ones,
encouraging models that are both accurate and probabilistically well-calibrated.

For multinomial Logistic Regression, the model generalizes to multiple classes using the softmax function,
assigning probabilities to each possible outcome while ensuring they sum to 1:

$$
P(y = k \mid x) = \frac{e^{\beta_k^T x}}{\sum_{j=1}^{K} e^{\beta_j^T x}}
$$

⸻

**Training logic**

Training involves iteratively adjusting coefficients to minimize the log-loss.
Because the loss function is convex, gradient-based methods such as Newton–Raphson, Iteratively Reweighted Least Squares (IRLS), or Stochastic Gradient Descent (SGD) are guaranteed to converge to a global minimum.

Conceptually, each iteration performs three steps:
	1.	Compute predicted probabilities from the current coefficients.
	2.	Measure how far predictions are from true outcomes (residuals).
	3.	Adjust coefficients proportionally to the direction of greatest improvement (the gradient).

This process continues until the changes in coefficients are small enough to indicate convergence.
The model’s simplicity makes training stable, efficient, and reproducible, even for moderately large datasets.

⸻

**Assumptions and limitations**

Logistic Regression is powerful but relies on several assumptions:
	•	Linearity in the log-odds: the relationship between predictors and the logit of the outcome must be linear.
	•	Independent observations: errors across samples should be uncorrelated.
	•	No extreme multicollinearity: predictors should not be highly correlated, as this destabilizes coefficients.
	•	Sufficient sample size: large enough to estimate reliable probabilities.

Limitations include:
	•	Poor performance when decision boundaries are nonlinear.
	•	Sensitivity to outliers and missing values.
	•	Difficulty capturing complex feature interactions without manual feature engineering.

Nonetheless, these same constraints make it highly interpretable, a quality often lost in more complex algorithms.

⸻

**Key hyperparameters (conceptual view)**

Although Logistic Regression is mathematically straightforward, several configuration choices influence its behavior:
	•	Regularization strength (C or λ): controls overfitting by shrinking coefficients.
	•	Penalty type (L1, L2, Elastic Net): determines how regularization is applied (sparse vs smooth solutions).
	•	Solver: optimization algorithm (e.g., "liblinear", "lbfgs", "saga").
	•	Class weights: rebalance the influence of minority classes in imbalanced datasets.

These hyperparameters govern the bias–variance trade-off, balancing simplicity and generalization.

⸻

**Evaluation focus**

Because Logistic Regression produces probabilities, it should be evaluated not only for classification accuracy but also for probability calibration —
how well predicted probabilities match actual observed frequencies.

Key metrics include:
	•	Log-loss: direct measure of probabilistic accuracy.
	•	ROC–AUC: overall discrimination power between classes.
	•	PR–AUC: preferred in imbalanced classification tasks.
	•	Brier score: calibration and reliability of predicted probabilities.
	•	Confusion matrix & F1-score: evaluation under chosen decision thresholds.

Interpreting coefficient signs and magnitudes adds a qualitative layer —
connecting statistical learning with domain understanding.

⸻

**When to use / When not to use**

Use Logistic Regression when:
	•	The relationship between predictors and the outcome is roughly linear.
	•	You need interpretable coefficients and well-calibrated probabilities.
	•	The dataset is of moderate size and not excessively high-dimensional.
	•	Transparency and explainability are priorities (e.g., healthcare, public policy, credit scoring).

Avoid Logistic Regression when:
	•	Data exhibit strong nonlinearities or complex feature interactions.
	•	Predictors are highly correlated or numerous relative to observations.
	•	Decision boundaries are highly irregular or discontinuous.
	•	You prioritize pure predictive accuracy over interpretability.

⸻

**References**

Canonical papers
	1.	Cox, D. R. (1958). The Regression Analysis of Binary Sequences. Journal of the Royal Statistical Society, Series B.
	2.	McCullagh, P. & Nelder, J. (1989). Generalized Linear Models (2nd ed.). Chapman & Hall.
	3.	Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). Applied Logistic Regression (3rd ed.). Wiley.

Web resources

1.	StatQuest – Logistic Regression Clearly Explained
	https://statquest.org/video/logistic-regression-clearly-explained/
3.	Scikit-learn User Guide: Logistic Regression￼
	https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

⸻

----

Logistic Regression established the foundation for probabilistic classification —
a world where decisions are guided by likelihood and evidence rather than hard rules.
Yet, its linear boundary can only go so far.
As data became more complex, new methods emerged to control overfitting and handle correlated or numerous predictors.

The next step in our journey explores how Regularized Logistic Regression extends this model —
adding flexibility without losing interpretability, through the elegant mathematics of penalization.

----

#### 2. Regularized Logistic Regression (L1, L2, Elastic Net)

What is it?

Regularized Logistic Regression is an enhanced version of traditional Logistic Regression that adds a penalty term to the loss function to control model complexity.
While standard logistic regression seeks coefficients that perfectly fit the data, regularization constrains them to remain small or sparse — improving generalization, stability, and interpretability.

This idea emerged from the evolution of penalized likelihood methods in the late 20th century, especially from the work on ridge regression (Hoerl & Kennard, 1970) and Lasso (Tibshirani, 1996).
By integrating these penalties into logistic regression, statisticians and data scientists obtained a model that balances fit and simplicity, preventing overfitting in high-dimensional or correlated datasets.

⸻

Why use it?

Regularized Logistic Regression is preferred when:
	•	You have many predictors or potential multicollinearity.
	•	The model overfits the training data.
	•	You want automatic feature selection (especially with L1).
	•	You need better stability and generalization without losing interpretability.

Common applications:
	•	Credit scoring with dozens of financial indicators.
	•	Text or NLP classification (with many sparse features).
	•	Biomedical studies where predictors are correlated (e.g., genetic markers).
	•	Marketing models where variable selection is needed.

⸻

Intuition

Regularization is like a gentle constraint placed on the model —
it says: “fit the data well, but don’t overreact.”

In standard logistic regression, coefficients can grow large to accommodate small patterns or noise.
Regularization keeps them small or pushes some to zero (in the case of L1), which simplifies the model and improves its ability to generalize.

Imagine tuning a musical instrument:
	•	Without regularization, each string (feature) vibrates freely, sometimes creating noise.
	•	With regularization, you tighten them just enough to maintain harmony — a cleaner, more stable sound.

In geometric terms, regularization reshapes the optimization landscape:
	•	L2 (Ridge) uses circular (Euclidean) constraints, shrinking all coefficients smoothly.
	•	L1 (Lasso) uses diamond-shaped constraints, which naturally “cut” some coefficients to zero.
	•	Elastic Net blends both worlds — it shrinks most coefficients (L2) but can also eliminate the weakest (L1).

⸻

Mathematical foundation

Regularized Logistic Regression minimizes the penalized log-loss function:

$$
\text{Loss}{\text{reg}}(\beta) = - \sum{i=1}^{n} \Big[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \Big] + \lambda P(\beta)
$$

where p_i = \frac{1}{1 + e^{-(\beta_0 + \beta^T x_i)}},
and P(\beta) is the penalty term that depends on the chosen regularization type:
	•	L1 (Lasso):
$$
P(\beta) = \sum_{j=1}^{p} |\beta_j|
$$
Encourages sparsity by forcing irrelevant coefficients to zero.
	•	L2 (Ridge):
$$
P(\beta) = \sum_{j=1}^{p} \beta_j^2
$$
Shrinks all coefficients toward zero smoothly, stabilizing correlated variables.
	•	Elastic Net:
$$
P(\beta) = \alpha \sum_{j=1}^{p} |\beta_j| + (1 - \alpha) \sum_{j=1}^{p} \beta_j^2
$$
Combines both penalties, with \alpha \in [0,1] controlling the balance between sparsity (L1) and smoothness (L2).

The λ (lambda) parameter controls the strength of regularization:
	•	Large λ → stronger penalty → simpler model (higher bias, lower variance).
	•	Small λ → weaker penalty → model behaves like standard logistic regression.

⸻

Training logic

The training process is similar to ordinary logistic regression but includes the regularization term in the optimization objective.
Because the penalty can make the function non-differentiable (especially with L1), solvers use coordinate descent, SGD, or proximal gradient methods to find the optimal coefficients.

The iterative logic can be summarized as:

1.	Compute predicted probabilities using the current coefficients.
2.	
3.	Calculate the gradient of the loss plus the penalty.
4.	
5.	Update coefficients in the opposite direction of the gradient, adjusted by the learning rate.
6.	
7.	For L1 penalties, coefficients that shrink below a threshold become exactly zero.

This training approach ensures stability and convergence, even for large or sparse datasets.

⸻

Assumptions and limitations

The assumptions remain mostly the same as for standard logistic regression:

•	Linearity in the log-odds.

•	Independence of observations.

•	No severe outliers or missingness.

However, regularization relaxes the requirement of uncorrelated predictors, as L2 helps stabilize correlated variables and L1 can remove redundant ones.

Limitations:
	•	Choice of λ and α is critical — too high can underfit, too low can overfit.
	•	Coefficients lose their direct interpretability when heavily regularized.
	•	L1 may behave unstably when predictors are highly correlated (Elastic Net often helps).

⸻

Key hyperparameters (conceptual view)

•	λ (Regularization strength): controls the trade-off between fit and simplicity.
Larger λ increases the penalty, leading to smaller coefficients.

•	Penalty type:

•	"l1" for Lasso (sparse model).

•	"l2" for Ridge (smooth shrinkage).

•	"elasticnet" for a combination.

•	α (Elastic Net mixing parameter): balances L1 and L2 penalties (α=1 → Lasso, α=0 → Ridge).

•	Solver: must support the chosen penalty (e.g., "liblinear" for L1, "saga" for Elastic Net).

•	Class weights: optionally adjust for imbalanced data.

These parameters define how much regularization is applied and which type of structure is favored in the solution.

⸻

Evaluation focus

The evaluation metrics are the same as for ordinary logistic regression (log-loss, ROC–AUC, PR–AUC, Brier score),
but special attention should be given to bias–variance trade-offs and feature selection stability.

When tuning λ:
	•	Track both training and validation log-loss curves to avoid under/overfitting.
	•	Use cross-validation (e.g., k-fold CV) to find the optimal λ.
	•	Examine which variables remain active (non-zero) — this provides interpretability insights.

💡 Tip:
In practical applications, Elastic Net often performs best when the number of features is large and correlated —
offering the robustness of Ridge and the feature selection power of Lasso.

⸻

When to use / When not to use

Use Regularized Logistic Regression when:

•	You have many predictors (high dimensionality).

•	Variables are correlated or redundant.

•	You need better generalization than standard logistic regression.

•	You want to perform embedded feature selection.

Avoid it when:

•	Interpretability of individual coefficients is critical (since penalties distort raw magnitudes).

•	The dataset is small and simple (standard logistic regression suffices).

•	The relationship between features and outcomes is strongly nonlinear (consider tree-based or kernel methods).

⸻

References

Canonical papers
	1.	Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. Technometrics, 12(1), 55–67.
	2.	Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. Journal of the Royal Statistical Society, Series B, 58(1), 267–288.
	3.	Zou, H., & Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. Journal of the Royal Statistical Society, Series B, 67(2), 301–320.

Web resources
	1.	StatQuest – Ridge, Lasso, and Elastic Net Regression￼
	2.	Scikit-learn User Guide: Regularization in Logistic Regression￼


----

Regularized Logistic Regression introduced discipline into linear models — teaching them to resist noise, ignore irrelevant predictors, and focus on signal. Yet, it remains confined to the assumption that decision boundaries are linear in the feature space.

To go beyond that — to model more complex, nonlinear separations — we must leave the realm of pure probability and enter that of geometry and distance.

The next model family, Linear Discriminant Analysis (LDA),
embodies this shift: it keeps a probabilistic heart but builds its boundary using the geometry of variance and covariance: a bridge between statistics and pattern recognition.

----

#### 3. Linear Discriminant Analysis (LDA)

#### 4. Quadratic Discriminant Analysis (QDA)

#### 5. Naive Bayes (Gaussian, Multinomial, Bernoulli, Complement)


### B. Margin-based Models

### C. Instance-based Models

### D. Tree-based Models

### E. Ensemble Models

### F. Neural Networks for Classification

### G. Multiclass & Multilabel Strategies


##



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


