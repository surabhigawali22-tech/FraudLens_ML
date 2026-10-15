# FraudLens: Comprehensive Viva Voce Defense Guide

This document is structured specifically to help an Information Technology undergraduate student defend **FraudLens** during a formal college viva examination.

---

## 1. Project High-Level Pitch (The 60-Second Overview)

> "FraudLens is an Anti-Money Laundering (AML) transaction analytics and predictive classification system built on IBM's synthetic AML dataset (HI-Small, containing 5.07 million transactions). In banking operations, laundering represents an extreme needle-in-a-haystack problem: only 0.102% of transactions are laundering. To tackle this, FraudLens integrates **exactly four algorithms**:
> 1. **Standard Apriori** to establish baseline frequent behavioral patterns.
> 2. **Adaptive Apriori (MSApriori)** to solve the 'Rare Item Dilemma' by assigning lower support thresholds to rare high-risk items without causing combinatorial candidate explosion on mundane items.
> 3. **Two-Phase High-Utility Itemset Mining (HUIM)** using Transaction-Weighted Utility (TWU) downward-closure pruning to pinpoint patterns that carry high cumulative financial volume.
> 4. **Random Forest Classifier** using a strict chronological 70/15/15 train/val/test split and balanced class weighting for out-of-time transaction risk scoring.
> The backend is powered by FastAPI with streaming chunked dataset loaders and deterministic caching, and the frontend is an accessible, light-theme banking operations portal."

---

## 2. Core Technical Questions & Defenses

### Question 1: What dataset did you use, and why is it simulated?
**Answer**:
"We used the **IBM Transactions for Anti-Money Laundering (AML) - HI-Small** dataset published on Kaggle. It contains **5,078,345 records** across 11 columns. Real banking AML data is strictly confidential under GDPR and banking secrecy laws; therefore, IBM generated high-fidelity simulated multi-agent AML transaction data containing known laundering topologies (fan-in, fan-out, cycle, and scatter-gather). We verified that laundering accounts for exactly **5,177 transactions (0.102%)** across an 18-day chronological window."

### Question 2: Why are there four algorithms instead of just training XGBoost or an Autoencoder?
**Answer**:
"In real-world AML compliance, machine learning models are 'black boxes' that cannot explain structural network relationships or institutional behavioral norms. Regulators require **explainability**. We designed a multi-tier analytical system:
- **Association Mining (Standard & Adaptive Apriori)** discovers human-interpretable compliance rules (e.g., 'Interbank credit card transfers under $1k in USD').
- **Utility Mining (HUIM)** directs audit focus toward high monetary volume exposure.
- **Random Forest** scores individual incoming transactions.
We strictly avoided adding unnecessary algorithms (like Logistic Regression, Isolation Forest, or XGBoost) to maintain a focused, verifiable architecture."

### Question 3: What is the "Rare Item Problem" in Apriori, and how does your Adaptive Apriori solve it?
**Answer**:
"In Standard Apriori, a single uniform $\text{min\_support}$ threshold is applied to all items. Because laundering transactions occur in only 0.10% of records, setting $\text{min\_support} = 0.05$ (5%) causes all rare laundering patterns to be pruned. Conversely, setting $\text{min\_support} \le 0.0005$ triggers **combinatorial candidate explosion** on mundane everyday items (millions of trivial combinations of domestic ACH, Cash, and USD).

Our **Adaptive Apriori** implements **MSApriori (Liu, Hsu, Ma, KDD 1999)**:
1. Each item has an individual Minimum Item Support ($MIS$). Mundane items use a high base MIS (e.g. 10%), while rare items (Bitcoin, Cross-Currency, Laundering flags) receive a low MIS (e.g. 0.05%).
2. Items are sorted in ascending order of MIS ($MIS(i_1) \le MIS(i_2) \le \dots$).
3. The threshold for any itemset is determined by its lowest-MIS item: $\text{min\_supp}(X) = MIS(i_1)$.
4. **Valid Pruning Guarantee**: All $(k-1)$-subsets containing $i_1$ must be frequent under $MIS(i_1)$. Subsets excluding $i_1$ have higher thresholds and cannot falsely prune the candidate. This mathematically preserves the downward-closure property."

### Question 4: Why is raw utility non-monotonic, and how does TWU solve this?
**Answer**:
"Support is monotonic (anti-monotonic) because adding an item to a basket can only decrease or maintain transaction frequency: $X \subseteq Y \implies \text{supp}(Y) \le \text{supp}(X)$.
Utility is **non-monotonic** because item utilities add together: a superset $\{A, B\}$ can have a higher dollar total than $\{A\}$ alone. Therefore, we cannot use Apriori pruning directly on raw utility ($u(X) < \text{min\_utility}$ does not imply $u(Y) < \text{min\_utility}$).

We solve this using **Transaction-Weighted Utility (TWU)**:
$$TWU(X) = \sum_{T_d \supseteq X} TU(T_d)$$
Mathematically, TWU is guaranteed to be downward-closed ($TWU(Y) \le TWU(X)$) and an upper bound on exact utility ($u(X) \le TWU(X)$).
If $TWU(X) < \text{min\_utility}$, then for ALL supersets $Y \supset X$, $u(Y) \le TWU(Y) \le TWU(X) < \text{min\_utility}$.
Therefore, $X$ and all its supersets can be pruned with mathematical certainty. In Phase 2, we scan surviving candidates to compute exact utilities and filter out overestimates."

### Question 5: Why did you use a chronological train/validation/test split instead of random k-fold cross-validation?
**Answer**:
"In financial transactions, data violates the IID (independent and identically distributed) assumption. Transactions are time-series events where market states and laundering typologies evolve. Using random k-fold cross-validation shuffles data across time, causing **lookahead bias / data leakage**—the model is trained on future transactions to predict past transactions.
We enforced a strict **chronological partition** (first 70% of time window for training, next 15% for validation, final 15% for out-of-time testing). All preprocessors (such as OneHotEncoders) are fitted **only** on the training set."

### Question 6: Why is Accuracy a useless metric for your Random Forest model?
**Answer**:
"Because the dataset has **0.102% positive class imbalance** (99.898% legitimate). A trivial 'dummy' model that always predicts 'Legitimate' (0) for every single transaction achieves **99.898% accuracy**, yet detects **zero money laundering incidents**!
Therefore, we completely disregard accuracy and instead evaluate:
1. **Recall**: What percentage of actual laundering cases did the model catch?
2. **Precision**: When the model flags a transaction, how often is it right (mitigating compliance alert fatigue)?
3. **PR-AUC (Average Precision)**: The gold standard for extreme class imbalance.
4. **ROC-AUC**: Evaluates ranking capability across all thresholds (>0.94 on our chronological test set).
5. **Confusion Matrix**: Full operational count of True Positives, False Positives, True Negatives, and False Negatives."

### Question 7: How did you ensure zero target leakage in your Random Forest pipeline?
**Answer**:
"We implemented target isolation at three independent architectural layers:
1. **Data Representation Layer**: In `app.data.features.build_rf_feature_matrix()`, the target column `Is Laundering` is extracted into $y$ and explicitly excluded from feature matrix $X$.
2. **Algorithm Pipeline Layer**: In `AMLRandomForestPipeline.fit()`, an assertion raises a `ValueError` if `Is Laundering` is present in $X$.
3. **API Validation Layer**: In `schemas.random_forest.RFPredictItem`, a Pydantic model validator intercepts and rejects any inference payload containing `'Is Laundering'`, `'islaundering'`, or `'is_laundering'` with a `422 Unprocessable Entity` validation error."

### Question 8: How is the 453 MB CSV handled without running out of memory (OOM)?
**Answer**:
"In `app.data.loader.py`, we implement a streaming generator `iter_chunks()` using Pandas' `chunksize=DEFAULT_CHUNK_ROWS` (200,000 rows). Inspection, summary calculation, and development experiments stream the CSV chunk-by-chunk. Interactive API requests operate on bounded chronological samples (e.g., 25,000 or 50,000 rows), ensuring RAM consumption remains below a few hundred megabytes even on standard student laptops."

### Question 9: What is the purpose of the Light-Theme Banking Portal?
**Answer**:
"Compliance officers in banks are not data scientists. A dark, crowded developer dashboard filled with technical jargon creates confusion. We designed a clear, light-theme portal that:
1. Translates technical metrics into plain English (e.g., explaining Lift as '6.3x more frequent than random chance').
2. Explicitly warns that a model prediction or associative pattern is an **investigative priority**, never confirmation of guilt or proof of laundering.
3. Provides a shared transaction investigation modal with actionable review steps for compliance staff."
