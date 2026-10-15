# FraudLens: Comprehensive Algorithm Guide & Viva Defense

This document provides a thorough theoretical and practical explanation of the four algorithms implemented in **FraudLens**, including their mathematical formulation, algorithmic guarantees, design trade-offs, and viva defense answers.

---

## 1. Project Overview & Problem Context

- **Application Domain**: Anti-Money Laundering (AML) transaction-pattern mining and predictive classification.
- **Dataset**: IBM Transactions for Anti-Money Laundering (AML), `HI-Small_Trans.csv` (5,078,345 records, 453.63 MB).
- **Core Challenge**: Extreme class imbalance (0.102% positive laundering rate) and dual objective:
  1. Discover interpretable associative and high-value pattern rules (unsupervised pattern mining).
  2. Classify individual suspicious transactions accurately without future data leakage (supervised classification).
- **Architecture**: Exactly four modular, independently testable algorithms:
  1. **Standard Apriori** (Association rule mining)
  2. **Adaptive Apriori / MSApriori** (Item-adaptive minimum support mining)
  3. **Two-Phase High-Utility Itemset Mining (HUIM)** (Monetary volume utility mining)
  4. **Random Forest Classifier** (Chronological AML transaction classifier)

---

## 2. Algorithm 1: Standard Apriori

### 2.1 Theoretical Foundation & Mathematics
The Apriori algorithm (Agrawal & Srikant, 1994) discovers frequent itemsets in a transaction database $D = \{T_1, T_2, \dots, T_n\}$ where each transaction $T_d \subseteq I$ is a set of categorical items.

1. **Support Count**:
   $$\text{count}(X) = |\{T_d \in D \mid X \subseteq T_d\}|$$
2. **Relative Support**:
   $$\text{supp}(X) = \frac{\text{count}(X)}{|D|}$$
   An itemset $X$ is frequent if $\text{supp}(X) \ge \text{min\_support}$.
3. **The Apriori Property (Downward Closure / Anti-Monotonicity)**:
   $$\forall X \subseteq Y \implies \text{supp}(Y) \le \text{supp}(X)$$
   *Theorem*: If an itemset $X$ is infrequent ($\text{supp}(X) < \text{min\_support}$), then any superset $Y \supset X$ is guaranteed to be infrequent and can be pruned immediately without scanning the database.

4. **Association Rule Generation**:
   A rule $X \to Y$ (where $X \cap Y = \emptyset$) represents an associative pattern from antecedent $X$ to consequent $Y$:
   - **Support**: $\text{supp}(X \to Y) = \text{supp}(X \cup Y)$
   - **Confidence**: Conditional probability $P(Y \mid X)$:
     $$\text{conf}(X \to Y) = \frac{\text{supp}(X \cup Y)}{\text{supp}(X)}$$
   - **Lift**: Measure of dependency/correlation beyond random chance:
     $$\text{lift}(X \to Y) = \frac{\text{conf}(X \to Y)}{\text{supp}(Y)} = \frac{\text{supp}(X \cup Y)}{\text{supp}(X) \times \text{supp}(Y)}$$
     - $\text{lift} = 1$: $X$ and $Y$ are statistically independent.
     - $\text{lift} > 1$: Positive association (presence of $X$ strongly implies $Y$).
     - $\text{lift} < 1$: Negative association / substitution.

### 2.2 Algorithm Procedure
1. **L1 Generation**: Find all 1-itemsets whose support $\ge \text{min\_support}$.
2. **Candidate Generation ($L_{k-1} \times L_{k-1}$ Join)**:
   Join pairs of frequent $(k-1)$-itemsets sharing the first $k-2$ items in sorted canonical order.
3. **Pruning Step**:
   For each candidate $c \in C_k$, verify that every $(k-1)$-subset of $c$ exists in $L_{k-1}$. If any subset is missing, prune $c$.
4. **Support Counting**: Scan transactions to count occurrences of remaining candidates.
5. **Rule Generation**: Partition frequent itemsets into antecedent and consequent subsets, filtering by $\text{min\_confidence}$.

### 2.3 Limitations in AML
- **The Rare Item Dilemma**: Laundering patterns occur in $\approx 0.10\%$ of transactions. Setting $\text{min\_support} \le 0.001$ causes candidate explosion on mundane items (`USD`, `ACH`), while setting $\text{min\_support} = 0.05$ completely misses all laundering patterns.

---

## 3. Algorithm 2: Adaptive Apriori (MSApriori)

### 3.1 What Adapts, How, and Why?
- **What adapts?** The minimum support threshold is no longer uniform across all items. Each item $i$ is assigned an individual Minimum Item Support $MIS(i)$.
- **Why does it adapt?** High-risk, rare items (e.g., `Laundering:True`, `Format:Bitcoin`, `AmountBand:VeryHigh`) require a low threshold so their patterns can be surfaced, whereas frequent mundane items (e.g., `PaymentCurr:US Dollar`, `Format:ACH`) require a higher threshold to prevent combinatorial explosion.
- **Itemset Threshold**:
  $$\text{min\_support}(X) = \min_{i \in X} MIS(i)$$

### 3.2 Preserving Downward-Closure with Exact Mathematical Rigor
If arbitrary thresholds are assigned to itemsets naively, downward closure fails: a subset $S \subset X$ might only contain high-$MIS$ items and fail its own higher threshold, causing standard pruning to discard candidates incorrectly.

**The MSApriori Solution (Liu, Hsu, Ma, KDD 1999)**:
1. Sort all items in the vocabulary in ascending order of their $MIS$:
   $$MIS(i_1) \le MIS(i_2) \le \dots \le MIS(i_m)$$
2. For any canonically sorted itemset $X = [i_1, i_2, \dots, i_k]$, the lowest $MIS$ item is ALWAYS the first item $i_1$:
   $$\text{min\_support}(X) = MIS(i_1)$$
3. **Valid Pruning Rule**:
   For candidate $c = [i_1, i_2, \dots, i_k]$, every $(k-1)$-subset containing $i_1$ MUST be frequent under $MIS(i_1)$.
   Subsets excluding $i_1$ (specifically $c \setminus \{i_1\}$) have higher thresholds $MIS(i_2) \ge MIS(i_1)$ and are **not** used to prune $c$.

---

## 4. Algorithm 3: Genuine High-Utility Itemset Mining (HUIM)

### 4.1 Theoretical Foundation & Definitions
Standard association mining treats all transactions and items as binary (present or absent). In banking, transaction amounts matter fundamentally: a pattern moving \$1,000,000 carries vastly greater risk than the same pattern moving \$10.

1. **Internal Utility $q(i, T_d)$**: The quantitative transaction volume or quantity of item $i$ in transaction $T_d$ (in FraudLens: `Amount Paid`).
2. **External Utility $p(i)$**: Intrinsic unit weight or risk multiplier of item $i$ (default = 1.0).
3. **Item Utility**: $u(i, T_d) = p(i) \times q(i, T_d)$
4. **Itemset Utility in Transaction**:
   $$u(X, T_d) = \sum_{i \in X} u(i, T_d) \quad \text{if } X \subseteq T_d, \text{ else } 0$$
5. **Itemset Utility in Database**:
   $$u(X) = \sum_{T_d \in D} u(X, T_d)$$
6. **Transaction Utility (TU)**:
   $$TU(T_d) = \sum_{i \in T_d} u(i, T_d)$$
7. **High-Utility Itemset (HUI)**: $X$ is high-utility if and only if $u(X) \ge \text{min\_utility}$.

### 4.2 Non-Monotonicity & The TWU Upper-Bound Theorem
Utility is **not** monotonic: $X \subset Y$ does **not** imply $u(Y) \le u(X)$. In fact, adding items can increase utility because item utilities sum together! Standard Apriori pruning cannot be used on raw utility.

**Transaction-Weighted Utility (TWU)**:
$$TWU(X) = \sum_{T_d \in D, X \subseteq T_d} TU(T_d)$$

**Theorem (TWU Downward-Closure Property)**:
1. $u(X) \le TWU(X)$ for any itemset $X$.
2. For any $X \subset Y$, $TWU(Y) \le TWU(X)$.
3. **Pruning Guarantee**: If $TWU(X) < \text{min\_utility}$, then for ALL supersets $Y \supset X$:
   $$u(Y) \le TWU(Y) \le TWU(X) < \text{min\_utility}$$
   Therefore, $X$ and all its supersets can be safely pruned!

### 4.3 Two-Phase Algorithm Procedure
- **Phase 1**: Mine High Transaction-Weighted Utilization Itemsets (HTWUIs) level-by-level using candidate generation and TWU downward-closure pruning.
- **Phase 2**: Calculate the exact utility $u(X)$ for each HTWUI candidate, filtering out overestimates to return true HUIs.

### 4.4 Financial Proxy Disclaimer
Transaction amount is an objective proxy for financial exposure. High-utility itemsets identify pattern combinations carrying large monetary volume. This proxy does **not** assume transaction amounts equal criminal profit or verified loss.

---

## 5. Algorithm 4: Random Forest Classifier

### 5.1 Model Selection & Architecture
- Sole classifier: `sklearn.ensemble.RandomForestClassifier`.
- An ensemble of decorrelated decision trees using bootstrap aggregation (bagging) and random feature subspace selection.

### 5.2 Chronological Splitting & Leakage Prevention
Financial transaction data violates the IID (independent and identically distributed) assumption. Random k-fold cross-validation causes **lookahead bias** (training on future events to predict past transactions).
- **Split Strategy**: Chronological partition:
  - First 70% of time window $\to$ Training set
  - Next 15% of time window $\to$ Validation set
  - Final 15% of time window $\to$ Out-of-time Test set
- **Preprocessing Isolation**: Categorical one-hot encoders are fitted strictly on `X_train`, then applied to `X_val` and `X_test`. Unseen categories are handled safely via `handle_unknown="ignore"`.
- **Target Isolation**: Target column `Is Laundering` is strictly isolated into $y$ and never present in $X$.

### 5.3 Class Imbalance Handling
In the IBM AML HI-Small dataset, only 0.102% of transactions are laundering.
- Handled using `class_weight="balanced"`, which assigns weights inversely proportional to class frequencies:
  $$w_j = \frac{n}{k \times n_j}$$
- Test set remains completely raw and un-resampled to reflect real-world operational distributions.

### 5.4 Comprehensive Evaluation Metrics
- **Precision**: $\frac{TP}{TP + FP}$ (avoids alert fatigue for compliance officers).
- **Recall**: $\frac{TP}{TP + FN}$ (detects actual money laundering cases).
- **F1-Score**: Harmonic mean of Precision and Recall.
- **ROC-AUC**: True positive rate vs false positive rate across all thresholds.
- **PR-AUC (Average Precision)**: Precision vs recall curve; the gold standard for severely imbalanced fraud detection.
- **Probability Calibration Disclaimer**: Predicted probabilities represent tree ensemble vote shares, not calibrated Bayesian probabilities of guilt.

---

## 6. Viva Defense: Questions & Answers

### Q1: Why did you choose these specific four algorithms?
**Answer**:
"The project addresses the complete AML analytical lifecycle. Standard Apriori establishes baseline association rule mining. Adaptive Apriori solves the fundamental 'Rare Item Problem' in AML by assigning lower support thresholds to rare suspicious items while keeping mundane items constrained. High-Utility Itemset Mining incorporates transaction monetary volume, identifying patterns that move large financial value. Finally, Random Forest provides supervised out-of-time classification to score incoming transactions."

### Q2: Why can't you just use Standard Apriori with a very low min_support for AML?
**Answer**:
"Setting a global min_support low enough to capture rare laundering transactions (around 0.10%) causes a combinatorial explosion of candidates among mundane everyday items like ACH transfers or US Dollar transactions. This exhausts memory and creates millions of trivial rules. Adaptive Apriori solves this by allowing item-specific minimum item support (MIS)."

### Q3: Why is utility non-monotonic, and how does TWU solve this?
**Answer**:
"Support is monotonic because adding items to a basket can only decrease or maintain transaction frequency ($X \subseteq Y \implies \text{supp}(Y) \le \text{supp}(X)$). Utility is non-monotonic because item utilities add together: a superset $\{A, B\}$ can have a higher total dollar utility than $\{A\}$ alone. We solve this using Transaction-Weighted Utility (TWU), which sums the total utilities of all transactions containing the itemset. Mathematically, TWU is guaranteed to be downward-closed ($TWU(Y) \le TWU(X)$) and an upper bound on utility ($u(X) \le TWU(X)$), allowing sound pruning."

### Q4: Why did you use a chronological split instead of random k-fold cross-validation?
**Answer**:
"In financial transactions, future events cannot be used to predict past events. Random k-fold cross-validation shuffles data across time, causing lookahead bias and data leakage where the model memorizes future market states. A chronological split (70% train, 15% validation, 15% test) strictly respects the arrow of time."

### Q5: What does your high ROC-AUC score mean when class imbalance is 0.10%?
**Answer**:
"ROC-AUC measures the probability that the classifier ranks a randomly chosen positive transaction higher than a randomly chosen negative transaction. While our model achieves a high ROC-AUC (>0.94), in severe class imbalance, ROC-AUC can be overly optimistic because large numbers of true negatives dominate the false positive rate. Therefore, we also report PR-AUC (Average Precision), Precision, Recall, and the full Confusion Matrix."
