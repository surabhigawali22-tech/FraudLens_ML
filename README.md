# FraudLens: AML Transaction Pattern Mining & Machine Learning System

FraudLens is an Anti-Money Laundering (AML) analytical system designed for transaction-pattern discovery and predictive classification on the synthetic IBM AML dataset (**HI-Small**, 5,078,345 records, 453.63 MB).

Developed as an academic Information Technology capstone project with modular, production-grade algorithms, complete FastAPI REST endpoints, a responsive light-theme React banking portal, and comprehensive viva defense documentation.

---

## 1. Algorithms Implemented

FraudLens strictly implements **exactly four algorithms** (no extraneous models like Logistic Regression, XGBoost, or Neural Networks):

1. **Standard Apriori**: Level-wise frequent itemset mining with canonical candidate generation ($L_{k-1} \times L_{k-1}$ join), downward-closure pruning, and association rule metrics (Support, Confidence, Lift).
2. **Adaptive Apriori (MSApriori)**: Solves the AML "Rare Item Dilemma" using item-specific Minimum Item Support ($MIS$), strictly preserving downward-closure pruning by sorting items in ascending $MIS$ order. Surfaces rare high-risk patterns without combinatorial candidate explosion.
3. **Genuine Utility-Based Itemset Mining (HUIM)**: Two-Phase High-Utility Itemset Mining using Transaction-Weighted Utility ($TWU$) downward-closure upper-bound pruning, using transaction monetary volume as an objective exposure proxy.
4. **Random Forest Classifier**: Out-of-time transaction classification using a chronological 70/15/15 train/val/test split, balanced class weighting (for 0.10% positive imbalance), strict target leakage prevention, and multi-metric evaluation (ROC-AUC > 0.94, PR-AUC, Confusion Matrix).

---

## 2. Quickstart & Installation (Windows 11, PowerShell)

### Step 1: Clone or Extract Repository
Ensure you are in the project root:
```powershell
cd C:\Users\surab\Downloads\Fraudlens_phase1\FraudLens
```

### Step 2: Backend Setup
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
*Note: If script execution is restricted in PowerShell, run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`*

### Step 3: Dataset Configuration
FraudLens automatically searches for `HI-Small_Trans.csv` in:
1. `FRAUDLENS_DATA_PATH` environment variable (if set)
2. `FraudLens\data\raw\HI-Small_Trans.csv`
3. Parent directory: `..\HI-Small_Trans.csv` (workspace root)

To set an explicit path:
```powershell
$env:FRAUDLENS_DATA_PATH = "C:\Users\surab\Downloads\Fraudlens_phase1\HI-Small_Trans.csv"
```

### Step 4: Frontend Setup
Open a separate PowerShell terminal:
```powershell
cd C:\Users\surab\Downloads\Fraudlens_phase1\FraudLens\frontend
npm install
```

---

## 3. Running FraudLens

### Start Backend (FastAPI)
From `FraudLens\backend`:
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger UI: **`http://127.0.0.1:8000/docs`**
- ReDoc Documentation: **`http://127.0.0.1:8000/redoc`**

### Start Frontend (React Banking Portal)
From `FraudLens\frontend`:
```powershell
npm run dev
```
- Banking Portal: **`http://localhost:5173`**
- The frontend Vite server automatically proxies `/api` calls to `http://127.0.0.1:8000`.

---

## 4. CLI Execution & Experimentation Commands

### Run Backend Unit & Integration Tests (61 / 61 Passing)
```powershell
cd FraudLens\backend
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
```

### Run Streaming Dataset Inspection
Inspects schema, class imbalance, and timestamps without loading all 453 MB into RAM:
```powershell
cd FraudLens\backend
python scripts/inspect_dataset.py --max-rows 50000
```

### Run Pattern Mining & ML Experiments (CLI)
Reproducible runs on documented bounded chronological slices:
```powershell
cd FraudLens\backend
# Run all 4 algorithms on 25,000 transactions:
python scripts/run_experiments.py --algorithm all --max-rows 25000

# Train and evaluate Random Forest on 200,000 transactions:
python scripts/run_experiments.py --algorithm rf --max-rows 200000

# Run specific mining algorithms:
python scripts/run_experiments.py --algorithm apriori --max-rows 25000
python scripts/run_experiments.py --algorithm adaptive --max-rows 25000
python scripts/run_experiments.py --algorithm huim --max-rows 25000
```

---

## 5. React Banking Portal (5 Core Pages)

The frontend is tailored for banking compliance officers and AML analysts:
1. **Banking Operations Overview**: Overall transaction monitoring KPIs, class balance breakdown (99.898% legitimate vs 0.102% laundering), active model status, and live transaction inspector.
2. **Standard Apriori (Pattern Discovery)**: Association rules between payment formats, transfer destinations, currency types, and volume tiers. Includes Support, Confidence, and Lift metrics with plain-language business explanations.
3. **Adaptive Apriori (MSApriori)**: Solves the Rare Item Dilemma by assigning customized Minimum Item Support ($MIS$) to rare high-risk indicators (`Laundering:True`, `Format:Bitcoin`).
4. **Utility Mining (HUIM)**: Two-Phase High-Utility Itemset Mining uncovering transaction combinations driving massive cumulative dollar volume throughput using Transaction-Weighted Utility ($TWU$) pruning.
5. **Random Forest Classifier**: Live single-transaction risk scoring form (with strict target leakage prevention) and model evaluation dashboard reporting Precision, Recall, F1, ROC-AUC (>0.94), PR-AUC, and Confusion Matrix.

---

## 6. Complete Documentation Suite

- **[docs/ALGORITHM_EXPLANATION.md](docs/ALGORITHM_EXPLANATION.md)**: Mathematical formulations, downward-closure proofs, pruning guarantees, complexity analysis, and Viva defense questions.
- **[docs/API_GUIDE.md](docs/API_GUIDE.md)**: Complete REST API guide with endpoint descriptions, JSON payloads, and PowerShell examples.
- **[docs/VIVA_QUESTIONS.md](docs/VIVA_QUESTIONS.md)**: 30+ categorized college viva defense questions covering theory, implementation, and ethics.
- **[docs/HANDOVER.md](docs/HANDOVER.md)**: Technical handover, system verification results, and operational checklists.
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)**: Comprehensive phase-by-phase completion status and architectural decisions.
