# FraudLens: Technical Handover & System Verification (Phases 1–8)

## 1. Executive Summary
- **Project**: FraudLens — Anti-Money Laundering (AML) Pattern Mining & Classification System
- **Status**: Phases 1–8 Completed & 100% Verified
- **Test Suite Status**: **61 / 61 Tests Passing (100% Pass Rate)**
- **Verified Dataset**: IBM AML Synthetic Dataset (`HI-Small_Trans.csv`, 453.63 MB, 5,078,345 records)
- **Active Zip Archive**: `FraudLens_latest.zip` (verified byte-for-byte in workspace root)
- **Baseline Archives**: `FraudLens_baseline.zip` (Phase 1-5), `FraudLens_baseline_phase6.zip`

---

## 2. Verified File & Project Structure

```
FraudLens/
├── backend/
│   ├── app/
│   │   ├── algorithms/
│   │   │   ├── __init__.py
│   │   │   ├── adaptive_apriori.py     # Phase 3: Multiple Minimum Supports Apriori
│   │   │   ├── apriori.py              # Phase 2: Standard Level-wise Apriori
│   │   │   ├── random_forest.py        # Phase 5: Leak-free chronological RF pipeline (smart holdout boundary)
│   │   │   └── utility_mining.py       # Phase 4: Two-Phase High-Utility Itemset Mining
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes_algorithms.py    # GET /api/algorithms registry
│   │   │   ├── routes_dataset.py       # GET /api/dataset/summary & preview
│   │   │   ├── routes_exports.py       # GET /api/exports/{result_id} CSV stream
│   │   │   ├── routes_health.py        # GET /api/health
│   │   │   ├── routes_jobs.py          # GET /api/jobs and GET /api/jobs/{id}
│   │   │   ├── routes_mining.py        # POST /api/apriori, /adaptive-apriori, /utility-mining
│   │   │   └── routes_rf.py            # POST /api/random-forest/train, predict, GET evaluation
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── cache.py                # Deterministic SHA-256 result cache
│   │   │   └── config.py               # Paths, CORS, and resource limits
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── features.py             # Feature extraction (baskets vs tabular matrix)
│   │   │   ├── inspection.py           # Streaming dataset schema/imbalance inspector
│   │   │   └── loader.py               # Chunked CSV streaming generator
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── common.py               # Health, Job, and Error schemas
│   │   │   ├── dataset.py              # DatasetSummary and Preview schemas
│   │   │   ├── mining.py               # Apriori, Adaptive, HUIM schemas
│   │   │   └── random_forest.py        # Train, Predict, Evaluation schemas
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── dataset_service.py      # Cached dataset statistics
│   │   │   ├── export_service.py       # CSV generation and attachment streaming
│   │   │   └── job_service.py          # Asynchronous background task manager
│   │   └── main.py                     # FastAPI application factory with CORS
│   ├── scripts/
│   │   ├── inspect_dataset.py          # CLI streaming dataset inspector
│   │   └── run_experiments.py          # CLI experiment runner for all 4 algorithms
│   ├── tests/
│   │   ├── test_adaptive_apriori.py    # 4 unit tests
│   │   ├── test_api_dataset.py         # 4 API integration tests
│   │   ├── test_api_health.py          # 3 API integration tests
│   │   ├── test_api_jobs_exports.py    # 5 API integration tests
│   │   ├── test_api_mining.py          # 6 API integration tests
│   │   ├── test_api_rf.py              # 5 API integration tests (isolated disk mocks)
│   │   ├── test_apriori.py             # 9 unit tests
│   │   ├── test_inspection.py          # 8 unit tests
│   │   ├── test_random_forest.py       # 11 unit tests (focused regressions)
│   │   └── test_utility_mining.py      # 6 unit tests
│   └── requirements.txt
├── data/
│   ├── artifacts/
│   │   └── random_forest_model.joblib  # Trained model serialized artifact
│   ├── outputs/
│   │   ├── dataset_inspection.json     # Saved streaming inspection results
│   │   ├── experiment_results_all.json # Documented 25k multi-algorithm run
│   │   └── experiment_results_rf.json  # Documented 200k RF evaluation run
│   └── raw/
├── docs/
│   ├── ALGORITHM_EXPLANATION.md        # Comprehensive mathematical formulation & proofs
│   ├── API_GUIDE.md                    # REST API endpoint reference & JSON samples
│   ├── HANDOVER.md                     # Technical handover & system verification
│   └── VIVA_QUESTIONS.md               # 30+ Viva defense questions and answers
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx              # Navigation header with active indicators
│   │   │   └── TransactionModal.jsx    # Drill-down transaction inspection modal
│   │   ├── pages/
│   │   │   ├── AdaptiveAprioriPage.jsx # Rare pattern mining (MSApriori)
│   │   │   ├── AprioriPage.jsx         # Level-wise association rules & explanations
│   │   │   ├── OverviewPage.jsx        # Compliance monitoring KPIs & transaction table
│   │   │   ├── RandomForestPage.jsx    # Live scoring form & out-of-time evaluation (sample selector)
│   │   │   └── UtilityMiningPage.jsx   # Two-Phase HUIM with TWU dollar volume exposure
│   │   ├── services/
│   │   │   └── api.js                  # Axios/Fetch API client wrapper
│   │   ├── App.jsx                     # Top-level state and tab routing
│   │   ├── index.css                   # Cohesive light-theme banking design system
│   │   └── main.jsx                    # React entrypoint
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── PROJECT_STATUS.md
└── README.md
```

---

## 3. Verified End-to-End Experimental Results

All experiments were executed on real data from `HI-Small_Trans.csv`:

| Algorithm | Sample Size | Execution Time | Findings / Output |
| :--- | :---: | :---: | :--- |
| **Standard Apriori** | 25,000 rows | 2.95s | 342 itemsets, 1503 rules ($min\_supp=0.05, min\_conf=0.5$). Top Lift: 3.81. |
| **Adaptive Apriori** | 25,000 rows | 2.63s | 250 itemsets, 1339 rules ($base\_supp=0.10, rare\_MIS=0.0005$). Prunes mundane pairs. |
| **Two-Phase HUIM** | 25,000 rows | 3.40s | 109 High-Utility Itemsets ($min\_utility=\$1.09B$, Total Volume=\$21.9B). |
| **Random Forest (Fast)** | 50,000 rows | 0.35s | Train=35,000, Val=5,621, Test=9,379. **ROC-AUC: 0.4995**, CM={'tn': 9378, 'fp': 0, 'fn': 1, 'tp': 0}. |
| **Random Forest (Mid)** | 100,000 rows | 0.54s | Train=70,000, Val=15,000, Test=15,000. **ROC-AUC: 0.7193**, PR-AUC: 0.0020. |
| **Random Forest (Full)** | 200,000 rows | 1.19s | Train=140,000, Val=30,000, Test=30,000. **ROC-AUC: 0.9443**, PR-AUC: 0.0006. |

---

## 4. Windows PowerShell Startup Instructions

### 4.1 Backend (Terminal 1)
```powershell
cd C:\Users\surab\Downloads\Fraudlens_phase1\FraudLens\backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- Swagger UI: `http://127.0.0.1:8000/docs`

### 4.2 Frontend (Terminal 2)
```powershell
cd C:\Users\surab\Downloads\Fraudlens_phase1\FraudLens\frontend
npm run dev
```
- Banking Portal: `http://localhost:5173`

### 4.3 Running Verification Tests
```powershell
cd C:\Users\surab\Downloads\Fraudlens_phase1\FraudLens\backend
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -v
```

---

## 5. Key Compliance & Academic Guardrails

1. **Human-in-the-Loop Language**: Flagged transactions or high-confidence rules represent *prioritized indicators for human compliance review*, never legal proof of money laundering or criminal guilt.
2. **Strict Target Isolation**: The target column `Is Laundering` is completely excluded from feature generation and is rejected with HTTP 422 if submitted to inference endpoints.
3. **No Metric Fabrication**: All reported metrics (ROC-AUC 0.9443, 5,078,345 rows, 0.102% positive class) derive directly from empirical computation on `HI-Small_Trans.csv`.
4. **Chronological Splitting**: Random Forest training uses an out-of-time chronological train/val/test split to mirror realistic production conditions.

---

## 6. Backup Archive Location
- **Archive Path**: `C:\Users\surab\Downloads\Fraudlens_phase1\FraudLens_latest.zip`
- **Integrity**: Every source file, configuration, test, and documentation file is verified byte-for-byte using `backup_tool.py`. Raw CSV files, virtual environments, and node modules are properly excluded.
