# FraudLens: Complete REST API Guide & Integration Reference

The **FraudLens REST API** is built using FastAPI and exposes transaction-pattern mining, dataset streaming, and supervised Random Forest inference.

---

## 1. Server Architecture & Base URL

- **Default Development Host**: `http://127.0.0.1:8000`
- **Interactive OpenAPI Documentation (Swagger UI)**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`
- **API Base Prefix**: `/api`
- **Configured CORS Origins**: `http://localhost:5173`, `http://127.0.0.1:5173`, `http://localhost:3000`, `http://127.0.0.1:3000`

---

## 2. API Endpoints Reference

### 2.1 System & Health

#### `GET /api/health`
Checks backend availability, verifies the real dataset CSV location, checks for trained Random Forest model artifacts, and reports available host RAM.

**PowerShell**:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -Method Get
```

**Response (200 OK)**:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "dataset_available": true,
  "dataset_path": "C:\\Users\\surab\\Downloads\\Fraudlens_phase1\\HI-Small_Trans.csv",
  "model_available": true,
  "model_path": "C:\\Users\\surab\\Downloads\\Fraudlens_phase1\\FraudLens\\data\\artifacts\\random_forest_model.joblib",
  "free_ram_mb": null
}
```

#### `GET /api/algorithms`
Returns descriptions, categories, and readiness status for the four algorithms.

**Response (200 OK)**:
```json
{
  "algorithms": [
    {
      "id": "apriori",
      "name": "Standard Apriori",
      "description": "Level-wise frequent itemset mining...",
      "category": "Association Mining",
      "readiness_status": "ready",
      "default_parameters": { "min_support": 0.05, "min_confidence": 0.5, "max_len": 4, "sample_size": 25000 }
    },
    {
      "id": "adaptive_apriori",
      "name": "Adaptive Apriori (MSApriori)",
      "description": "Project-specific Multiple Minimum Supports Apriori...",
      "category": "Association Mining",
      "readiness_status": "ready"
    },
    {
      "id": "utility_mining",
      "name": "Two-Phase High-Utility Itemset Mining (HUIM)",
      "category": "Utility Mining",
      "readiness_status": "ready"
    },
    {
      "id": "random_forest",
      "name": "Random Forest Classifier",
      "category": "Supervised ML",
      "readiness_status": "model_available"
    }
  ]
}
```

---

### 2.2 Dataset Streaming & Inspection

#### `GET /api/dataset/summary`
Returns schema metadata, column types, row counts, and target class distribution without loading the 453 MB file fully into RAM.

**Query Parameters**:
- `force_refresh` (bool, default: `false`): Re-runs streaming inspection if `true`.

**Response (200 OK)**:
```json
{
  "path": "C:\\Users\\surab\\Downloads\\Fraudlens_phase1\\HI-Small_Trans.csv",
  "file_size_mb": 453.63,
  "row_count": 5078345,
  "columns": ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"],
  "target_column": "Is Laundering",
  "target_distribution": {
    "0": { "count": 5073168, "percent": 99.898 },
    "1": { "count": 5177, "percent": 0.102 }
  },
  "data_readiness_status": "verified"
}
```

#### `GET /api/dataset/preview`
Returns a safe, bounded preview of raw transaction records.

**Query Parameters**:
- `limit` (int, default: `10`, range: `1` to `100`).

---

### 2.3 Pattern & Utility Mining

#### `POST /api/apriori/run`
Executes Standard Apriori on a bounded sample. Supports deterministic caching.

**Request Body**:
```json
{
  "min_support": 0.08,
  "min_confidence": 0.50,
  "max_len": 3,
  "sample_size": 25000,
  "include_retrospective_target": false,
  "run_in_background": false
}
```

**Response (200 OK)**:
```json
{
  "algorithm": "Standard Apriori",
  "sample_size": 25000,
  "runtime_seconds": 0.84,
  "cached": false,
  "result_id": "7f9b09a...",
  "itemsets_count": 84,
  "rules_count": 142,
  "top_rules": [
    {
      "antecedent": ["Format:Credit Card"],
      "consequent": ["AmountBand:Low(<1k)", "Transfer:InterBank"],
      "support": 0.052,
      "confidence": 0.614,
      "lift": 6.34
    }
  ],
  "export_url": "/api/exports/7f9b09a..."
}
```

#### `POST /api/adaptive-apriori/run`
Executes Adaptive Apriori (MSApriori) using item-specific Minimum Item Support thresholds.

**Request Body**:
```json
{
  "base_min_support": 0.10,
  "min_confidence": 0.40,
  "mis_dict": {
    "Laundering:True": 0.0005,
    "Format:Bitcoin": 0.005,
    "Currency:CrossCurrency": 0.02
  },
  "max_len": 3,
  "sample_size": 25000,
  "include_retrospective_target": true
}
```

#### `POST /api/utility-mining/run`
Executes Two-Phase High-Utility Itemset Mining with Transaction-Weighted Utility (TWU) downward-closure pruning.

**Request Body**:
```json
{
  "min_utility_pct": 0.05,
  "max_len": 3,
  "sample_size": 25000
}
```

---

### 2.4 Random Forest Classifier

#### `POST /api/random-forest/train`
Trains the Random Forest model on an out-of-time chronological sample and saves the model artifact to `data/artifacts/random_forest_model.joblib`.

**Request Body**:
```json
{
  "sample_size": 50000,
  "n_estimators": 100,
  "max_depth": 15,
  "decision_threshold": 0.40,
  "train_ratio": 0.70,
  "val_ratio": 0.15,
  "run_in_background": false
}
```

#### `POST /api/random-forest/predict`
Scores incoming transaction records against the fitted Random Forest model.

> **Target Leakage Enforcement**: The schema strictly rejects any payload containing `"Is Laundering"` or case-variants with a `422 Unprocessable Entity` validation error.

**Request Body**:
```json
{
  "transactions": [
    {
      "Timestamp": "2022/09/01 15:30",
      "From Bank": 10,
      "To Bank": 20,
      "Account": "8000EBD30",
      "Account.1": "8000F5340",
      "Amount Received": 95000.0,
      "Amount Paid": 95000.0,
      "Receiving Currency": "US Dollar",
      "Payment Currency": "US Dollar",
      "Payment Format": "Wire"
    }
  ],
  "decision_threshold": 0.40
}
```

**Response (200 OK)**:
```json
{
  "total_transactions": 1,
  "flagged_count": 1,
  "threshold_used": 0.40,
  "predictions": [
    {
      "transaction_index": 0,
      "probability_score": 0.7850,
      "predicted_class": 1,
      "is_flagged": true,
      "risk_level": "high"
    }
  ],
  "disclaimer": "Model prediction scores represent tree ensemble voting proportions and ranking metrics, not calibrated real-world probabilities of money laundering or proof of guilt."
}
```

#### `GET /api/random-forest/evaluation`
Returns saved out-of-time evaluation metrics (ROC-AUC > 0.94, PR-AUC, Confusion Matrix, feature importances).

---

### 2.5 Job Tracking & Result Exports

#### `GET /api/jobs/{job_id}`
Returns background job lifecycle state: `"queued"`, `"running"`, `"completed"`, or `"failed"`.

#### `GET /api/exports/{result_id}`
Returns a downloadable CSV file attachment containing the completed association rules, high-utility itemsets, or prediction outputs.
