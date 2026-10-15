"""API endpoints for pattern and utility mining algorithms."""
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

from app.algorithms.adaptive_apriori import AdaptiveAprioriMiner
from app.algorithms.apriori import AprioriMiner
from app.algorithms.utility_mining import TwoPhaseHUIMiner
from app.core.cache import global_cache
from app.core.config import get_dataset_path
from app.data import loader
from app.data.features import build_association_baskets
from app.schemas.common import JobResponse
from app.schemas.mining import (
    AdaptiveAprioriRequest,
    AprioriRequest,
    HUIMRequest,
    MiningItemsetItem,
    MiningResponse,
    MiningRuleItem,
)
from app.services.export_service import export_service
from app.services.job_service import job_service

router = APIRouter(tags=["Pattern & Utility Mining"])


def _execute_apriori(req_dict: dict[str, Any]) -> dict[str, Any]:
    csv_path = get_dataset_path()
    sample_size = req_dict.get("sample_size", 25_000)
    df = next(loader.iter_chunks(csv_path, chunksize=sample_size, max_rows=sample_size))

    start = time.time()
    baskets, _ = build_association_baskets(
        df,
        include_retrospective_target=req_dict.get("include_retrospective_target", False),
    )
    miner = AprioriMiner(
        min_support=req_dict["min_support"],
        min_confidence=req_dict["min_confidence"],
        max_len=req_dict.get("max_len"),
    )
    miner.fit(baskets)
    elapsed = time.time() - start

    rules = miner.get_rules()
    itemsets = miner.get_frequent_itemsets()
    result_id = uuid.uuid4().hex

    # Register export
    export_service.register_export(result_id, "apriori_rules", rules, title="Standard Apriori Rules")

    return {
        "algorithm": "Standard Apriori",
        "sample_size": len(df),
        "runtime_seconds": round(elapsed, 4),
        "cached": False,
        "result_id": result_id,
        "itemsets_count": len(itemsets),
        "rules_count": len(rules),
        "top_rules": rules[:50],
        "top_itemsets": itemsets[:50],
        "export_url": f"/api/exports/{result_id}",
        "parameters": req_dict,
    }


def _execute_adaptive_apriori(req_dict: dict[str, Any]) -> dict[str, Any]:
    csv_path = get_dataset_path()
    sample_size = req_dict.get("sample_size", 25_000)
    df = next(loader.iter_chunks(csv_path, chunksize=sample_size, max_rows=sample_size))

    start = time.time()
    baskets, _ = build_association_baskets(
        df,
        include_retrospective_target=req_dict.get("include_retrospective_target", True),
    )

    domain_mis = req_dict.get("mis_dict") or {
        "Laundering:True": 0.0005,
        "Format:Bitcoin": 0.005,
        "Currency:CrossCurrency": 0.02,
        "AmountBand:VeryHigh(>50k)": 0.02,
    }

    miner = AdaptiveAprioriMiner(
        base_min_support=req_dict["base_min_support"],
        min_confidence=req_dict["min_confidence"],
        mis_dict=domain_mis,
        use_auto_mis=req_dict.get("use_auto_mis", False),
        beta=req_dict.get("beta", 0.5),
        lowest_support=req_dict.get("lowest_support", 0.005),
        max_len=req_dict.get("max_len"),
    )
    miner.fit(baskets)
    elapsed = time.time() - start

    rules = miner.get_rules()
    itemsets = miner.get_frequent_itemsets()
    result_id = uuid.uuid4().hex

    export_service.register_export(result_id, "adaptive_apriori_rules", rules, title="Adaptive Apriori Rules")

    return {
        "algorithm": "Adaptive Apriori (MSApriori)",
        "sample_size": len(df),
        "runtime_seconds": round(elapsed, 4),
        "cached": False,
        "result_id": result_id,
        "itemsets_count": len(itemsets),
        "rules_count": len(rules),
        "top_rules": rules[:50],
        "top_itemsets": itemsets[:50],
        "export_url": f"/api/exports/{result_id}",
        "parameters": req_dict,
    }


def _execute_huim(req_dict: dict[str, Any]) -> dict[str, Any]:
    csv_path = get_dataset_path()
    sample_size = req_dict.get("sample_size", 25_000)
    df = next(loader.iter_chunks(csv_path, chunksize=sample_size, max_rows=sample_size))

    start = time.time()
    baskets, amounts = build_association_baskets(df, include_retrospective_target=False)
    total_volume = sum(amounts)

    min_u = req_dict.get("min_utility")
    if min_u is None:
        pct = req_dict.get("min_utility_pct", 0.05) or 0.05
        min_u = max(1.0, pct * total_volume)

    miner = TwoPhaseHUIMiner(
        min_utility=min_u,
        max_len=req_dict.get("max_len", 3),
    )
    miner.fit(baskets, quantities=amounts)
    elapsed = time.time() - start

    huis = miner.get_high_utility_itemsets()
    result_id = uuid.uuid4().hex

    export_service.register_export(result_id, "huim_itemsets", huis, title="High Utility Itemsets")

    return {
        "algorithm": "Two-Phase High-Utility Itemset Mining (HUIM)",
        "sample_size": len(df),
        "runtime_seconds": round(elapsed, 4),
        "cached": False,
        "result_id": result_id,
        "itemsets_count": len(huis),
        "rules_count": 0,
        "top_rules": [],
        "top_itemsets": huis[:50],
        "export_url": f"/api/exports/{result_id}",
        "parameters": {**req_dict, "effective_min_utility": min_u, "total_sample_volume": total_volume},
    }


@router.post("/apriori/run", response_model=MiningResponse | JobResponse)
def run_apriori(req: AprioriRequest) -> Any:
    csv_path = get_dataset_path()
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset CSV not found at {csv_path}.")

    req_dict = req.model_dump()
    cache_key = global_cache.generate_key("apriori", req_dict, csv_path)
    cached = global_cache.get(cache_key)
    if cached:
        cached_resp = dict(cached)
        cached_resp["cached"] = True
        return MiningResponse(**cached_resp)

    if req.run_in_background:
        job_id = job_service.submit_job("apriori", _execute_apriori, req_dict)
        job_info = job_service.get_job(job_id)
        return JobResponse(**job_info)

    try:
        res = _execute_apriori(req_dict)
        global_cache.set(cache_key, res)
        return MiningResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Standard Apriori execution failed: {e}")


@router.post("/adaptive-apriori/run", response_model=MiningResponse | JobResponse)
def run_adaptive_apriori(req: AdaptiveAprioriRequest) -> Any:
    csv_path = get_dataset_path()
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset CSV not found at {csv_path}.")

    req_dict = req.model_dump()
    cache_key = global_cache.generate_key("adaptive_apriori", req_dict, csv_path)
    cached = global_cache.get(cache_key)
    if cached:
        cached_resp = dict(cached)
        cached_resp["cached"] = True
        return MiningResponse(**cached_resp)

    if req.run_in_background:
        job_id = job_service.submit_job("adaptive_apriori", _execute_adaptive_apriori, req_dict)
        job_info = job_service.get_job(job_id)
        return JobResponse(**job_info)

    try:
        res = _execute_adaptive_apriori(req_dict)
        global_cache.set(cache_key, res)
        return MiningResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Adaptive Apriori execution failed: {e}")


@router.post("/utility-mining/run", response_model=MiningResponse | JobResponse)
def run_utility_mining(req: HUIMRequest) -> Any:
    csv_path = get_dataset_path()
    if not csv_path.is_file():
        raise HTTPException(status_code=404, detail=f"Dataset CSV not found at {csv_path}.")

    req_dict = req.model_dump()
    cache_key = global_cache.generate_key("huim", req_dict, csv_path)
    cached = global_cache.get(cache_key)
    if cached:
        cached_resp = dict(cached)
        cached_resp["cached"] = True
        return MiningResponse(**cached_resp)

    if req.run_in_background:
        job_id = job_service.submit_job("utility_mining", _execute_huim, req_dict)
        job_info = job_service.get_job(job_id)
        return JobResponse(**job_info)

    try:
        res = _execute_huim(req_dict)
        global_cache.set(cache_key, res)
        return MiningResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Two-Phase HUIM execution failed: {e}")
