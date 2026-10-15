"""Data transformation and representation module for FraudLens.

Provides separate, dedicated representations for:
1. Association & Utility Mining (categorical transaction baskets and monetary utilities)
2. Random Forest Classification (tabular numeric/categorical feature matrix with strict leakage prevention)
"""
from __future__ import annotations

from typing import Any, Tuple
import numpy as np
import pandas as pd


def categorize_amount(amount: float) -> str:
    """Bin transaction amount into interpretable financial volume tiers."""
    if amount < 1_000:
        return "AmountBand:Low(<1k)"
    elif amount < 10_000:
        return "AmountBand:Medium(1k-10k)"
    elif amount < 50_000:
        return "AmountBand:High(10k-50k)"
    else:
        return "AmountBand:VeryHigh(>50k)"


def categorize_hour(hour: int) -> str:
    """Bin transaction hour into time-of-day segments."""
    if 0 <= hour < 6:
        return "TimeBand:Night(00-06)"
    elif 6 <= hour < 12:
        return "TimeBand:Morning(06-12)"
    elif 12 <= hour < 18:
        return "TimeBand:Afternoon(12-18)"
    else:
        return "TimeBand:Evening(18-24)"


def build_association_baskets(
    df: pd.DataFrame,
    include_retrospective_target: bool = False,
) -> tuple[list[list[str]], list[float]]:
    """Convert raw AML transaction DataFrame into association mining baskets and transaction utilities.

    Args:
        df: DataFrame containing verified columns:
            Timestamp, From Bank, Account, To Bank, Account.1,
            Amount Received, Receiving Currency, Amount Paid, Payment Currency,
            Payment Format, Is Laundering.
        include_retrospective_target: If True, adds 'Target:Laundering' or 'Target:Legitimate'
            strictly for retrospective association rule analysis. This is NEVER passed to ML models.

    Returns:
        tuple of (baskets, amounts)
        - baskets: list of categorical item lists
        - amounts: list of transaction amount utilities (Amount Paid)
    """
    baskets: list[list[str]] = []
    amounts: list[float] = []

    # Parse timestamps for time band
    ts_series = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M", errors="coerce")

    # Determine bank relationship and currency relationship
    is_same_bank = df["From Bank"] == df["To Bank"]
    is_same_curr = df["Payment Currency"] == df["Receiving Currency"]
    paid_amounts = df["Amount Paid"].astype(float).values

    for idx, row in df.reset_index(drop=True).iterrows():
        basket: list[str] = []

        # 1. Payment format
        fmt = str(row["Payment Format"]).strip()
        if fmt:
            basket.append(f"Format:{fmt}")

        # 2. Payment currency
        p_curr = str(row["Payment Currency"]).strip()
        if p_curr:
            basket.append(f"PaymentCurr:{p_curr}")

        # 3. Currency relationship (cross-currency vs domestic)
        if is_same_curr.iloc[idx]:
            basket.append("Currency:SingleCurrency")
        else:
            basket.append("Currency:CrossCurrency")

        # 4. Bank relationship (internal transfer vs interbank)
        if is_same_bank.iloc[idx]:
            basket.append("Transfer:InternalBank")
        else:
            basket.append("Transfer:InterBank")

        # 5. Amount tier
        amt = float(paid_amounts[idx])
        basket.append(categorize_amount(amt))

        # 6. Time band
        ts = ts_series.iloc[idx]
        if pd.notna(ts):
            basket.append(categorize_hour(ts.hour))

        # Optional retrospective label
        if include_retrospective_target and "Is Laundering" in row:
            label = int(row["Is Laundering"])
            basket.append("Laundering:True" if label == 1 else "Laundering:False")

        baskets.append(basket)
        amounts.append(max(0.01, amt))

    return baskets, amounts


def build_rf_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract tabular feature matrix X and target y for Random Forest classification.

    Guarantees:
    - Target 'Is Laundering' is strictly isolated into y and NEVER present in X.
    - No label-derived features are included in X (strict target leakage prevention).
    - Unnecessary post-event and non-generalizable raw string identifiers (Account strings)
      are excluded.
    """
    if "Is Laundering" not in df.columns:
        raise ValueError("Target column 'Is Laundering' is required in DataFrame.")

    y = df["Is Laundering"].astype(int).copy()

    # Create engineering features
    X = pd.DataFrame(index=df.index)

    # 1. Numeric features
    amt_paid = df["Amount Paid"].astype(float)
    amt_recv = df["Amount Received"].astype(float)

    X["amount_paid"] = amt_paid
    X["amount_received"] = amt_recv
    X["amount_diff"] = (amt_paid - amt_recv).abs()
    X["amount_ratio"] = amt_recv / (amt_paid + 1e-5)
    X["log_amount_paid"] = np.log1p(np.maximum(0.0, amt_paid))

    # 2. Bank network features
    from_bank = df["From Bank"].astype(int)
    to_bank = df["To Bank"].astype(int)
    X["from_bank"] = from_bank
    X["to_bank"] = to_bank
    X["is_same_bank"] = (from_bank == to_bank).astype(int)

    # 3. Account relationship features (without leaking high-cardinality ID strings)
    # Checks whether transaction is to the same account (self-transfer)
    acc_from = df["Account"].astype(str)
    acc_to = df["Account.1"].astype(str) if "Account.1" in df.columns else df.iloc[:, 4].astype(str)
    X["is_same_account"] = (acc_from == acc_to).astype(int)

    # 4. Temporal features from Timestamp
    ts_series = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M", errors="coerce")
    X["hour"] = ts_series.dt.hour.fillna(0).astype(int)
    X["day_of_week"] = ts_series.dt.dayofweek.fillna(0).astype(int)
    X["is_weekend"] = X["day_of_week"].isin([5, 6]).astype(int)

    # 5. Categorical relationship features
    curr_paid = df["Payment Currency"].astype(str)
    curr_recv = df["Receiving Currency"].astype(str)
    X["is_cross_currency"] = (curr_paid != curr_recv).astype(int)
    X["payment_format"] = df["Payment Format"].astype(str)
    X["payment_currency"] = curr_paid

    return X, y
