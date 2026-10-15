"""Genuine High-Utility Itemset Mining (HUIM) using the Two-Phase Algorithm with TWU Pruning.

Mathematical Definitions:
1. Transaction Database D = {T_1, T_2, ..., T_n}:
   Each transaction T_d has items, where each item i in T_d has an internal utility (quantity) q(i, T_d).
2. External Utility p(i):
   Unit profit, unit weight, or risk factor of item i.
3. Utility of an item in a transaction:
   u(i, T_d) = p(i) * q(i, T_d)
4. Utility of an itemset X in a transaction T_d:
   u(X, T_d) = sum_{i in X} u(i, T_d) if X is a subset of T_d, else 0
5. Utility of an itemset X in database D:
   u(X) = sum_{T_d in D} u(X, T_d)
6. Transaction Utility (TU) of transaction T_d:
   TU(T_d) = sum_{i in T_d} u(i, T_d)
7. Transaction-Weighted Utility (TWU) of itemset X:
   TWU(X) = sum_{T_d in D, X subset of T_d} TU(T_d)
8. High-Utility Itemset (HUI):
   An itemset X is high-utility if and only if u(X) >= min_utility.

TWU Downward-Closure Property & Pruning Guarantee:
- Raw utility u(X) is non-monotonic: superset Y of X can have higher utility than X.
- TWU(X) is strictly monotonic (anti-monotonic / downward closed):
  For any X subset of Y: TWU(Y) <= TWU(X).
  Furthermore, u(X) <= TWU(X) for all X.
- Pruning Theorem: If TWU(X) < min_utility, then u(Y) <= TWU(Y) <= TWU(X) < min_utility
  for ANY superset Y of X. Therefore, X and all its supersets can be safely PRUNED!

Two-Phase HUIM Algorithm (Liu et al., 2005):
- Phase 1: Level-wise generation of High Transaction-Weighted Utilization Itemsets (HTWUIs)
  using candidate generation and TWU downward-closure pruning.
- Phase 2: Exact utility verification on surviving HTWUI candidates to filter out false positives
  and return true high-utility itemsets where exact u(X) >= min_utility.

AML Domain Proxy Note:
In FraudLens, transaction amount (e.g. Amount Paid) is used as an objective quantitative
internal utility proxy q(i, T_d) representing financial transaction volume exposure.
This proxy does NOT assert that transaction amounts equal criminal profit or verified loss.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Collection, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class HighUtilityItemset:
    """Represents a discovered high-utility itemset."""
    items: frozenset[str]
    utility: float
    twu: float
    support_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": sorted(list(self.items)),
            "utility": round(self.utility, 2),
            "twu": round(self.twu, 2),
            "support_count": self.support_count,
        }


@dataclass(frozen=True)
class UtilityTransaction:
    """Represents a single transaction with per-item utilities."""
    items: frozenset[str]
    item_utilities: dict[str, float]
    tu: float  # Transaction Utility = sum of item_utilities


class TwoPhaseHUIMiner:
    """Two-Phase High-Utility Itemset Miner with TWU Upper-Bound Pruning."""

    def __init__(
        self,
        min_utility: float,
        external_utilities: Mapping[str, float] | None = None,
        max_len: int | None = None,
        max_candidates: int = 50_000,
    ) -> None:
        if min_utility <= 0:
            raise ValueError(f"min_utility must be > 0, got {min_utility}")
        if max_len is not None and max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")
        if max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {max_candidates}")

        self.min_utility = float(min_utility)
        self.external_utilities = dict(external_utilities) if external_utilities else {}
        self.max_len = max_len
        self.max_candidates = max_candidates

        self.high_utility_itemsets_: list[HighUtilityItemset] = []
        self.htwui_candidates_: list[frozenset[str]] = []
        self.total_database_utility_: float = 0.0
        self.transaction_count_: int = 0

    def _parse_input_transactions(
        self,
        transactions: Iterable[Collection[str]],
        quantities: Iterable[Mapping[str, float] | float] | None = None,
    ) -> list[UtilityTransaction]:
        """Convert input data into a standardized list of UtilityTransactions."""
        tx_list = list(transactions)
        qty_list = list(quantities) if quantities is not None else None

        if qty_list is not None and len(tx_list) != len(qty_list):
            raise ValueError(
                f"Length mismatch: {len(tx_list)} transactions but {len(qty_list)} quantity records."
            )

        parsed: list[UtilityTransaction] = []
        for idx, t in enumerate(tx_list):
            cleaned_items = [str(item).strip() for item in t if str(item).strip()]
            if not cleaned_items:
                continue

            unique_items = frozenset(cleaned_items)
            q_spec = qty_list[idx] if qty_list is not None else None

            item_utils: dict[str, float] = {}
            for item in unique_items:
                ext_u = self.external_utilities.get(item, 1.0)
                if isinstance(q_spec, Mapping):
                    qty = float(q_spec.get(item, 1.0))
                elif isinstance(q_spec, (int, float)):
                    # Uniform transaction-level quantity proxy
                    qty = float(q_spec)
                else:
                    qty = 1.0

                u_val = ext_u * qty
                if u_val > 0:
                    item_utils[item] = u_val

            tu = sum(item_utils.values())
            if tu > 0:
                parsed.append(UtilityTransaction(items=unique_items, item_utilities=item_utils, tu=tu))

        return parsed

    @staticmethod
    def _generate_candidates_k(
        prev_frequent: list[frozenset[str]],
        k: int,
    ) -> list[frozenset[str]]:
        """Join step (Lk-1 x Lk-1) to produce candidate k-itemsets."""
        candidates: set[frozenset[str]] = set()
        sorted_prev = [sorted(list(x)) for x in prev_frequent]
        n = len(sorted_prev)

        for i in range(n):
            for j in range(i + 1, n):
                itemset1 = sorted_prev[i]
                itemset2 = sorted_prev[j]
                if itemset1[: k - 2] == itemset2[: k - 2]:
                    cand = frozenset(itemset1) | frozenset(itemset2)
                    if len(cand) == k:
                        candidates.add(cand)
                else:
                    break

        return sorted(list(candidates), key=lambda x: sorted(list(x)))

    @staticmethod
    def _prune_candidates_twu(
        candidates: Iterable[frozenset[str]],
        prev_htwui_set: set[frozenset[str]],
        k: int,
    ) -> list[frozenset[str]]:
        """Apriori downward-closure pruning for TWU candidates."""
        pruned: list[frozenset[str]] = []
        for cand in candidates:
            is_valid = True
            for subset in itertools.combinations(cand, k - 1):
                if frozenset(subset) not in prev_htwui_set:
                    is_valid = False
                    break
            if is_valid:
                pruned.append(cand)
        return pruned

    def fit(
        self,
        transactions: Iterable[Collection[str]],
        quantities: Iterable[Mapping[str, float] | float] | None = None,
    ) -> TwoPhaseHUIMiner:
        """Run the Two-Phase High-Utility Itemset Mining algorithm."""
        tx_data = self._parse_input_transactions(transactions, quantities)
        self.transaction_count_ = len(tx_data)
        self.total_database_utility_ = sum(t.tu for t in tx_data)
        self.high_utility_itemsets_ = []
        self.htwui_candidates_ = []

        if self.transaction_count_ == 0:
            return self

        # ---------------------------------------------------------
        # PHASE 1: Mine High Transaction-Weighted Utilization Itemsets (HTWUIs)
        # ---------------------------------------------------------
        # Step 1.1: 1-item TWU calculation
        all_items: set[str] = set()
        for t in tx_data:
            all_items.update(t.items)

        c1 = [frozenset([item]) for item in sorted(all_items)]
        c1_twu: dict[frozenset[str], float] = {c: 0.0 for c in c1}

        for t in tx_data:
            for item in t.items:
                single = frozenset([item])
                c1_twu[single] += t.tu

        current_htwui: list[frozenset[str]] = []
        htwui_twu_map: dict[frozenset[str], float] = {}

        for cand, twu_val in c1_twu.items():
            if twu_val >= self.min_utility:
                current_htwui.append(cand)
                htwui_twu_map[cand] = twu_val
                self.htwui_candidates_.append(cand)

        k = 2
        while current_htwui:
            if self.max_len is not None and k > self.max_len:
                break

            prev_set = set(current_htwui)
            ck = self._generate_candidates_k(current_htwui, k)
            ck = self._prune_candidates_twu(ck, prev_set, k)

            if not ck:
                break

            if len(ck) > self.max_candidates:
                raise RuntimeError(
                    f"Candidate explosion at level k={k}: {len(ck)} candidates exceeded "
                    f"limit of {self.max_candidates}. Increase min_utility or set max_len."
                )

            # Compute TWU for level k candidates
            ck_twu: dict[frozenset[str], float] = {c: 0.0 for c in ck}
            for t in tx_data:
                for cand in ck:
                    if cand.issubset(t.items):
                        ck_twu[cand] += t.tu

            current_htwui = []
            for cand, twu_val in ck_twu.items():
                if twu_val >= self.min_utility:
                    current_htwui.append(cand)
                    htwui_twu_map[cand] = twu_val
                    self.htwui_candidates_.append(cand)

            k += 1

        # ---------------------------------------------------------
        # PHASE 2: Exact Utility Verification (filter out TWU overestimates)
        # ---------------------------------------------------------
        for candidate in self.htwui_candidates_:
            exact_utility = 0.0
            support_count = 0

            for t in tx_data:
                if candidate.issubset(t.items):
                    support_count += 1
                    # u(X, T_d) = sum_{i in X} u(i, T_d)
                    itemset_util_in_t = sum(t.item_utilities[item] for item in candidate)
                    exact_utility += itemset_util_in_t

            if exact_utility >= self.min_utility:
                hui = HighUtilityItemset(
                    items=candidate,
                    utility=exact_utility,
                    twu=htwui_twu_map.get(candidate, exact_utility),
                    support_count=support_count,
                )
                self.high_utility_itemsets_.append(hui)

        # Sort descending by exact utility, then length
        self.high_utility_itemsets_.sort(key=lambda x: (x.utility, len(x.items)), reverse=True)
        return self

    def get_high_utility_itemsets(self) -> list[dict[str, Any]]:
        return [hui.to_dict() for hui in self.high_utility_itemsets_]
