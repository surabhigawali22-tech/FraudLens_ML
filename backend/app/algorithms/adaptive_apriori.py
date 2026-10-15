"""Adaptive Apriori Algorithm for Association Rule Mining (MSApriori).

Implements the Multiple Minimum Supports Apriori (MSApriori) strategy designed
to solve the "Rare Item Problem" in Anti-Money Laundering (AML) pattern mining.

Mathematical Foundation:
- Standard Apriori imposes a uniform global min_support, causing rare patterns
  (e.g., suspicious high-risk transfers) to be missed unless min_support is set
  extremely low, which triggers combinatorial explosion on mundane items.
- In MSApriori (Liu, Hsu, Ma, KDD 1999):
  1. Each item i has an individual Minimum Item Support MIS(i).
  2. For an itemset X = {i_1, ..., i_k}, min_support(X) = min_{i in X} MIS(i).
  3. Items are sorted in ascending order of MIS: MIS(i_1) <= MIS(i_2) <= ... <= MIS(i_m).
  4. For any sorted itemset X, min_support(X) = MIS(i_1).
  5. Valid pruning rule: Any (k-1)-subset containing i_1 must be frequent under MIS(i_1).
     Subsets excluding i_1 have higher thresholds and are not used to prune.
  This preserves mathematical correctness and downward-closure under item-level adaptation.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Collection, Iterable


@dataclass(frozen=True)
class AdaptiveItemset:
    """Represents a frequent itemset under adaptive minimum item support."""
    items: frozenset[str]
    support_count: int
    support: float
    min_support_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": sorted(list(self.items)),
            "support_count": self.support_count,
            "support": round(self.support, 6),
            "min_support_threshold": round(self.min_support_threshold, 6),
        }


@dataclass(frozen=True)
class AdaptiveAssociationRule:
    """Represents an association rule X -> Y under adaptive mining."""
    antecedent: frozenset[str]
    consequent: frozenset[str]
    support: float
    confidence: float
    lift: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "antecedent": sorted(list(self.antecedent)),
            "consequent": sorted(list(self.consequent)),
            "support": round(self.support, 6),
            "confidence": round(self.confidence, 6),
            "lift": round(self.lift, 6),
        }


class AdaptiveAprioriMiner:
    """Adaptive Apriori algorithm with item-specific minimum item support (MSApriori)."""

    def __init__(
        self,
        base_min_support: float = 0.05,
        min_confidence: float = 0.5,
        mis_dict: dict[str, float] | None = None,
        use_auto_mis: bool = False,
        beta: float = 0.5,
        lowest_support: float = 0.005,
        max_len: int | None = None,
        max_candidates: int = 50_000,
    ) -> None:
        """Initialize Adaptive Apriori.

        Args:
            base_min_support: Default MIS for items not explicitly specified.
            min_confidence: Minimum rule confidence threshold.
            mis_dict: Optional manual mapping of item -> MIS value.
            use_auto_mis: If True, automatically compute MIS for unspecified items
                          using beta * freq(i). If False, unspecified items use base_min_support.
            beta: Support-difference scaling factor for auto-computed MIS:
                  MIS(i) = max(lowest_support, beta * freq(i)).
            lowest_support: Absolute floor for auto-computed MIS.
            max_len: Maximum length of itemsets to discover.
            max_candidates: Maximum candidate threshold per level to prevent OOM.
        """
        if not (0.0 < base_min_support <= 1.0):
            raise ValueError(f"base_min_support must be in (0.0, 1.0], got {base_min_support}")
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be in [0.0, 1.0], got {min_confidence}")
        if not (0.0 < beta <= 1.0):
            raise ValueError(f"beta must be in (0.0, 1.0], got {beta}")
        if not (0.0 < lowest_support <= 1.0):
            raise ValueError(f"lowest_support must be in (0.0, 1.0], got {lowest_support}")
        if max_len is not None and max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")
        if max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {max_candidates}")

        self.base_min_support = base_min_support
        self.min_confidence = min_confidence
        self.mis_dict = dict(mis_dict) if mis_dict else {}
        self.use_auto_mis = use_auto_mis
        self.beta = beta
        self.lowest_support = lowest_support
        self.max_len = max_len
        self.max_candidates = max_candidates

        self.frequent_itemsets_: list[AdaptiveItemset] = []
        self.rules_: list[AdaptiveAssociationRule] = []
        self.itemset_support_map_: dict[frozenset[str], float] = {}
        self.computed_mis_: dict[str, float] = {}
        self.sorted_items_: list[str] = []
        self.item_order_: dict[str, int] = {}
        self.transaction_count_: int = 0

    @staticmethod
    def _to_transaction_sets(transactions: Iterable[Collection[str]]) -> list[frozenset[str]]:
        baskets: list[frozenset[str]] = []
        for t in transactions:
            cleaned = frozenset(str(item).strip() for item in t if str(item).strip())
            if cleaned:
                baskets.append(cleaned)
        return baskets

    def _init_mis_and_ordering(
        self,
        baskets: list[frozenset[str]],
    ) -> tuple[dict[str, float], dict[str, int], list[str]]:
        """Compute MIS for all items and sort items by ascending MIS.

        Items with identical MIS are sorted lexicographically to guarantee determinism.
        """
        n = len(baskets)
        raw_counts: dict[str, int] = {}
        for b in baskets:
            for item in b:
                raw_counts[item] = raw_counts.get(item, 0) + 1

        computed_mis: dict[str, float] = {}
        for item, count in raw_counts.items():
            if item in self.mis_dict:
                computed_mis[item] = float(self.mis_dict[item])
            elif self.use_auto_mis:
                # Adaptive formula: max(LS, beta * freq(i)) capped at base_min_support
                freq = count / n
                auto_mis = max(self.lowest_support, self.beta * freq)
                computed_mis[item] = min(auto_mis, self.base_min_support)
            else:
                computed_mis[item] = self.base_min_support

        # Sort items in ascending order of MIS; secondary sort by item name
        sorted_items = sorted(computed_mis.keys(), key=lambda x: (computed_mis[x], x))
        item_order = {item: idx for idx, item in enumerate(sorted_items)}
        return computed_mis, item_order, sorted_items

    def _min_support_for_itemset(self, itemset: frozenset[str]) -> float:
        """Threshold for an itemset is the minimum MIS among its items."""
        return min(self.computed_mis_.get(item, self.base_min_support) for item in itemset)

    def _sort_itemset(self, itemset: Iterable[str]) -> list[str]:
        """Sort an itemset according to the global MIS ascending order."""
        return sorted(list(itemset), key=lambda x: (self.item_order_.get(x, 999999), x))

    def _generate_candidate_k(
        self,
        prev_frequent: list[list[str]],
        k: int,
    ) -> list[list[str]]:
        """Join step for MSApriori on MIS-ordered itemsets."""
        candidates: list[list[str]] = []
        n = len(prev_frequent)

        for i in range(n):
            for j in range(i + 1, n):
                itemset1 = prev_frequent[i]
                itemset2 = prev_frequent[j]

                # Check if first k-2 items match
                if itemset1[: k - 2] == itemset2[: k - 2]:
                    item1_last = itemset1[-1]
                    item2_last = itemset2[-1]
                    # Since itemsets are sorted, item1_last must precede item2_last
                    if self.item_order_[item1_last] < self.item_order_[item2_last]:
                        cand = itemset1 + [item2_last]
                        candidates.append(cand)
                else:
                    break
        return candidates

    def _prune_candidate_k(
        self,
        candidates: list[list[str]],
        prev_frequent_sets: set[frozenset[str]],
        k: int,
    ) -> list[list[str]]:
        """Downward-closure pruning under MSApriori.

        For candidate c = [i_1, i_2, ..., i_k], every (k-1) subset s MUST be frequent
        EXCEPT possibly the subset c \\ {i_1}.
        Why? Because c \\ {i_1} has minimum MIS = MIS(i_2) >= MIS(i_1), which is a higher
        threshold; hence c \\ {i_1} being infrequent under MIS(i_2) does not prove
        c is infrequent under MIS(i_1).
        Subsets containing i_1 share the exact same threshold MIS(i_1); if any of them
        is not frequent, c CANNOT be frequent under downward closure.
        """
        pruned: list[list[str]] = []
        for cand in candidates:
            i1 = cand[0]
            is_valid = True
            for subset in itertools.combinations(cand, k - 1):
                # If subset contains the minimum MIS item i1, it MUST be in prev_frequent_sets
                if i1 in subset:
                    if frozenset(subset) not in prev_frequent_sets:
                        is_valid = False
                        break
            if is_valid:
                pruned.append(cand)
        return pruned

    def fit(self, transactions: Iterable[Collection[str]]) -> AdaptiveAprioriMiner:
        """Mine frequent itemsets using MSApriori."""
        baskets = self._to_transaction_sets(transactions)
        self.transaction_count_ = len(baskets)
        self.frequent_itemsets_ = []
        self.rules_ = []
        self.itemset_support_map_ = {}

        if self.transaction_count_ == 0:
            return self

        # Step 1: Compute MIS and sort items
        self.computed_mis_, self.item_order_, self.sorted_items_ = self._init_mis_and_ordering(baskets)

        # Count 1-item frequencies
        c1_counts: dict[str, int] = {item: 0 for item in self.sorted_items_}
        for basket in baskets:
            for item in basket:
                if item in c1_counts:
                    c1_counts[item] += 1

        # Step 2: L1 - frequent 1-itemsets
        # In MSApriori, an item i is frequent if support(i) >= MIS(i)
        l1_items: list[str] = []
        for item in self.sorted_items_:
            count = c1_counts[item]
            supp = count / self.transaction_count_
            mis = self.computed_mis_[item]
            if supp >= mis:
                l1_items.append(item)
                itemset = frozenset([item])
                self.frequent_itemsets_.append(
                    AdaptiveItemset(
                        items=itemset,
                        support_count=count,
                        support=supp,
                        min_support_threshold=mis,
                    )
                )
                self.itemset_support_map_[itemset] = supp

        # Step 3: Level-wise mining for k >= 2
        current_l: list[list[str]] = [[item] for item in l1_items]
        k = 2

        while current_l:
            if self.max_len is not None and k > self.max_len:
                break

            prev_frequent_sets = {frozenset(x) for x in current_l}
            ck = self._generate_candidate_k(current_l, k)
            ck = self._prune_candidate_k(ck, prev_frequent_sets, k)

            if not ck:
                break

            if len(ck) > self.max_candidates:
                raise RuntimeError(
                    f"Candidate explosion at level k={k}: {len(ck)} candidates exceeded "
                    f"limit of {self.max_candidates}. Increase base_min_support or lowest_support."
                )

            # Count support
            ck_frozen = [frozenset(c) for c in ck]
            ck_counts = {c: 0 for c in ck_frozen}
            for basket in baskets:
                for cand in ck_frozen:
                    if cand.issubset(basket):
                        ck_counts[cand] += 1

            current_l = []
            for cand_list, cand_frozen in zip(ck, ck_frozen):
                count = ck_counts[cand_frozen]
                supp = count / self.transaction_count_
                threshold = self._min_support_for_itemset(cand_frozen)

                if supp >= threshold:
                    self.frequent_itemsets_.append(
                        AdaptiveItemset(
                            items=cand_frozen,
                            support_count=count,
                            support=supp,
                            min_support_threshold=threshold,
                        )
                    )
                    self.itemset_support_map_[cand_frozen] = supp
                    current_l.append(cand_list)

            k += 1

        # Step 4: Rule generation
        self._generate_rules()
        return self

    def _generate_rules(self) -> None:
        """Generate association rules under adaptive framework."""
        self.rules_ = []
        for itemset_obj in self.frequent_itemsets_:
            itemset = itemset_obj.items
            if len(itemset) < 2:
                continue

            itemset_support = itemset_obj.support
            items_list = list(itemset)

            for r in range(1, len(items_list)):
                for antecedent_tuple in itertools.combinations(items_list, r):
                    antecedent = frozenset(antecedent_tuple)
                    consequent = itemset - antecedent

                    antecedent_support = self.itemset_support_map_.get(antecedent, 0.0)
                    consequent_support = self.itemset_support_map_.get(consequent, 0.0)

                    if antecedent_support <= 0.0 or consequent_support <= 0.0:
                        continue

                    confidence = itemset_support / antecedent_support
                    if confidence >= self.min_confidence:
                        lift = confidence / consequent_support
                        rule = AdaptiveAssociationRule(
                            antecedent=antecedent,
                            consequent=consequent,
                            support=itemset_support,
                            confidence=confidence,
                            lift=lift,
                        )
                        self.rules_.append(rule)

        self.rules_.sort(key=lambda r: (r.lift, r.confidence, r.support), reverse=True)

    def get_frequent_itemsets(self) -> list[dict[str, Any]]:
        return [fi.to_dict() for fi in self.frequent_itemsets_]

    def get_rules(self) -> list[dict[str, Any]]:
        return [rule.to_dict() for rule in self.rules_]
