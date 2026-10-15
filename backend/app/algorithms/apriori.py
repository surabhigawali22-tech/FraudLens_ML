"""Standard Apriori Algorithm for Association Rule Mining.

Implements the classic level-wise Apriori algorithm with:
- Itemset representation as canonical frozensets.
- Level-wise candidate generation (Lk-1 x Lk-1 join) and Apriori downward-closure pruning.
- Exact support counting against transaction database.
- Association rule generation with support, confidence, and lift metrics.
- Safeguards against combinatorial explosion (max_len, max_candidates).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Collection, Iterable, Sequence


@dataclass(frozen=True)
class Itemset:
    """Represents a frequent itemset."""
    items: frozenset[str]
    support_count: int
    support: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": sorted(list(self.items)),
            "support_count": self.support_count,
            "support": round(self.support, 6),
        }


@dataclass(frozen=True)
class AssociationRule:
    """Represents an association rule X -> Y."""
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


class AprioriMiner:
    """Standard Apriori frequent itemset and association rule miner."""

    def __init__(
        self,
        min_support: float = 0.05,
        min_confidence: float = 0.5,
        max_len: int | None = None,
        max_candidates: int = 50_000,
    ) -> None:
        if not (0.0 < min_support <= 1.0):
            raise ValueError(f"min_support must be in (0.0, 1.0], got {min_support}")
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be in [0.0, 1.0], got {min_confidence}")
        if max_len is not None and max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")
        if max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {max_candidates}")

        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_len = max_len
        self.max_candidates = max_candidates

        self.frequent_itemsets_: list[Itemset] = []
        self.rules_: list[AssociationRule] = []
        self.itemset_support_map_: dict[frozenset[str], float] = {}
        self.transaction_count_: int = 0

    @staticmethod
    def _to_transaction_sets(transactions: Iterable[Collection[str]]) -> list[frozenset[str]]:
        """Convert input transactions into a list of frozensets, stripping empty items."""
        baskets: list[frozenset[str]] = []
        for t in transactions:
            cleaned = frozenset(str(item).strip() for item in t if str(item).strip())
            if cleaned:
                baskets.append(cleaned)
        return baskets

    def _generate_c1(self, transactions: list[frozenset[str]]) -> list[frozenset[str]]:
        """Generate 1-item candidates from the unique items in all transactions."""
        items: set[str] = set()
        for t in transactions:
            items.update(t)
        return [frozenset([item]) for item in sorted(items)]

    @staticmethod
    def _generate_candidates_k(
        prev_frequent: list[frozenset[str]],
        k: int,
    ) -> list[frozenset[str]]:
        """Generate candidate k-itemsets by joining frequent (k-1)-itemsets (Lk-1 x Lk-1 join).

        Two itemsets of length k-1 are joined if they share their first k-2 items
        when sorted in canonical order.
        """
        candidates: set[frozenset[str]] = set()
        sorted_prev = [sorted(list(itemset)) for itemset in prev_frequent]
        n = len(sorted_prev)

        for i in range(n):
            for j in range(i + 1, n):
                itemset1 = sorted_prev[i]
                itemset2 = sorted_prev[j]
                # Check if first k-2 items match
                if itemset1[: k - 2] == itemset2[: k - 2]:
                    # Union creates a candidate of length k
                    candidate = frozenset(itemset1) | frozenset(itemset2)
                    if len(candidate) == k:
                        candidates.add(candidate)
                else:
                    # Because itemsets are sorted, once prefixes diverge we can't join with
                    # further elements having different prefixes for itemset1
                    break

        return sorted(list(candidates), key=lambda x: sorted(list(x)))

    @staticmethod
    def _prune_candidates(
        candidates: Iterable[frozenset[str]],
        prev_frequent_set: set[frozenset[str]],
        k: int,
    ) -> list[frozenset[str]]:
        """Prune candidates whose (k-1)-subsets are not in Lk-1 (Apriori downward-closure property)."""
        pruned: list[frozenset[str]] = []
        for candidate in candidates:
            # Check all (k-1) subsets
            is_valid = True
            for subset in itertools.combinations(candidate, k - 1):
                if frozenset(subset) not in prev_frequent_set:
                    is_valid = False
                    break
            if is_valid:
                pruned.append(candidate)
        return pruned

    def fit(self, transactions: Iterable[Collection[str]]) -> AprioriMiner:
        """Mine frequent itemsets and generate association rules from transactions."""
        baskets = self._to_transaction_sets(transactions)
        self.transaction_count_ = len(baskets)
        self.frequent_itemsets_ = []
        self.rules_ = []
        self.itemset_support_map_ = {}

        if self.transaction_count_ == 0:
            return self

        min_count = self.min_support * self.transaction_count_

        # Step 1: L1 Frequent 1-itemsets
        c1 = self._generate_c1(baskets)
        current_l: list[frozenset[str]] = []

        # Count support for C1
        c1_counts: dict[frozenset[str], int] = {c: 0 for c in c1}
        for basket in baskets:
            for item in basket:
                single = frozenset([item])
                if single in c1_counts:
                    c1_counts[single] += 1

        for itemset, count in c1_counts.items():
            support = count / self.transaction_count_
            if count >= min_count:
                itemset_obj = Itemset(items=itemset, support_count=count, support=support)
                self.frequent_itemsets_.append(itemset_obj)
                self.itemset_support_map_[itemset] = support
                current_l.append(itemset)

        # Step 2: Level-wise mining for k >= 2
        k = 2
        while current_l:
            if self.max_len is not None and k > self.max_len:
                break

            prev_set = set(current_l)
            # Candidate generation
            ck = self._generate_candidates_k(current_l, k)
            # Downward closure pruning
            ck = self._prune_candidates(ck, prev_set, k)

            if not ck:
                break

            if len(ck) > self.max_candidates:
                raise RuntimeError(
                    f"Candidate explosion at level k={k}: {len(ck)} candidates exceeded "
                    f"limit of {self.max_candidates}. Increase min_support or set max_len."
                )

            # Support counting across all transactions
            ck_counts: dict[frozenset[str], int] = {cand: 0 for cand in ck}
            for basket in baskets:
                # Optimized check: candidate must be subset of basket
                for cand in ck:
                    if cand.issubset(basket):
                        ck_counts[cand] += 1

            current_l = []
            for cand, count in ck_counts.items():
                support = count / self.transaction_count_
                if count >= min_count:
                    itemset_obj = Itemset(items=cand, support_count=count, support=support)
                    self.frequent_itemsets_.append(itemset_obj)
                    self.itemset_support_map_[cand] = support
                    current_l.append(cand)

            k += 1

        # Step 3: Generate association rules
        self._generate_rules()
        return self

    def _generate_rules(self) -> None:
        """Generate association rules from frequent itemsets with len >= 2."""
        self.rules_ = []
        for itemset_obj in self.frequent_itemsets_:
            itemset = itemset_obj.items
            if len(itemset) < 2:
                continue

            itemset_support = itemset_obj.support

            # Generate all non-empty proper subsets as antecedents
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
                        rule = AssociationRule(
                            antecedent=antecedent,
                            consequent=consequent,
                            support=itemset_support,
                            confidence=confidence,
                            lift=lift,
                        )
                        self.rules_.append(rule)

        # Sort rules descending by lift, then confidence
        self.rules_.sort(key=lambda r: (r.lift, r.confidence, r.support), reverse=True)

    def get_frequent_itemsets(self) -> list[dict[str, Any]]:
        """Return frequent itemsets formatted as dictionaries."""
        return [fi.to_dict() for fi in self.frequent_itemsets_]

    def get_rules(self) -> list[dict[str, Any]]:
        """Return association rules formatted as dictionaries."""
        return [rule.to_dict() for rule in self.rules_]
