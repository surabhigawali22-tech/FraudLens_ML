"""Hand-checkable unit tests for the Standard Apriori Algorithm.
Verifies support counting, candidate generation, downward-closure pruning, and rule metrics.
"""
import unittest

from app.algorithms.apriori import AprioriMiner, AssociationRule, Itemset


class TestApriori(unittest.TestCase):
    def setUp(self):
        # Classic textbook market basket transaction dataset
        self.transactions = [
            ["bread", "milk", "beer"],
            ["bread", "diaper", "beer", "eggs"],
            ["milk", "diaper", "beer", "cola"],
            ["bread", "milk", "diaper", "beer"],
            ["bread", "milk", "diaper", "cola"],
        ]

    def test_bread_beer_pair_support_count_is_exact(self):
        """Reproduce and verify the exact support count for ('bread', 'beer').

        Transactions containing both 'bread' and 'beer':
        T0: ['bread', 'milk', 'beer'] -> YES
        T1: ['bread', 'diaper', 'beer', 'eggs'] -> YES
        T2: ['milk', 'diaper', 'beer', 'cola'] -> NO ('bread' missing)
        T3: ['bread', 'milk', 'diaper', 'beer'] -> YES
        T4: ['bread', 'milk', 'diaper', 'cola'] -> NO ('beer' missing)

        Total transactions = 5
        Occurrences of ('bread', 'beer') = exactly 3 transactions (T0, T1, T3).
        Support = 3 / 5 = 0.60.
        """
        miner = AprioriMiner(min_support=0.4, min_confidence=0.5)
        miner.fit(self.transactions)

        itemsets = {itemset.items: itemset for itemset in miner.frequent_itemsets_}
        target_pair = frozenset(["bread", "beer"])

        self.assertIn(target_pair, itemsets)
        pair_itemset = itemsets[target_pair]
        self.assertEqual(pair_itemset.support_count, 3)
        self.assertAlmostEqual(pair_itemset.support, 0.60, places=5)

    def test_three_transaction_bread_beer_test(self):
        """Specifically check 3-transaction case where ('bread', 'beer') occurs exactly twice."""
        three_tx = [
            ["bread", "milk", "beer"],           # 1: has bread, beer
            ["bread", "diaper", "beer", "eggs"],  # 2: has bread, beer
            ["milk", "diaper", "beer", "cola"],   # 3: has beer only
        ]
        miner = AprioriMiner(min_support=0.5, min_confidence=0.5)
        miner.fit(three_tx)

        itemsets = {itemset.items: itemset for itemset in miner.frequent_itemsets_}
        target_pair = frozenset(["bread", "beer"])

        self.assertIn(target_pair, itemsets)
        self.assertEqual(itemsets[target_pair].support_count, 2)
        self.assertAlmostEqual(itemsets[target_pair].support, 2 / 3, places=5)

    def test_candidate_generation_k2(self):
        """Test L1 x L1 join to produce C2 candidates."""
        l1 = [frozenset(["A"]), frozenset(["B"]), frozenset(["C"])]
        c2 = AprioriMiner._generate_candidates_k(l1, 2)
        expected = [frozenset(["A", "B"]), frozenset(["A", "C"]), frozenset(["B", "C"])]
        self.assertEqual(set(c2), set(expected))

    def test_downward_closure_pruning(self):
        """Test that candidate ABC is pruned if one of its subsets (BC) is not frequent."""
        prev_frequent_set = {frozenset(["A", "B"]), frozenset(["A", "C"])}  # missing BC
        candidates = [frozenset(["A", "B", "C"])]
        pruned = AprioriMiner._prune_candidates(candidates, prev_frequent_set, 3)
        self.assertEqual(pruned, [])  # ABC must be pruned because BC is missing

    def test_downward_closure_retention(self):
        """Test that candidate ABC is kept if all its subsets are frequent."""
        prev_frequent_set = {
            frozenset(["A", "B"]),
            frozenset(["A", "C"]),
            frozenset(["B", "C"]),
        }
        candidates = [frozenset(["A", "B", "C"])]
        pruned = AprioriMiner._prune_candidates(candidates, prev_frequent_set, 3)
        self.assertEqual(pruned, [frozenset(["A", "B", "C"])])

    def test_association_rule_metrics(self):
        """Hand-calculate metrics for rule: diaper -> beer.

        In self.transactions (N=5):
        diaper appears in T1, T2, T3, T4 (count = 4, supp = 4/5 = 0.80)
        beer appears in T0, T1, T2, T3 (count = 4, supp = 4/5 = 0.80)
        {diaper, beer} appears in T1, T2, T3 (count = 3, supp = 3/5 = 0.60)

        Rule diaper -> beer:
        Support = 0.60
        Confidence = supp({diaper, beer}) / supp({diaper}) = 0.60 / 0.80 = 0.75
        Lift = confidence / supp({beer}) = 0.75 / 0.80 = 0.9375
        """
        miner = AprioriMiner(min_support=0.5, min_confidence=0.7)
        miner.fit(self.transactions)

        rules = miner.get_rules()
        matched = [
            r for r in rules
            if set(r["antecedent"]) == {"diaper"} and set(r["consequent"]) == {"beer"}
        ]
        self.assertEqual(len(matched), 1)
        r = matched[0]
        self.assertAlmostEqual(r["support"], 0.60, places=5)
        self.assertAlmostEqual(r["confidence"], 0.75, places=5)
        self.assertAlmostEqual(r["lift"], 0.9375, places=5)

    def test_max_len_constraint(self):
        """Ensure max_len restricts itemset generation to specified length."""
        miner = AprioriMiner(min_support=0.2, max_len=1)
        miner.fit(self.transactions)
        for itemset in miner.frequent_itemsets_:
            self.assertEqual(len(itemset.items), 1)
        self.assertEqual(len(miner.rules_), 0)  # rules require length >= 2

    def test_empty_transactions(self):
        miner = AprioriMiner(min_support=0.1)
        miner.fit([])
        self.assertEqual(miner.get_frequent_itemsets(), [])
        self.assertEqual(miner.get_rules(), [])

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            AprioriMiner(min_support=0.0)
        with self.assertRaises(ValueError):
            AprioriMiner(min_support=1.5)
        with self.assertRaises(ValueError):
            AprioriMiner(min_confidence=-0.1)
        with self.assertRaises(ValueError):
            AprioriMiner(max_len=0)


if __name__ == "__main__":
    unittest.main()
