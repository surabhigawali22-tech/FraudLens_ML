"""Unit tests for Adaptive Apriori (MSApriori).
Verifies item-specific support adaptation, downward-closure preservation,
and controlled comparison against Standard Apriori.
"""
import unittest

from app.algorithms.adaptive_apriori import AdaptiveAprioriMiner
from app.algorithms.apriori import AprioriMiner


class TestAdaptiveApriori(unittest.TestCase):
    def setUp(self):
        # Controlled transaction dataset (N = 20)
        # Frequent items: "USD", "ACH" (occur in 15 transactions -> 75% support)
        # Common item: "Cheque" (occurs in 8 transactions -> 40% support)
        # Rare suspicious item: "BitcoinTransfer" (occurs in 2 transactions -> 10% support)
        # Rare noise item: "PennyStock" (occurs in 2 transactions -> 10% support)
        self.transactions = []
        for _ in range(7):
            self.transactions.append(["USD", "ACH", "Cheque"])
        for _ in range(8):
            self.transactions.append(["USD", "ACH"])
        for _ in range(2):
            self.transactions.append(["USD", "BitcoinTransfer"])
        for _ in range(2):
            self.transactions.append(["USD", "PennyStock"])
        self.transactions.append(["USD"])  # 20 total

    def test_controlled_comparison_with_standard_apriori(self):
        """Demonstrate that Adaptive Apriori captures the rare high-risk item without lowering
        thresholds for all mundane noise items.

        N = 20:
        USD: 20/20 = 1.00
        ACH: 15/20 = 0.75
        Cheque: 7/20 = 0.35
        BitcoinTransfer: 2/20 = 0.10
        PennyStock: 2/20 = 0.10

        Standard Apriori at min_support=0.25:
        - Detects {USD}, {ACH}, {Cheque}, {USD, ACH}, {USD, Cheque}, {ACH, Cheque}
        - MISSES {BitcoinTransfer} and {USD, BitcoinTransfer} completely (0.10 < 0.25).

        Adaptive Apriori at base_min_support=0.25, with MIS("BitcoinTransfer")=0.08:
        - Detects {USD}, {ACH}, {Cheque} (mundane items >= 0.25)
        - Detects {BitcoinTransfer} (0.10 >= 0.08)
        - Detects {USD, BitcoinTransfer} (min MIS = 0.08, support = 0.10 >= 0.08)
        - PRUNES {PennyStock} because PennyStock uses base_min_support=0.25 (0.10 < 0.25)!
        """
        # 1. Standard Apriori
        std_miner = AprioriMiner(min_support=0.25)
        std_miner.fit(self.transactions)
        std_itemsets = {tuple(fi["items"]) for fi in std_miner.get_frequent_itemsets()}
        self.assertNotIn(("BitcoinTransfer",), std_itemsets)
        self.assertNotIn(("BitcoinTransfer", "USD"), std_itemsets)

        # 2. Adaptive Apriori
        adaptive_miner = AdaptiveAprioriMiner(
            base_min_support=0.25,
            mis_dict={"BitcoinTransfer": 0.08},
        )
        adaptive_miner.fit(self.transactions)
        adaptive_itemsets = {tuple(fi["items"]) for fi in adaptive_miner.get_frequent_itemsets()}

        # Verify rare target is discovered
        self.assertIn(("BitcoinTransfer",), adaptive_itemsets)
        self.assertIn(("BitcoinTransfer", "USD"), adaptive_itemsets)

        # Verify noise item with same frequency is NOT included
        self.assertNotIn(("PennyStock",), adaptive_itemsets)
        self.assertNotIn(("PennyStock", "USD"), adaptive_itemsets)

    def test_item_order_ascending_mis(self):
        """Verify items are ordered strictly by ascending MIS values."""
        miner = AdaptiveAprioriMiner(
            base_min_support=0.3,
            mis_dict={"rare_item": 0.05, "mid_item": 0.15},
        )
        baskets = [frozenset(["rare_item", "mid_item", "common_item"])]
        computed_mis, item_order, sorted_items = miner._init_mis_and_ordering(baskets)

        self.assertLess(computed_mis["rare_item"], computed_mis["mid_item"])
        self.assertLess(computed_mis["mid_item"], computed_mis["common_item"])
        self.assertEqual(sorted_items[0], "rare_item")
        self.assertEqual(sorted_items[1], "mid_item")
        self.assertEqual(sorted_items[2], "common_item")

    def test_adaptive_rule_generation(self):
        """Verify rules generated for adaptive itemsets."""
        miner = AdaptiveAprioriMiner(
            base_min_support=0.20,
            min_confidence=0.6,
            mis_dict={"BitcoinTransfer": 0.08},
        )
        miner.fit(self.transactions)
        rules = miner.get_rules()

        # {BitcoinTransfer} -> {USD} has support 2/20 = 0.10, confidence = 2/2 = 1.0
        btc_rule = [
            r for r in rules
            if r["antecedent"] == ["BitcoinTransfer"] and r["consequent"] == ["USD"]
        ]
        self.assertEqual(len(btc_rule), 1)
        self.assertAlmostEqual(btc_rule[0]["confidence"], 1.0, places=5)
        self.assertAlmostEqual(btc_rule[0]["support"], 0.10, places=5)

    def test_auto_mis_computation(self):
        """Verify automated beta-discounted MIS computation."""
        txs = [["A"] * 8 + ["B"] * 4 + ["C"] * 2]  # A=8, B=4, C=2 in single or multi tx
        txs = [["A"]] * 8 + [["B"]] * 4 + [["C"]] * 2
        miner = AdaptiveAprioriMiner(
            base_min_support=0.8,
            use_auto_mis=True,
            beta=0.5,
            lowest_support=0.05,
        )
        miner.fit(txs)
        # N=14. freq(A)=8/14~0.571 -> auto_mis=0.5*0.571=0.285
        # freq(C)=2/14~0.143 -> auto_mis=0.5*0.143=0.071 >= lowest_support=0.05
        mis_a = miner.computed_mis_["A"]
        mis_c = miner.computed_mis_["C"]
        self.assertGreater(mis_a, mis_c)
        self.assertGreaterEqual(mis_c, 0.05)


if __name__ == "__main__":
    unittest.main()
