"""Hand-checkable unit tests for Two-Phase High-Utility Itemset Mining (HUIM).
Verifies exact utility calculation, TWU upper-bound property, downward-closure pruning,
and AML volume proxy compatibility.
"""
import unittest

from app.algorithms.utility_mining import HighUtilityItemset, TwoPhaseHUIMiner, UtilityTransaction


class TestUtilityMining(unittest.TestCase):
    def setUp(self):
        # Textbook utility dataset:
        # External utilities (unit profit/risk weight):
        self.external_utilities = {
            "A": 5.0,
            "B": 2.0,
            "C": 1.0,
            "D": 2.0,
            "E": 3.0,
        }

        # Transactions with item quantities
        self.transactions = [
            ["A", "C", "D"],
            ["A", "C", "E"],
            ["B", "C", "D", "E"],
            ["B", "C", "D"],
            ["B", "C", "E"],
        ]

        self.quantities = [
            {"A": 1, "C": 1, "D": 1},      # T1: 5*1 + 1*1 + 2*1 = 8
            {"A": 2, "C": 6, "E": 2},      # T2: 5*2 + 1*6 + 3*2 = 22
            {"B": 4, "C": 2, "D": 3, "E": 1},  # T3: 2*4 + 1*2 + 2*3 + 3*1 = 19
            {"B": 2, "C": 2, "D": 1},      # T4: 2*2 + 1*2 + 2*1 = 8
            {"B": 1, "C": 2, "E": 2},      # T5: 2*1 + 1*2 + 3*2 = 10
        ]
        # Total DB Utility = 8 + 22 + 19 + 8 + 10 = 67.0

    def test_known_high_utility_itemsets_and_totals(self):
        """Verify exact utility values match mathematical derivation for min_utility = 20.0."""
        miner = TwoPhaseHUIMiner(
            min_utility=20.0,
            external_utilities=self.external_utilities,
        )
        miner.fit(self.transactions, self.quantities)

        results = {tuple(h["items"]): h for h in miner.get_high_utility_itemsets()}

        # Hand-calculated high-utility itemsets:
        # {B, D}: in T3(14) + T4(6) = 20.0
        self.assertIn(("B", "D"), results)
        self.assertEqual(results[("B", "D")]["utility"], 20.0)

        # {B, C, D}: in T3(16) + T4(8) = 24.0
        self.assertIn(("B", "C", "D"), results)
        self.assertEqual(results[("B", "C", "D")]["utility"], 24.0)

        # {B, C, E}: in T3(13) + T5(10) = 23.0
        self.assertIn(("B", "C", "E"), results)
        self.assertEqual(results[("B", "C", "E")]["utility"], 23.0)

        # {A, C}: in T1(6) + T2(16) = 22.0
        self.assertIn(("A", "C"), results)
        self.assertEqual(results[("A", "C")]["utility"], 22.0)

        # Verify single items (all utility < 20.0) are NOT returned as high utility
        for single in [("A",), ("B",), ("C",), ("D",), ("E",)]:
            self.assertNotIn(single, results)

    def test_twu_upper_bound_property(self):
        """Verify the fundamental theorem: u(X) <= TWU(X) for all itemsets X."""
        miner = TwoPhaseHUIMiner(
            min_utility=1.0,  # capture all itemsets
            external_utilities=self.external_utilities,
        )
        miner.fit(self.transactions, self.quantities)

        for hui in miner.high_utility_itemsets_:
            self.assertLessEqual(
                hui.utility,
                hui.twu + 1e-6,
                f"Violation of TWU upper-bound property for {hui.items}: u={hui.utility} > twu={hui.twu}",
            )

    def test_twu_downward_closure_monotonicity(self):
        """Verify TWU monotonicity: if X is subset of Y, then TWU(Y) <= TWU(X)."""
        miner = TwoPhaseHUIMiner(
            min_utility=5.0,
            external_utilities=self.external_utilities,
        )
        miner.fit(self.transactions, self.quantities)
        twu_map = {hui.items: hui.twu for hui in miner.high_utility_itemsets_}

        bcd = frozenset(["B", "C", "D"])
        bd = frozenset(["B", "D"])

        if bcd in twu_map and bd in twu_map:
            self.assertLessEqual(twu_map[bcd], twu_map[bd])

    def test_aml_transaction_volume_proxy(self):
        """Verify AML transaction-level volume proxy (uniform amount per transaction)."""
        aml_tx = [
            ["Wire", "USD", "HighRiskJurisdiction"],
            ["ACH", "USD"],
            ["Wire", "USD", "HighRiskJurisdiction"],
        ]
        # Amounts: T1 = $50,000, T2 = $500, T3 = $40,000
        amounts = [50000.0, 500.0, 40000.0]

        miner = TwoPhaseHUIMiner(min_utility=80000.0)
        miner.fit(aml_tx, quantities=amounts)

        huis = miner.get_high_utility_itemsets()
        self.assertGreater(len(huis), 0)

        # Wire + HighRiskJurisdiction appears in T1 & T3 ($90,000 total volume)
        wire_hr = next(
            (h for h in huis if set(h["items"]) == {"HighRiskJurisdiction", "Wire"}),
            None,
        )
        self.assertIsNotNone(wire_hr)
        # In T1 (50k each item) + T3 (40k each item) -> itemset utility = (50k+50k) + (40k+40k) = 180,000
        self.assertGreaterEqual(wire_hr["utility"], 80000.0)

    def test_empty_transactions(self):
        miner = TwoPhaseHUIMiner(min_utility=10.0)
        miner.fit([])
        self.assertEqual(miner.get_high_utility_itemsets(), [])

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            TwoPhaseHUIMiner(min_utility=-5.0)
        with self.assertRaises(ValueError):
            TwoPhaseHUIMiner(min_utility=0.0)
        with self.assertRaises(ValueError):
            TwoPhaseHUIMiner(min_utility=10.0, max_len=0)


if __name__ == "__main__":
    unittest.main()
