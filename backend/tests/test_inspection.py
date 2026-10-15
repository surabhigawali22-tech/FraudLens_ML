"""Tests use a tiny HAND-WRITTEN CSV (test fixture only, not dataset results).
Run from backend/:  python -m unittest discover -s tests -v
"""
import tempfile
import unittest
from pathlib import Path

from app.data import loader
from app.data.inspection import find_target_column, inspect_csv

HEADER = ("Timestamp,From Bank,Account,To Bank,Account,Amount Received,"
          "Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering")
ROWS = [
    "2022/09/01 00:20,10,A1,20,B1,100.0,US Dollar,100.0,US Dollar,Cheque,0",
    "2022/09/01 00:21,10,A2,20,B2,50.5,Euro,55.0,US Dollar,ACH,0",
    "2022/09/01 01:00,11,A3,11,A4,9000.0,US Dollar,9000.0,US Dollar,Wire,1",
    "2022/09/01 02:00,12,A5,13,B5,,Euro,20.0,Euro,Credit Card,0",   # one missing value
    "2022/09/02 03:00,12,A6,13,B6,75.0,Euro,75.0,Euro,Wire,1",
]


def write_csv(d, header=HEADER, rows=ROWS) -> Path:
    p = Path(d) / "tiny.csv"
    p.write_text("\n".join([header] + rows) + "\n", encoding="utf-8")
    return p


class InspectionTests(unittest.TestCase):
    def test_row_count_and_target_distribution(self):
        with tempfile.TemporaryDirectory() as d:
            r = inspect_csv(write_csv(d), chunksize=2)
        self.assertEqual(r["row_count"], 5)
        self.assertEqual(r["target_column"], "Is Laundering")
        self.assertEqual(r["target_distribution"]["1"]["count"], 2)
        self.assertEqual(r["target_distribution"]["0"]["count"], 3)

    def test_chunk_size_does_not_change_results(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_csv(d)
            a, b = inspect_csv(p, chunksize=1), inspect_csv(p, chunksize=1000)
        for k in ("row_count", "missing_values", "target_distribution"):
            self.assertEqual(a[k], b[k])

    def test_missing_values_counted(self):
        with tempfile.TemporaryDirectory() as d:
            r = inspect_csv(write_csv(d), chunksize=2)
        self.assertEqual(r["missing_values"]["Amount Received"], 1)
        self.assertEqual(r["missing_values"]["Amount Paid"], 0)

    def test_duplicate_account_header_reported(self):
        with tempfile.TemporaryDirectory() as d:
            r = inspect_csv(write_csv(d))
        self.assertEqual(r["duplicate_header_names"], ["Account"])
        self.assertIn("Account.1", r["pandas_columns"])

    def test_target_missing_is_reported_not_invented(self):
        header = HEADER.replace(",Is Laundering", "")
        rows = [x.rsplit(",", 1)[0] for x in ROWS]
        with tempfile.TemporaryDirectory() as d:
            r = inspect_csv(write_csv(d, header, rows))
        self.assertIsNone(r["target_column"])
        self.assertIsNone(r["target_distribution"])

    def test_max_rows_partial_run(self):
        with tempfile.TemporaryDirectory() as d:
            r = inspect_csv(write_csv(d), chunksize=2, max_rows=3)
        self.assertEqual(r["row_count"], 3)

    def test_missing_file_raises_clear_error(self):
        with self.assertRaises(loader.DatasetNotFoundError):
            loader.resolve_path("definitely/not/here.csv")

    def test_find_target_column_variants(self):
        self.assertEqual(find_target_column(["x", "is_laundering"]), "is_laundering")
        self.assertIsNone(find_target_column(["x", "y"]))


if __name__ == "__main__":
    unittest.main()
