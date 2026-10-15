"""Service for generating downloadable CSV exports for completed mining and ML outputs."""
from __future__ import annotations

import csv
import io
import threading
from typing import Any


class ExportService:
    def __init__(self, max_exports: int = 100) -> None:
        self._exports: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._max_exports = max_exports

    def register_export(
        self,
        result_id: str,
        export_type: str,
        data: list[dict[str, Any]],
        title: str | None = None,
    ) -> str:
        with self._lock:
            if len(self._exports) >= self._max_exports and result_id not in self._exports:
                oldest = next(iter(self._exports))
                del self._exports[oldest]

            self._exports[result_id] = {
                "export_type": export_type,
                "title": title or export_type,
                "data": data,
            }
        return result_id

    def get_export_csv(self, result_id: str) -> tuple[str, str] | None:
        """Return tuple of (filename, csv_string) or None if result_id is not found."""
        with self._lock:
            entry = self._exports.get(result_id)
            if not entry:
                return None

        export_type = entry["export_type"]
        rows = entry["data"]
        filename = f"{export_type}_{result_id[:8]}.csv"

        if not rows:
            return filename, "No data available\n"

        output = io.StringIO()

        if export_type in ("apriori_rules", "adaptive_apriori_rules"):
            writer = csv.writer(output)
            writer.writerow(["Antecedent", "Consequent", "Support", "Confidence", "Lift"])
            for r in rows:
                ant = " + ".join(r.get("antecedent", []))
                con = " + ".join(r.get("consequent", []))
                writer.writerow([ant, con, r.get("support"), r.get("confidence"), r.get("lift")])

        elif export_type == "huim_itemsets":
            writer = csv.writer(output)
            writer.writerow(["Itemset", "Utility", "TWU", "Support Count"])
            for r in rows:
                items = " + ".join(r.get("items", []))
                writer.writerow([items, r.get("utility"), r.get("twu"), r.get("support_count")])

        elif export_type == "predictions":
            writer = csv.writer(output)
            writer.writerow(["Transaction Index", "Probability Score", "Predicted Class", "Is Flagged", "Risk Level"])
            for r in rows:
                writer.writerow([
                    r.get("transaction_index"),
                    r.get("probability_score"),
                    r.get("predicted_class"),
                    r.get("is_flagged"),
                    r.get("risk_level"),
                ])
        else:
            # Generic dictionary list export
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                # Format nested lists nicely
                clean_row = {}
                for k, v in r.items():
                    clean_row[k] = " + ".join(v) if isinstance(v, list) else v
                writer.writerow(clean_row)

        return filename, output.getvalue()


export_service = ExportService()
