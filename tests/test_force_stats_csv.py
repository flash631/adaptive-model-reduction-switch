import csv
import unittest
from pathlib import Path


def _parse_float(s: str) -> float | None:
    s = (s or "").strip()
    if not s or s.lower() in {"n/a", "na", "not recorded"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


class TestForceStatsCSV(unittest.TestCase):
    def test_force_stats_csv_schema_and_invariants(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / "report" / "tables" / "force_stats.csv"
        self.assertTrue(path.exists(), f"Missing canonical force-stats CSV: {path}")

        required_cols = [
            "case",
            "source",
            "Cd_mean",
            "Cd_rms",
            "Cl_mean",
            "Cl_rms",
            "f_peak_Hz",
            "St",
            "D",
            "span",
            "Aref",
            "beta",
            "U_inf",
            "Re",
        ]

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            self.assertIsNotNone(reader.fieldnames)
            self.assertEqual(reader.fieldnames, required_cols)

            rows = list(reader)

        self.assertGreater(len(rows), 0, "force_stats.csv has no data rows")

        checked_not_equal = 0
        checked_st = 0
        for r in rows:
            cl_rms = _parse_float(r["Cl_rms"])
            f_peak = _parse_float(r["f_peak_Hz"])
            if cl_rms is not None and f_peak is not None:
                self.assertGreater(abs(cl_rms - f_peak), 1e-12, f"Cl_rms equals f_peak_Hz for {r['case']} / {r['source']}")
                checked_not_equal += 1

            st = _parse_float(r["St"])
            D = _parse_float(r["D"])
            U = _parse_float(r["U_inf"])
            if st is not None and f_peak is not None and D is not None and U is not None and U != 0.0:
                self.assertLessEqual(abs(st - (f_peak * D / U)), 1e-6, f"St formula mismatch for {r['case']} / {r['source']}")
                checked_st += 1

        self.assertGreater(checked_not_equal, 0, "No rows had both Cl_rms and f_peak_Hz to validate")
        self.assertGreater(checked_st, 0, "No rows had St,f_peak_Hz,D,U_inf to validate")


if __name__ == "__main__":
    unittest.main()

