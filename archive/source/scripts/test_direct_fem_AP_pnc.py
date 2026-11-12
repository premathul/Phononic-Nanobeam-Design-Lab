"""Dependency-light internal checks for the direct FEM result artifacts."""
import csv, math, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class DirectFEMTests(unittest.TestCase):
    def test_workbook_validation_table(self):
        rows=list(csv.DictReader(open(ROOT/'results/tables/hole_validation.csv')))
        self.assertEqual(len(rows),26); self.assertTrue(all(r['validation_status']=='PASS' for r in rows))
    def test_full_sweep(self):
        rows=list(csv.DictReader(open(ROOT/'results/tables/frequency_sweep.csv')))
        self.assertEqual(len(rows),111); self.assertTrue(all(r['solver_status']=='PASS' for r in rows))
        self.assertTrue(all(math.isfinite(float(r['relative_transmission_dB'])) for r in rows))
        self.assertTrue(all(float(r['solver_residual_PnC'])<1e-7 and float(r['solver_residual_reference'])<1e-7 for r in rows))
    def test_five_frequency_table(self):
        rows=list(csv.DictReader(open(ROOT/'results/tables/five_frequency_transmission.csv')))
        self.assertEqual([float(r['frequency_GHz']) for r in rows],[2.,4.,6.,8.,10.])
        self.assertTrue(all(r['solver_status']=='PASS' for r in rows))
if __name__=='__main__': unittest.main(verbosity=2)
