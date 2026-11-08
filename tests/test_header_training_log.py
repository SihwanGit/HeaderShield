import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from utils.header_training_log import HeaderTrainingLog, slice_metadata


class HeaderTrainingLogTests(unittest.TestCase):
    def test_assignments_are_unique_but_borrowed_updates_are_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            log = HeaderTrainingLog(SimpleNamespace(output_dir=directory, num_headers=5))
            features = dict(energy=0.02, zero_crossing_rate=0.1, spectral_centroid=1000)
            first = dict(slice_id=0, source_file="source.wav", start_sample=0,
                         end_sample=16000, start_sec=0, end_sec=1)
            second = dict(first, slice_id=1, start_sample=6400, end_sample=22400,
                          start_sec=0.4, end_sec=1.4)
            log.assignment(first, 1, 1, features)
            log.assignment(second, 2, 2, features)
            log.assignment(first, 1, 1, features)  # Another epoch must not inflate distribution.
            log.training(1, 1, first, 1, [(3, 0.1, 2.0, "borrowed"), (0, 0.2, 4.0, "default")], 10 / 3)
            log.summary("epoch-end", 1)
            summary = json.loads((log.directory / "summary.json").read_text())
            self.assertEqual(summary["unique_classified_slices"], 2)
            self.assertEqual(summary["headers"][1]["assigned_percent"], 50)
            self.assertEqual(summary["headers"][3]["assigned_percent"], 0)
            self.assertEqual(summary["headers"][3]["update_percent"], 50)
            self.assertEqual(summary["headers"][0]["mean_loss"], 4)
            self.assertIsNone(summary["headers"][4]["mean_loss"])
            with (log.directory / "header_assignment_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                distribution = list(csv.DictReader(stream))
            self.assertEqual(len(distribution), 5)
            self.assertEqual(distribution[1]["assigned_slices"], "1")
            self.assertEqual(float(distribution[1]["assigned_percent"]), 50)
            self.assertEqual(distribution[3]["assigned_slices"], "0")
            self.assertEqual(distribution[3]["total_slices"], "2")
            with (log.directory / "slice_assignments.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)
            with (log.directory / "training_steps.csv").open(encoding="utf-8-sig", newline="") as stream:
                updates = list(csv.DictReader(stream))
            self.assertEqual(updates[0]["sample_type"], "borrowed")
            self.assertAlmostEqual(sum(float(row["normalized_weight"]) for row in updates), 1)

    def test_empty_summary_and_separate_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=directory, num_headers=6)
            first, second = HeaderTrainingLog(args), HeaderTrainingLog(args)
            self.assertNotEqual(first.directory, second.directory)
            first.summary("empty")
            result = json.loads((first.directory / "summary.json").read_text())
            self.assertTrue(all(row["assigned_percent"] == 0 for row in result["headers"]))
            with (first.directory / "header_assignment_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.assertTrue(all(float(row["assigned_percent"]) == 0 for row in csv.DictReader(stream)))

    def test_resampled_positions(self):
        batch = dict(source_position=[6400], source_slice_id=[7], source_file=["source.wav"],
                     source_waveform=[SimpleNamespace(shape=(1, 26400))])
        metadata = slice_metadata(batch, 0, 16000)
        self.assertEqual(metadata["start_sec"], 0.4)
        self.assertEqual(metadata["end_sec"], 2.05)
        self.assertEqual(metadata["slice_id"], 7)


if __name__ == "__main__":
    unittest.main()
