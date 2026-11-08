import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prepare_headerbank_evaluation import main


class PrepareTests(unittest.TestCase):
    def test_export_all_variants_and_copy_identical_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data/pgirl1/0001.wav"
            bank = root / "banks/girl1/header_bank.pt"
            source.parent.mkdir(parents=True)
            bank.parent.mkdir(parents=True)
            source.write_bytes(b"source")
            bank.write_bytes(b"model")
            output = root / "export"
            argv = ["prepare", "--source-root", str(root / "data"), "--bank-root", str(root / "banks"),
                    "--output-root", str(output), "--speakers", "girl1"]
            with patch("sys.argv", argv), patch("prepare_headerbank_evaluation.subprocess.run") as run:
                main()
            self.assertEqual(run.call_count, 6)  # Selector + forced 0..4
            self.assertEqual((output / "source/pgirl1/0001.wav").read_bytes(), source.read_bytes())
            forced = [call.args[0][call.args[0].index("--force_header_index") + 1]
                      for call in run.call_args_list if "--force_header_index" in call.args[0]]
            self.assertEqual(forced, ["0", "1", "2", "3", "4"])
            with patch("sys.argv", argv), patch("prepare_headerbank_evaluation.subprocess.run") as run:
                with self.assertRaises(SystemExit):
                    main()
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
