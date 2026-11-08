"""Exercise the actual selection function without requiring installed torch models."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple


class FakeBank:
    def __init__(self, **kwargs):
        self.num_headers = 5

    def load(self, path):
        pass

    def _validate_header_index(self, index):
        if not 0 <= index < self.num_headers:
            raise IndexError(index)

    def apply_header(self, mel, index):
        return (mel, index)


class HeaderSelectionTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "apply_header_only.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "apply_header_bank")
        module = ast.Module(body=[fn], type_ignores=[])
        namespace = dict(torch=SimpleNamespace(Tensor=object), HeaderBank=FakeBank,
                         RuleBasedHeaderSelector=object, Optional=Optional, Tuple=Tuple)
        exec(compile(module, str(path), "exec"), namespace)
        self.apply = namespace["apply_header_bank"]
        self.waveform = SimpleNamespace(detach=lambda: SimpleNamespace(cpu=lambda: "waveform"))
        self.selector = SimpleNamespace(select=lambda *a, **kw: (1, {"energy": 0.005}))

    def run_selection(self, forced=None):
        return self.apply("mel", self.waveform, "bank.pt", "cpu", 80, 5, 100, self.selector, forced)

    def test_auto_and_forced_keep_selector_provenance(self):
        _, selected, features = self.run_selection()
        self.assertEqual(selected, 1)
        self.assertEqual(features["selector_header"], 1)
        result, selected, features = self.run_selection(4)
        self.assertEqual(result, ("mel", 4))
        self.assertEqual(selected, 4)
        self.assertEqual(features["selector_header"], 1)

    def test_invalid_force_does_not_silently_clamp(self):
        for index in (-1, 5):
            with self.assertRaises(IndexError):
                self.run_selection(index)


if __name__ == "__main__":
    unittest.main()
