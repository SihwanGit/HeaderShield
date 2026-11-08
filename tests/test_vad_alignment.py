import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'analysis'))
from run_vad_alignment import ROOT, detect, silence, torch, np, RuleBasedHeaderSelector


class AlignmentTests(unittest.TestCase):
    def test_silent_file_is_not_forced_into_pair(self):
        onset, flags = detect(np.zeros(16000, dtype=np.float32), 2)
        self.assertIsNone(onset)
        self.assertEqual(silence(flags, 0, 4800), 1)

    def test_real_speech_with_prepended_silence(self):
        import soundfile as sf
        import torchaudio
        files = sorted((ROOT/'dataset').rglob('*.wav'))
        if not files:
            self.skipTest('Optional local speech dataset is not distributed')
        path = files[0]
        raw, sr = sf.read(path, dtype='float32', always_2d=True)
        wave = torch.from_numpy(raw.T.copy())
        if sr != 16000:
            wave = torchaudio.functional.resample(wave, sr, 16000)
        x = wave.mean(0).clamp(-1,1).numpy()
        onset, _ = detect(x, 2)
        self.assertIsNotNone(onset)
        padded = np.concatenate([np.zeros(8000, dtype=np.float32), x])
        shifted, _ = detect(padded, 2)
        self.assertGreaterEqual(shifted, 8000)
        self.assertLessEqual(abs(shifted-onset-8000), 640)
        selector = RuleBasedHeaderSelector()
        self.assertEqual(selector.select(torch.from_numpy(padded[:4800])), 1)
        a = selector.extract_features(torch.from_numpy(x[onset:]))
        b = selector.extract_features(torch.from_numpy(padded[onset+8000:]))
        self.assertEqual(a, b)


if __name__ == '__main__':
    unittest.main()
