from pathlib import Path
import unittest

from waterint.io.xyz import read_xyz


class XYZReaderTests(unittest.TestCase):
    def test_reads_multiframe_xyz(self):
        path = Path(__file__).resolve().parents[1] / "examples/density_xyz/input/input.xyz"
        frames = list(read_xyz(path))
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].symbols[0], "O")
        self.assertEqual(frames[0].positions.shape, (9, 3))


if __name__ == "__main__":
    unittest.main()
