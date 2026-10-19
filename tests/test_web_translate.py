import shutil
import subprocess
import unittest
from pathlib import Path


class TestFrontend(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for frontend regression tests")
    def test_subtitle_message_transitions(self):
        result = subprocess.run(
            [shutil.which("node"), str(Path(__file__).with_suffix(".js"))],
            text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
