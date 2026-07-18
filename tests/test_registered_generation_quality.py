import json
from pathlib import Path
import shutil
import tempfile
import unittest

from train_eval.verify_registered_generation_quality import verify


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "train_eval/generation_quality_campaign_v1.json"
RESULT = ROOT / "train_eval/generation_quality_result_20260718"


class RegisteredGenerationQualityTests(unittest.TestCase):
    def test_sealed_result_verifies(self):
        record = verify(
            CAMPAIGN, RESULT, verify_runtime=False, require_current_source=False)
        self.assertEqual(record["status"], "verified")
        self.assertEqual(record["prompt_count"], 12)
        self.assertEqual(record["pool_good"]["f3"]["good"], 0)
        self.assertFalse(record["current_source_checked"])

    def test_strict_current_source_mode_detects_development_drift(self):
        with self.assertRaisesRegex(RuntimeError, "registered source drift"):
            verify(CAMPAIGN, RESULT, verify_runtime=False)

    def test_result_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "result"
            shutil.copytree(RESULT, copied)
            path = copied / "result.json"
            record = json.loads(path.read_text())
            record["pool_good"]["f3"]["good"] = 1
            path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(RuntimeError, "result hash mismatch"):
                verify(CAMPAIGN, copied, verify_runtime=False,
                       require_current_source=False)


if __name__ == "__main__":
    unittest.main()
