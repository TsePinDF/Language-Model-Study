from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tmt.jobs import start_job


class JobSecurityTests(unittest.TestCase):
    def test_environment_values_are_not_persisted(self) -> None:
        token = "hf_secret_test_value"
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir)
            with (
                mock.patch("tmt.jobs.JOB_DIR", job_dir),
                mock.patch("tmt.jobs.subprocess.Popen") as popen,
            ):
                job = start_job(
                    "dataset",
                    [sys.executable, "scripts/download_dataset.py", "owner/name"],
                    {"dataset": "owner/name"},
                    environment={"HF_TOKEN": token},
                )

            persisted_text = (job_dir / f"{job['job_id']}.json").read_text(encoding="utf-8")
            persisted = json.loads(persisted_text)
            process_environment = popen.call_args.kwargs["env"]

        self.assertNotIn(token, persisted_text)
        self.assertEqual(persisted["environment_keys"], ["HF_TOKEN"])
        self.assertEqual(process_environment["HF_TOKEN"], token)


if __name__ == "__main__":
    unittest.main()
