from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from scripts.download_dataset import default_output_dir, format_bytes, parse_dataset_id


class DownloadDatasetTests(unittest.TestCase):
    def test_parses_dataset_id(self) -> None:
        self.assertEqual(parse_dataset_id("Open-Orca/OpenOrca"), "Open-Orca/OpenOrca")

    def test_parses_dataset_url(self) -> None:
        self.assertEqual(
            parse_dataset_id("https://huggingface.co/datasets/Open-Orca/OpenOrca"),
            "Open-Orca/OpenOrca",
        )

    def test_ignores_url_page_suffix(self) -> None:
        self.assertEqual(
            parse_dataset_id("https://huggingface.co/datasets/Open-Orca/OpenOrca/tree/main"),
            "Open-Orca/OpenOrca",
        )

    def test_rejects_non_hugging_face_url(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_dataset_id("https://example.com/datasets/Open-Orca/OpenOrca")

    def test_default_output_dir(self) -> None:
        self.assertEqual(
            default_output_dir("Open-Orca/OpenOrca"),
            Path("data/huggingface/Open-Orca--OpenOrca"),
        )

    def test_formats_bytes(self) -> None:
        self.assertEqual(format_bytes(0), "0.0 B")
        self.assertEqual(format_bytes(1536), "1.5 KiB")
        self.assertEqual(format_bytes(4 * 1024**3), "4.0 GiB")


if __name__ == "__main__":
    unittest.main()
