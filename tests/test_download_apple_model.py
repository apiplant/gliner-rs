"""Small offline checks for the pinned-checkpoint integrity rules."""

import hashlib
import io
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import download_apple_model as downloader


class DownloadIntegrityTests(unittest.TestCase):
    def test_lfs_payload_uses_sha256_not_pointer_blob_id(self):
        payload = b"safetensors payload"
        pointer_blob = b"version https://git-lfs.github.com/spec/v1\n" + b"oid sha256:00\n"
        pointer_id = hashlib.sha1(f"blob {len(pointer_blob)}\0".encode() + pointer_blob).hexdigest()
        expected = {
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "blobId": pointer_id,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "model.safetensors"
            with patch.object(downloader, "urlopen", return_value=nullcontext(io.BytesIO(payload))):
                downloader.download_file("owner/model", "revision", "model.safetensors", target, expected)
            self.assertEqual(target.read_bytes(), payload)

    def test_regular_git_blob_is_verified_by_git_blob_id(self):
        payload = b'{"config":true}\n'
        blob_id = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        expected = {"size": len(payload), "sha256": None, "blobId": blob_id}
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "config.json"
            with patch.object(downloader, "urlopen", return_value=nullcontext(io.BytesIO(payload))):
                downloader.download_file("owner/model", "revision", "config.json", target, expected)
            self.assertEqual(target.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
