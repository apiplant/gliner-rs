#!/usr/bin/env python3
"""Download a pinned GLiNER2.5 checkpoint for the Apple port."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


MODELS = {
    "small": ("fastino/gliner2.5-small-v1", "df5910e44bc4ffdb0d95399a83b0ca4516349aa5"),
    "base": ("fastino/gliner2.5-base-v1", "54785d51df8d86a0d8f2eb212fec15871aad2c4d"),
    "multi": ("fastino/gliner2.5-multi-v1", "cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3"),
}
EXPECTED_FILES = (
    "config.json",
    "encoder_config/config.json",
    "tokenizer.json",
    "model.safetensors",
)
TIMEOUT_SECONDS = 120
CHUNK_SIZE = 1024 * 1024
USER_AGENT = "gliner-apple-model-fetch/1.0"


def request(url):
    return Request(url, headers={"User-Agent": USER_AGENT})


def get_metadata(repo, revision):
    url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    with urlopen(request(url), timeout=TIMEOUT_SECONDS) as response:
        data = json.load(response)
    siblings = {item.get("rfilename"): item for item in data.get("siblings", [])}
    result = {}
    for name in EXPECTED_FILES:
        item = siblings.get(name)
        if item is None:
            raise RuntimeError(f"Hugging Face revision is missing required file: {name}")
        lfs = item.get("lfs") or {}
        size = lfs.get("size", item.get("size"))
        digest = lfs.get("sha256")
        blob_id = item.get("blobId")
        if not isinstance(size, int) or size < 0:
            raise RuntimeError(f"Hugging Face API did not provide a valid size for {name}")
        if digest is not None and (not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None):
            raise RuntimeError(f"Hugging Face API returned an invalid LFS SHA-256 for {name}")
        # Small config/tokenizer files may be ordinary Git blobs, not LFS.
        # In that case the pinned revision's Git blob ID verifies the bytes;
        # the SHA-256 is still recorded after download in checkpoint.json.
        if digest is None and (not isinstance(blob_id, str) or re.fullmatch(r"[0-9a-fA-F]{40}", blob_id) is None):
            raise RuntimeError(f"Hugging Face API provided neither an LFS SHA-256 nor a Git blob ID for {name}")
        result[name] = {"size": size, "sha256": digest.lower() if digest else None,
                        "blobId": blob_id.lower() if blob_id else None}
    return result


def digest_file(path):
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(CHUNK_SIZE)
            if not block:
                break
            size += len(block)
            sha256.update(block)
    return size, sha256.hexdigest()


def is_verified(path, expected):
    if not path.is_file():
        return False
    size, sha256 = digest_file(path)
    if size != expected["size"]:
        return False
    if expected["sha256"] is not None:
        return sha256 == expected["sha256"]
    return git_blob_id(path, size) == expected["blobId"]


def git_blob_id(path, size):
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        while True:
            block = stream.read(CHUNK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download_file(repo, revision, name, destination, expected):
    url_name = quote(name, safe="/")
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{url_name}?download=true"
    part = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(request(url), timeout=TIMEOUT_SECONDS) as response, part.open("wb") as output:
            sha256 = hashlib.sha256()
            git_sha1 = hashlib.sha1(f"blob {expected['size']}\0".encode("ascii"))
            size = 0
            while True:
                block = response.read(CHUNK_SIZE)
                if not block:
                    break
                output.write(block)
                sha256.update(block)
                git_sha1.update(block)
                size += len(block)
        if size != expected["size"]:
            raise RuntimeError(f"{name}: downloaded {size} bytes, expected {expected['size']}")
        actual_digest = sha256.hexdigest()
        if expected["sha256"] is not None and actual_digest != expected["sha256"]:
            raise RuntimeError(f"{name}: SHA-256 mismatch (expected {expected['sha256']}, got {actual_digest})")
        actual_blob_id = git_sha1.hexdigest()
        # The API's blobId for LFS files identifies the small Git pointer blob,
        # not the downloaded LFS payload. Verify Git IDs only for non-LFS files.
        if expected["sha256"] is None and expected["blobId"] is not None and actual_blob_id != expected["blobId"]:
            raise RuntimeError(f"{name}: Git blob ID mismatch (expected {expected['blobId']}, got {actual_blob_id})")
        os.replace(part, destination)
    except Exception:
        try:
            part.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path, value):
    temporary = path.with_name(path.name + ".part")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(MODELS), default="small")
    parser.add_argument("--output", type=Path, help="checkpoint directory (default: .models/<repo-basename>)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    repo, revision = MODELS[args.variant]
    output = args.output or (Path.cwd() / ".models" / repo.rsplit("/", 1)[-1])
    output.mkdir(parents=True, exist_ok=True)
    try:
        metadata = get_metadata(repo, revision)
        for name in EXPECTED_FILES:
            target = output / name
            expected = metadata[name]
            if is_verified(target, expected):
                print(f"verified, skipping {name}")
                continue
            print(f"downloading {name}")
            download_file(repo, revision, name, target, expected)
        actual_files = {}
        for name in EXPECTED_FILES:
            size, sha256 = digest_file(output / name)
            actual_files[name] = {"size": size, "sha256": sha256}
        checkpoint = {"repo": repo, "revision": revision, "files": actual_files}
        atomic_write_json(output / "checkpoint.json", checkpoint)
    except (HTTPError, URLError, TimeoutError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"model download failed: {exc}", file=sys.stderr)
        return 1
    print(f"checkpoint ready: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
