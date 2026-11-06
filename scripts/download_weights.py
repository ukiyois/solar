"""Resolve and verify the author's actual HF filenames, including README typo.

Downloads the ImageNet-22K-to-1K FlashInternImage-L checkpoint of the DCNv4 authors from the Hugging Face
repository OpenGVLab/DCNv4. The file is looked up by its actual name through the Hugging Face API and pinned to
the repository revision; the download resumes through <name>.part with HTTP range requests and is accepted
only if size and SHA-256 match the LFS metadata. <name>.manifest.json records the provenance. For the default
file, whose SHA-256 is also pinned here, the tensor-only backbone weights/flash_intern_image_l_backbone.pt
(classification head removed) is extracted; training loads it with weights_only=True. Inference does not
need it, because the trained checkpoint contains the backbone.

usage (repository root): python scripts/download_weights.py [--output weights]
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def sha256(path):
    """Hex SHA-256 of a file, read in 8 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    """Download (or re-verify) the checkpoint, write its manifest and extract the tensor-only backbone."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="weights")
    parser.add_argument("--filename", default="flash_intern_image_l_22kto1k_384.pth")
    args = parser.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=6, backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504])))
    api = "https://huggingface.co/api/models/OpenGVLab/DCNv4"
    response = session.get(api, params={"blobs": "true"}, timeout=60)
    response.raise_for_status()
    metadata = response.json()
    entry = next(x for x in metadata["siblings"] if x["rfilename"] == args.filename)
    expected = entry["lfs"]["sha256"]
    size = entry["lfs"]["size"]
    final = folder / args.filename
    partial = final.with_suffix(final.suffix + ".part")
    revision = metadata["sha"]
    url = f"https://huggingface.co/OpenGVLab/DCNv4/resolve/{revision}/{args.filename}"
    if not (final.exists() and final.stat().st_size == size and sha256(final) == expected):
        for attempt in range(10):
            offset = partial.stat().st_size if partial.exists() else 0
            try:
                with session.get(url, params={"download": "true", "t": time.time_ns()},
                                  headers={"Range": f"bytes={offset}-"} if offset else {},
                                  stream=True, timeout=(30, 90)) as stream:
                    stream.raise_for_status()
                    # no 206 Partial Content: the server ignored the Range header and sends the whole file
                    if offset and stream.status_code != 206:
                        offset = 0
                    mode = "ab" if offset else "wb"
                    reported = offset // (100 << 20)
                    with partial.open(mode) as output:
                        for chunk in stream.iter_content(4 << 20):
                            if not chunk:
                                continue
                            output.write(chunk)
                            offset += len(chunk)
                            if offset // (100 << 20) > reported:
                                reported = offset // (100 << 20)
                                print(f"Downloaded {offset / (1 << 20):.0f} / {size / (1 << 20):.0f} MiB", flush=True)
                if partial.stat().st_size != size or sha256(partial) != expected:
                    raise RuntimeError("Size or SHA256 mismatch; partial file retained for inspection")
                partial.replace(final)
                break
            except (requests.RequestException, OSError) as exc:
                print(f"Retry {attempt + 1}: {type(exc).__name__}", flush=True)
                time.sleep(min(attempt + 1, 10))
        else:
            raise RuntimeError("Download failed after ten resumable attempts")
    manifest = {"repository": "OpenGVLab/DCNv4", "revision": revision,
                "filename": args.filename, "bytes": size, "sha256": expected, "url": url}
    final.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    if args.filename == "flash_intern_image_l_22kto1k_384.pth":
        # This exact author artifact contains a pickled YACS training configuration.
        # Permit that trusted artifact only after pinning its full SHA; runtime files
        # contain tensors alone and are subsequently loaded with weights_only=True.
        trusted_sha = "c694efa31c67cc246d836513080da98861a46a8ba3cee5d0dc6a8074f1a4384d"
        if expected != trusted_sha or sha256(final) != trusted_sha:
            raise RuntimeError("Author checkpoint changed; inspect before extracting its legacy metadata")
        import torch
        pure = folder / "flash_intern_image_l_backbone.pt"
        if not pure.exists():
            checkpoint = torch.load(final, map_location="cpu", mmap=True, weights_only=False)
            state = checkpoint["model"]
            state = {key: value for key, value in state.items() if not key.startswith(("head.", "conv_head."))}
            torch.save(state, pure)
        pure.with_suffix(".manifest.json").write_text(json.dumps({"source_sha256": trusted_sha,
            "sha256": sha256(pure), "bytes": pure.stat().st_size,
            "transform": "tensor-only state_dict; classification head/conv_head omitted"}, indent=2))
        print(f"Tensor-only backbone ready: {pure}", flush=True)


if __name__ == "__main__":
    main()
