"""Build the portable code archive for submission (competition images, caches and run outputs excluded).

usage: python scripts/package_project.py [--include-weights] [--output <zip path relative to the repository>]
--include-weights adds weights/final_a.pt (the submitted model, ~0.9 GB); the ImageNet backbone is
not needed for inference (it is part of the checkpoint) and is fetched by scripts/download_weights.py.
Default output: deliverables/solar_filaments_final_code.zip (solar_filaments_final_with_weights.zip with
--include-weights), plus a .manifest.json with its size and SHA-256. The archive holds DOCUMENTS and the
source FOLDERS only: data/, runs/ and weights/ are left out (except weights/final_a.pt with --include-weights).
"""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

DOCUMENTS = ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "pyproject.toml", "requirements.txt", ".gitignore",
             ".gitattributes")
FOLDERS = ("solar_filaments", "scripts", "tests", "configs", "notebooks", "docs", "paper", "research", "third_party")
BUILD_OUTPUTS = {".pyc", ".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".out", ".synctex.gz"}


def released(path, root):
    """Source files only: no caches, no LaTeX build outputs (paper/*.log, paper/main.pdf)."""
    relative = path.relative_to(root)
    if "__pycache__" in relative.parts or ".ipynb_checkpoints" in relative.parts or path.suffix in BUILD_OUTPUTS:
        return False
    return not (relative.parts[0] == "paper" and (path.suffix == ".log" or relative.name == "main.pdf"))


def sha256(path):
    """Hex SHA-256 of a file, read in 8 MiB blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    """Write the zip archive and its manifest, and print the manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--include-weights", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / (args.output or ("deliverables/solar_filaments_final_with_weights.zip" if args.include_weights
                                     else "deliverables/solar_filaments_final_code.zip"))
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [root / name for name in DOCUMENTS]
    for folder in FOLDERS:
        if (root / folder).exists():
            files += [p for p in (root / folder).rglob("*") if p.is_file() and released(p, root)]
    if args.include_weights:
        files.append(root / "weights/final_a.pt")
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(files)):
            if path.exists():
                archive.write(path, path.relative_to(root))
    manifest = {"path": str(output), "bytes": output.stat().st_size, "sha256": sha256(output),
                "includes_weights": args.include_weights, "includes_solar_data": False}
    if args.include_weights:
        manifest["final_a_sha256"] = sha256(root / "weights/final_a.pt")
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
