"""Build the pinned official CUDA operator in this interpreter's environment.

The operator is DCNv4 (OpenGVLab/DCNv4, commit COMMIT). The source is third_party/DCNv4_op when bundled (its
SOURCE.json must name the pinned commit), otherwise a clone in .cache/upstream/DCNv4 checked out at COMMIT. It
is copied to .cache/dcnv4-build-src, patched only in includes and package imports (kernels unchanged, see the
comments in main) and installed with pip --no-build-isolation. The build log is runs/setup/dcnv4_build.log;
the exit code is pip's. --arch sets TORCH_CUDA_ARCH_LIST (e.g. 8.6 for an RTX 3090).

usage (repository root): python scripts/build_dcnv4.py [--arch 8.6]
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "4b848f7dd7da74ff03f7d278f902c6fd05b391b5"


def main():
    """Prepare the pinned source, then build and install the operator into the running interpreter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default=None)
    args = parser.parse_args()
    source = ROOT / ".cache/upstream/DCNv4"
    bundled = ROOT / "third_party/DCNv4_op"
    if bundled.exists():
        provenance = json.loads((bundled / "SOURCE.json").read_text())
        if provenance["commit"] != COMMIT:
            raise RuntimeError("Bundled CUDA source commit differs from the expected pin")
        operator_source = bundled
    else:
        if not source.exists():
            subprocess.run(["git", "clone", "https://github.com/OpenGVLab/DCNv4.git", str(source)], check=True)
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if actual != COMMIT:
            subprocess.run(["git", "-C", str(source), "checkout", COMMIT], check=True)
        operator_source = source / "DCNv4_op"
    build_source = ROOT / ".cache/dcnv4-build-src"
    shutil.copytree(operator_source, build_source, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__"))
    # Narrow CUDA-side includes avoid nvcc/MSVC parsing the complete C++ autograd
    # frontend. The kernels and their numerical expressions remain unmodified.
    for path in (build_source / "src/cuda").glob("*.cu"):
        text = path.read_text()
        text = text.replace("#include <torch/torch.h>", "#include <ATen/ATen.h>").replace("torch::kHalf", "at::kHalf")
        path.write_text(text)
    # Import only the operation used here. The unrelated FlashDeformAttn Python
    # module hard-codes pre-Blackwell GPU capabilities and blocks package import.
    initializers = {"__init__.py": "from .functions import DCNv4Function\nfrom .modules import DCNv4\n",
        "functions/__init__.py": "from .dcnv4_func import DCNv4Function\n",
        "modules/__init__.py": "from .dcnv4 import DCNv4\n"}
    for name, text in initializers.items():
        (build_source / "DCNv4" / name).write_text(text)
    env = os.environ.copy()
    # Windows without a loaded MSVC environment: import the variables set by vcvars64.bat. cl.exe lives in
    # VC/Tools/MSVC/<version>/bin/Hostx64/x64, so parents[6] is VC; the second candidate is a machine-specific path.
    if os.name == "nt" and not env.get("INCLUDE"):
        compiler = shutil.which("cl")
        candidates = [Path(compiler).parents[6] / "Auxiliary/Build/vcvars64.bat"] if compiler else []
        candidates += [Path(r"D:/C++/produce/VC/Auxiliary/Build/vcvars64.bat")]
        vcvars = next((p for p in candidates if p.exists()), None)
        if vcvars:
            result = subprocess.run(f'call "{vcvars}" >nul && set', shell=True,
                                     capture_output=True, text=True, check=True)
            for line in result.stdout.splitlines():
                if "=" in line and not line.startswith("="):
                    key, value = line.split("=", 1)
                    env[key] = value
    env["MAX_JOBS"] = env.get("MAX_JOBS", "2")
    if args.arch:
        env["TORCH_CUDA_ARCH_LIST"] = args.arch
    # Windows: accept an MSVC newer than nvcc officially supports and silence warning 177 (declared but never
    # referenced); DISTUTILS_USE_SDK makes setuptools use the environment above; VSLANG=1033 = English messages.
    if os.name == "nt":
        env["NVCC_PREPEND_FLAGS"] = "-allow-unsupported-compiler -diag-suppress=177"
        env["DISTUTILS_USE_SDK"] = "1"
        env["VSLANG"] = "1033"
    log = ROOT / "runs/setup/dcnv4_build.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building official operator; detailed log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation",
            "--no-deps", "-v", str(build_source)], env=env, stdout=output, stderr=subprocess.STDOUT)
    print(json.dumps({"exit_code": result.returncode, "commit": COMMIT, "log": str(log)}), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
