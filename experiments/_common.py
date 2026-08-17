"""Shared bootstrap so experiments run without installing the package."""
import sys, pathlib, hashlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def seed(*parts) -> int:
    """Deterministic seed from arbitrary labels.

    Python's built-in hash() is randomised per process for str, so using it
    here would make results irreproducible across runs. MD5 is not used for
    security, only as a stable string-to-int map.
    """
    key = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.md5(key).digest()[:4], "big")
