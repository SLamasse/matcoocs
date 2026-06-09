#!/usr/bin/env python3
"""
build_ext.py
────────────
Compile l'extension C++ _specificity sans passer par setup.py ni
numpy.distutils (tous deux cassés sur Python 3.12 + numpy 2.x).

Usage
─────
    python build_ext.py          # compile (avec OpenMP si disponible)
    python build_ext.py --no-omp # compile sans OpenMP
    python build_ext.py --clean  # supprime le .so compilé
"""

import subprocess
import sys
import sysconfig
from pathlib import Path

# ── Chemins ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
SRC_CPP    = ROOT / "specificity.cpp"
OUT_DIR    = ROOT / "matrix"
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX")
OUT_SO     = OUT_DIR / f"_specificity{EXT_SUFFIX}"

import pybind11
PYBIND11_INCLUDE = pybind11.get_include()
PYTHON_INCLUDE   = sysconfig.get_paths()["include"]

CXX      = "g++"
CXXFLAGS = [
    "-O3", "-march=native", "-std=c++17",
    "-fPIC", "-shared", "-fvisibility=hidden",
    f"-I{PYBIND11_INCLUDE}",
    f"-I{ROOT / 'src_cpp'}",
    f"-I{PYTHON_INCLUDE}",
]


# ── Détection automatique de omp.h ────────────────────────────────────────────

def _find_omp_include() -> str | None:
    """
    Cherche omp.h dans les répertoires include de GCC.
    Sur Ubuntu, gcc installe omp.h dans un sous-dossier versionné
    (ex: /usr/lib/gcc/x86_64-linux-gnu/13/include/) mais ne l'expose
    pas toujours dans le chemin de recherche par défaut de g++.
    """
    import glob
    # Demander à gcc où sont ses includes internes
    result = subprocess.run(
        ["gcc", "-print-search-dirs"],
        capture_output=True, text=True,
    )
    candidates = []
    for line in result.stdout.splitlines():
        if line.startswith("install:"):
            # ex: install: /usr/lib/gcc/x86_64-linux-gnu/13/
            path = line.split(":", 1)[1].strip()
            candidates.append(Path(path) / "include")

    # Ajouter les chemins versionnés classiques en fallback
    for pattern in [
        "/usr/lib/gcc/x86_64-linux-gnu/*/include",
        "/usr/lib/gcc/aarch64-linux-gnu/*/include",
        "/usr/local/lib/gcc/*/include",
    ]:
        candidates += [Path(p) for p in sorted(glob.glob(pattern), reverse=True)]

    for candidate in candidates:
        if (candidate / "omp.h").exists():
            return str(candidate)
    return None


# ── Compilation ────────────────────────────────────────────────────────────────

def build(with_omp: bool = True) -> bool:
    flags = list(CXXFLAGS)

    if with_omp:
        omp_inc = _find_omp_include()
        if omp_inc:
            flags += ["-fopenmp", f"-I{omp_inc}"]
            print(f"[build_ext] OpenMP activé (omp.h trouvé dans {omp_inc})")
        else:
            print("[build_ext] omp.h introuvable → compilation sans OpenMP")
            with_omp = False

    mode = "avec OpenMP (multi-thread)" if with_omp else "sans OpenMP (mono-thread)"
    cmd  = [CXX] + flags + [str(SRC_CPP), "-o", str(OUT_SO)]

    print(f"[build_ext] Compilation {mode} → {OUT_SO.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[build_ext] ✗ Erreur :")
        print(result.stderr)
        return False

    if result.stderr.strip():
        print(result.stderr)
    print(f"[build_ext] ✓ Extension compilée : {OUT_SO}")
    return True


def clean() -> None:
    for so in OUT_DIR.glob("_specificity*.so"):
        so.unlink()
        print(f"[build_ext] Supprimé : {so}")


if __name__ == "__main__":
    if "--clean" in sys.argv:
        clean()
    elif "--no-omp" in sys.argv:
        ok = build(with_omp=False)
        sys.exit(0 if ok else 1)
    else:
        ok = build(with_omp=True)
        sys.exit(0 if ok else 1)
