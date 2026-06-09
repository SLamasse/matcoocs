"""
setup.py
────────
Compilation de l'extension C++ _specificity via PyBind11.

Usage
─────
    pip install -e .          # installation en mode développement
    python setup.py build_ext --inplace   # compilation locale

L'extension sera importée automatiquement par src/matrix/specificity.py
avec repli sur l'implémentation Python si la compilation échoue.
"""

from setuptools import setup, Extension, find_packages
import pybind11
import os
import sys

# ── Drapeaux de compilation ───────────────────────────────────────────────────
COMPILE_ARGS = ["-O3", "-march=native", "-std=c++17", "-fvisibility=hidden"]
LINK_ARGS    = []

# OpenMP : Linux/macOS
if sys.platform.startswith("linux"):
    COMPILE_ARGS += ["-fopenmp"]
    LINK_ARGS    += ["-fopenmp"]
elif sys.platform == "darwin":
    # Homebrew libomp
    COMPILE_ARGS += ["-Xpreprocessor", "-fopenmp"]
    LINK_ARGS    += ["-lomp"]

# ── Extension ─────────────────────────────────────────────────────────────────
specificity_ext = Extension(
    name     = "src.matrix._specificity",
    sources  = ["src_cpp/specificity.cpp"],
    define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    include_dirs = [
        pybind11.get_include(),
        "src_cpp",
    ],
    language = "c++",
    extra_compile_args = COMPILE_ARGS,
    extra_link_args    = LINK_ARGS,
)

# ── Métadonnées ───────────────────────────────────────────────────────────────
setup(
    name             = "sem-analysis",
    version          = "0.1.0",
    description      = "Pipeline d'analyse sémantique par cooccurrence",
    packages         = find_packages(where="."),
    package_dir      = {"": "."},
    ext_modules      = [specificity_ext],
    install_requires = [
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "umap-learn>=0.5",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "adjustText>=0.8",
        "prince>=0.13",
        "networkx>=3.0",
        "python-louvain>=0.16",
        "pybind11>=2.11",
    ],
    python_requires  = ">=3.10",
    zip_safe         = False,
)
