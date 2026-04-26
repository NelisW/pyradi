"""
fix_urls.py — Replace high-confidence stale URLs with their working equivalents.

Run from the pyradi folder:
    python fix_urls.py

Skips staleURLs.md itself (that file documents the changes, not a target).
Prints every substitution made: file, line number, old URL → new URL.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).parent.parent  # pyradi/pyradi/pyradi/
SKIP_FILES = {"staleURLs.md", "fix_urls.py", "check_urls.py", "session-handoff.md"}
SKIP_DIRS  = {".ipynb_checkpoints", "__pycache__", ".git"}
EXTENSIONS = {".py", ".rst", ".md"}

# ---------------------------------------------------------------------------
# Replacement table — ORDER MATTERS: longer / more specific strings first.
# A shorter string that is a prefix of a longer one must come AFTER it.
# ---------------------------------------------------------------------------
REPLACEMENTS = [
    # ── HDF Group: most-specific paths before shorter prefixes ──────────────
    ("https://www.hdfgroup.org/products/java/hdfview/index.html",
     "https://www.hdfgroup.org/download-hdfview/"),

    ("https://www.hdfgroup.org/products/java/hdfview/",
     "https://www.hdfgroup.org/download-hdfview/"),

    ("https://www.hdfgroup.org/HDF5/Tutor/tools.html",
     "https://support.hdfgroup.org/documentation/hdf5/latest/"),

    ("https://www.hdfgroup.org/HDF5/Tutor/",
     "https://support.hdfgroup.org/documentation/hdf5/latest/_getting_started.html"),

    ("https://www.hdfgroup.org/HDF5/",
     "https://www.hdfgroup.org/solutions/hdf5/"),

    # ── h5py ────────────────────────────────────────────────────────────────
    ("http://www.h5py.org/",
     "https://h5py.readthedocs.io/en/stable/"),

    ("http://docs.h5py.org/en/latest/",
     "https://h5py.readthedocs.io/en/stable/"),

    # ── matplotlib axes_grid colorbar (anchor URL before bare URL) ──────────
    ("http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html"
     "#colorbar-whose-height-or-width-in-sync-with-the-master-axes",
     "https://matplotlib.org/stable/gallery/axes_grid1/demo_colorbar_with_axes_divider.html"),

    ("http://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html",
     "https://matplotlib.org/stable/gallery/axes_grid1/demo_axes_grid.html"),

    ("http://matplotlib.org/mpl_toolkits/axes_grid/api/axis_artist_api.html",
     "https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.axisartist.axis_artist.html"),

    ("http://matplotlib.org/mpl_toolkits/axes_grid/users/axisartist.html",
     "https://matplotlib.org/stable/api/toolkits/axisartist.html"),

    # ── matplotlib versioned / old paths ────────────────────────────────────
    ("http://matplotlib.org/1.3.1/api/axes_api.html"
     "#matplotlib.axes.Axes.ticklabel_format",
     "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.ticklabel_format.html"),

    ("http://matplotlib.org/api/ticker_api.html",
     "https://matplotlib.org/stable/api/ticker_api.html"),

    ("http://matplotlib.org/examples/pylab_examples/newscalarformatter_demo.html",
     "https://matplotlib.org/stable/gallery/ticks/scalarformatter.html"),

    ("http://matplotlib.org/examples/axes_grid/demo_floating_axes.html",
     "https://matplotlib.org/stable/gallery/axisartist/demo_floating_axes.html"),

    ("http://matplotlib.org/devel/add_new_projection.html",
     "https://matplotlib.org/stable/gallery/misc/custom_projection.html"),

    # ── matplotlib SourceForge ───────────────────────────────────────────────
    ("http://matplotlib.sourceforge.net/examples/pylab_examples/"
     "colorbar_tick_labelling_demo.html",
     "https://matplotlib.org/stable/gallery/ticks/colorbar_tick_labelling_demo.html"),

    ("http://matplotlib.sourceforge.net/examples/pylab_examples/show_colormaps.html",
     "https://matplotlib.org/stable/gallery/color/colormap_reference.html"),

    # ── NumPy docs (versioned before unversioned) ────────────────────────────
    ("https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.einsum.html",
     "https://numpy.org/doc/stable/reference/generated/numpy.einsum.html"),

    ("https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html",
     "https://numpy.org/doc/stable/reference/generated/numpy.einsum.html"),

    ("http://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html",
     "https://numpy.org/doc/stable/reference/generated/numpy.histogram.html"),

    # ── SciPy docs (versioned before unversioned) ────────────────────────────
    ("https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/"
     "scipy.ndimage.filters.gaussian_filter.html",
     "https://docs.scipy.org/doc/scipy/reference/generated/"
     "scipy.ndimage.gaussian_filter.html"),

    ("http://docs.scipy.org/doc/scipy/reference/constants.html",
     "https://docs.scipy.org/doc/scipy/reference/constants.html"),

    # ── SciPy wiki / cookbook ───────────────────────────────────────────────
    ("http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps",
     "https://scipy-cookbook.readthedocs.io/items/Matplotlib_Show_colormaps.html"),

    ("http://wiki.scipy.org/Cookbook/SavitzkyGolay",
     "https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html"),

    ("http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps",
     "https://scipy-cookbook.readthedocs.io/items/Matplotlib_Show_colormaps.html"),

    ("http://www.scipy.org/Cookbook/Matplotlib/ColormapTransformations",
     "https://scipy.github.io/old-wiki/pages/Cookbook/Matplotlib/ColormapTransformations"),

    # ── Savitzky-Golay paper (dead university server) ────────────────────────
    ("http://www.wire.tu-bs.de/OLDWEB/mameyer/cmr/savgol.pdf",
     "https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html"),

    # ── janeriksolem.net (old Blogger date-path) ─────────────────────────────
    ("http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html",
     "http://www.janeriksolem.net/histogram-equalization-with-python-and.html"),

    # ── nbviewer: ipython domain → nbviewer.org ─────────────────────────────
    ("http://nbviewer.ipython.org/github/NelisW/ComputationalRadiometry"
     "/blob/master/02-PythonWhirlwindCheatSheet.ipynb",
     "https://nbviewer.org/github/NelisW/ComputationalRadiometry"
     "/blob/master/02-PythonWhirlwindCheatSheet.ipynb"),

    # ── nbviewer: jupyter.org domain → nbviewer.org / GitHub ────────────────
    ("http://nbviewer.jupyter.org/github/NelisW/ComputationalRadiometry"
     "/blob/master/07-Optical-Sources.ipynb",
     "https://nbviewer.org/github/NelisW/ComputationalRadiometry"
     "/blob/master/07-Optical-Sources.ipynb"),

    # bare directory listing → GitHub repo (longer path before shorter)
    ("http://nbviewer.jupyter.org/github/NelisW/ComputationalRadiometry/blob/master/",
     "https://github.com/NelisW/ComputationalRadiometry"),

    # ── OSSIM Trac ───────────────────────────────────────────────────────────
    ("https://ossim.trac.sourceforge.net/",
     "https://trac.osgeo.org/ossim/"),

    # ── Vaisala ──────────────────────────────────────────────────────────────
    ("http://www.vaisala.com/Vaisala%20Documents/Application%20notes/"
     "Humidity_Conversion_Formulas_B210973EN-D.pdf",
     "https://www.vaisala.com/en/ebook/make-your-job-easier-humidity-conversion-formulas"),

    # ── GitHub Help → GitHub Docs ────────────────────────────────────────────
    ("https://help.github.com/articles/creating-project-pages-manually/",
     "https://docs.github.com/en/pages/getting-started-with-github-pages/"
     "creating-a-github-pages-site"),

    ("https://help.github.com/articles/creating-pages-with-the-automatic-generator/",
     "https://docs.github.com/en/pages/getting-started-with-github-pages/"
     "creating-a-github-pages-site"),

    ("https://help.github.com/articles/user-organization-and-project-pages/",
     "https://docs.github.com/en/pages/getting-started-with-github-pages/"
     "about-github-pages"),
]


def fix_file(path: pathlib.Path) -> int:
    """Apply all replacements to one file. Returns count of substitutions made."""
    try:
        original = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  SKIP (read error: {e}): {path}")
        return 0

    text = original
    changes = []

    for old, new in REPLACEMENTS:
        if old not in text:
            continue
        # Count occurrences and record line numbers
        for lineno, line in enumerate(text.splitlines(), 1):
            if old in line:
                changes.append((lineno, old, new))
        text = text.replace(old, new)

    if text == original:
        return 0

    path.write_text(text, encoding="utf-8")
    rel = path.relative_to(ROOT)
    for lineno, old, new in changes:
        print(f"  [{rel}:{lineno}]")
        print(f"    - {old}")
        print(f"    + {new}")
    return len(changes)


def main():
    total = 0
    files_changed = 0

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in EXTENSIONS:
            continue
        if path.name in SKIP_FILES:
            continue
        if any(d in path.parts for d in SKIP_DIRS):
            continue

        n = fix_file(path)
        if n:
            files_changed += 1
            total += n

    print(f"\n{'='*60}")
    print(f"  {total} substitutions in {files_changed} files.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
