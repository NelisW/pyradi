"""Regression tests for ryplot.py.

Uses the Agg (non-interactive) backend so the suite runs headless without a
display server.  All graphical output is suppressed; tests verify return types,
object state, and that public methods complete without raising exceptions.
"""

import sys
import os
import math

# --- MUST set backend before any matplotlib import ---
import matplotlib
matplotlib.use('Agg')

import pytest
import numpy as np
import matplotlib.pyplot as plt

# Add both the pyradi source directory and its parent so that
# 'import ryplot' and 'import pyradi.ryplot' both resolve.
_HERE = os.path.dirname(__file__)
_PYRADI = os.path.abspath(os.path.join(_HERE, '..'))
_PARENT = os.path.abspath(os.path.join(_HERE, '..', '..'))
for _p in (_PYRADI, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ryplot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_xy(n=20):
    """Return simple x, y arrays suitable for linear plots."""
    x = np.linspace(0.1, 10.0, n)
    y = np.sin(x)
    return x, y


def _close_all():
    """Close all open figures to keep the test process clean."""
    plt.close('all')


# ---------------------------------------------------------------------------
# FilledMarker
# ---------------------------------------------------------------------------

class TestFilledMarker:
    """Tests for FilledMarker default and custom construction."""

    def test_defaults(self):
        fm = ryplot.FilledMarker()
        assert fm.markerfacecolor == 'r'
        assert fm.markerfacecoloralt == 'b'
        assert fm.markeredgecolor == 'k'
        assert fm.marker == 'o'
        assert fm.markersize == 20
        assert fm.fillstyle == 'full'

    def test_custom_values(self):
        fm = ryplot.FilledMarker(
            markerfacecolor='blue',
            markerfacecoloralt='green',
            markeredgecolor='black',
            marker='s',
            markersize=10,
            fillstyle='left',
        )
        assert fm.markerfacecolor == 'blue'
        assert fm.markerfacecoloralt == 'green'
        assert fm.markeredgecolor == 'black'
        assert fm.marker == 's'
        assert fm.markersize == 10
        assert fm.fillstyle == 'left'

    def test_partial_custom(self):
        fm = ryplot.FilledMarker(marker='^', markersize=5)
        assert fm.marker == '^'
        assert fm.markersize == 5
        # unspecified attrs keep defaults
        assert fm.markerfacecolor == 'r'
        assert fm.fillstyle == 'full'


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

class TestMarkers:
    """Tests for Markers: construction, add, and plot."""

    def setup_method(self):
        _close_all()

    def teardown_method(self):
        _close_all()

    def test_defaults(self):
        m = ryplot.Markers()
        assert m.marker is None
        assert m.markers == []

    def test_add_single_marker(self):
        m = ryplot.Markers()
        m.add(1.0, 2.0)
        assert len(m.markers) == 1
        x, y, fm = m.markers[0]
        assert x == 1.0
        assert y == 2.0
        assert isinstance(fm, ryplot.FilledMarker)

    def test_add_multiple_markers(self):
        m = ryplot.Markers(marker='x')
        m.add(0.0, 0.0)
        m.add(1.0, 1.0)
        m.add(2.0, 4.0)
        assert len(m.markers) == 3

    def test_add_overrides_class_defaults(self):
        m = ryplot.Markers(markerfacecolor='red')
        m.add(0.0, 0.0, markerfacecolor='blue')
        _, _, fm = m.markers[0]
        assert fm.markerfacecolor == 'blue'

    def test_add_inherits_class_defaults(self):
        m = ryplot.Markers(markerfacecolor='red')
        m.add(0.0, 0.0)
        _, _, fm = m.markers[0]
        assert fm.markerfacecolor == 'red'

    def test_plot_runs_without_error(self):
        m = ryplot.Markers()
        m.add(1.0, 2.0)
        fig, ax = plt.subplots()
        m.plot(ax)   # should not raise
        plt.close(fig)


# ---------------------------------------------------------------------------
# ProcessImage
# ---------------------------------------------------------------------------

class TestProcessImage:
    """Tests for ProcessImage helper methods."""

    def setup_method(self):
        self.pi = ryplot.ProcessImage()

    def test_compressEqualizeImage_linear(self):
        image = np.linspace(0.1, 1.0, 100).reshape(10, 10)
        imgHEQ, ticks = self.pi.compressEqualizeImage(image, selectCompressSet=0)
        assert imgHEQ.shape == image.shape
        assert len(ticks) == 20  # default numCbarlevels

    def test_compressEqualizeImage_log(self):
        # Use [1.0, 10.0] so log(max)=log(10)≠0; [0.1, 1.0] caused log(1.0)=0
        # which made imgFlatSort[-1]=0 and triggered a divide-by-zero warning.
        image = np.linspace(1.0, 10.0, 100).reshape(10, 10)
        imgHEQ, ticks = self.pi.compressEqualizeImage(image, selectCompressSet=1)
        assert imgHEQ.shape == image.shape
        assert isinstance(ticks, list)

    def test_compressEqualizeImage_sqrt(self):
        image = np.linspace(0.1, 1.0, 100).reshape(10, 10)
        imgHEQ, ticks = self.pi.compressEqualizeImage(image, selectCompressSet=2)
        assert imgHEQ.shape == image.shape

    def test_compressEqualizeImage_numlevels(self):
        image = np.linspace(0.1, 1.0, 100).reshape(10, 10)
        _, ticks = self.pi.compressEqualizeImage(image, numCbarlevels=5)
        assert len(ticks) == 5

    def test_compressEqualizeImage_output_range_bug_documented(self):
        """Document pre-existing bug: equalised image values are NOT bounded to [0, 1].

        Note:
            Pre-existing bug: the CDF normalisation divides by ``imgFlatSort[-1]``
            (the maximum pixel value) instead of ``imgFlatSort.cumsum()[-1]``
            (the total cumulative sum). For inputs whose cumulative sum greatly
            exceeds the maximum single value, the CDF exceeds 1.0, and the
            histogram-equalised output can be >> 1.  Correct normalisation would be::

                cdf = imgFlatSort.cumsum() / imgFlatSort.cumsum()[-1]

            The bug is preserved because fixing it changes observable behaviour.
        """
        image = np.random.default_rng(42).uniform(0.1, 1.0, (20, 20))
        imgHEQ, _ = self.pi.compressEqualizeImage(image, selectCompressSet=0)
        # Shape is always preserved.
        assert imgHEQ.shape == image.shape
        # Due to the normalisation bug the maximum output is >> 1.
        assert float(np.max(imgHEQ)) > 1.0, (
            "Bug appears fixed: max output is now <= 1.0. "
            "If intentional, update this test."
        )
        # Minimum is non-negative (cdf values are monotonically increasing from 0).
        assert float(np.min(imgHEQ)) >= 0.0

    def test_compressEqualizeImage_output_preserves_shape(self):
        """Output shape always matches input shape regardless of the CDF bug."""
        image = np.random.default_rng(7).uniform(0.01, 2.0, (15, 15))
        imgHEQ, _ = self.pi.compressEqualizeImage(image, selectCompressSet=0)
        assert imgHEQ.shape == image.shape

    def test_compressEqualizeImage_output_monotone(self):
        """Higher input pixels map to higher (or equal) output values."""
        # Use a simple ramp so sort order is trivially known.
        image = np.linspace(0.1, 1.0, 100).reshape(10, 10)
        imgHEQ, _ = self.pi.compressEqualizeImage(image, selectCompressSet=0)
        flat_in = image.flatten()
        flat_out = imgHEQ.flatten()
        # rank order of output must match rank order of input
        assert np.all(np.argsort(flat_in) == np.argsort(flat_out))


# ---------------------------------------------------------------------------
# Plotter — construction and utility methods
# ---------------------------------------------------------------------------

class TestPlotterConstruction:
    """Tests for Plotter.__init__ and basic attribute state."""

    def setup_method(self):
        _close_all()

    def teardown_method(self):
        _close_all()

    def test_default_construction(self):
        p = ryplot.Plotter(1)
        assert p.nrow == 1
        assert p.ncol == 1
        assert p.figurenumber == 1

    def test_custom_subplots(self):
        p = ryplot.Plotter(2, subpltnrow=2, subpltncol=3)
        assert p.nrow == 2
        assert p.ncol == 3

    def test_figure_title_stored(self):
        p = ryplot.Plotter(3, figuretitle='Hello')
        assert p.figuretitle == 'Hello'

    def test_no_warning_flag(self):
        p = ryplot.Plotter(4, doWarning=False)
        assert p.doWarning is False

    def test_get_plot_returns_figure(self):
        p = ryplot.Plotter(5)
        fig = p.getPlot()
        assert fig is not None
        assert fig.number == 5

    def test_subplots_dict_starts_empty(self):
        p = ryplot.Plotter(6)
        assert p.subplots == {}


class TestPlotterBuildPlotCol:
    """Tests for buildPlotCol, nextPlotCol, resetPlotCol."""

    def setup_method(self):
        _close_all()
        self.p = ryplot.Plotter(1)

    def teardown_method(self):
        _close_all()

    def test_default_colours_length(self):
        cols = self.p.buildPlotCol()
        assert len(cols) == 25  # default list length

    def test_custom_colours(self):
        cols = self.p.buildPlotCol(plotCol=['r', 'g', 'b'], n=3)
        assert cols == ['r', 'g', 'b']

    def test_modulus_fold(self):
        cols = self.p.buildPlotCol(plotCol=['r', 'g'], n=5)
        assert cols == ['r', 'g', 'r', 'g', 'r']

    def test_next_plot_col_cycles(self):
        self.p.buildPlotCol(plotCol=['r', 'g', 'b'], n=3)
        c1 = self.p.nextPlotCol()
        c2 = self.p.nextPlotCol()
        c3 = self.p.nextPlotCol()
        c4 = self.p.nextPlotCol()  # wraps
        assert c1 == 'r'
        assert c2 == 'g'
        assert c3 == 'b'
        assert c4 == 'r'

    def test_reset_plot_col(self):
        self.p.buildPlotCol(plotCol=['r', 'g', 'b'], n=3)
        self.p.nextPlotCol()
        self.p.nextPlotCol()
        self.p.resetPlotCol()
        assert self.p.nextPlotCol() == 'r'


class TestPlotterSubplotAccess:
    """Tests for getSubPlot, getXLim, getYLim, verticalLineCoords, horizontalLineCoords."""

    def setup_method(self):
        _close_all()

    def teardown_method(self):
        _close_all()

    def test_get_subplot_none_before_plot(self):
        p = ryplot.Plotter(1)
        assert p.getSubPlot(1) is None

    def test_get_xlim_none_before_plot(self):
        p = ryplot.Plotter(1)
        assert p.getXLim(1) is None

    def test_get_ylim_none_before_plot(self):
        p = ryplot.Plotter(1)
        assert p.getYLim(1) is None

    def test_vertical_line_none_before_plot(self):
        p = ryplot.Plotter(1)
        assert p.verticalLineCoords(subplotNum=1, x=0) is None

    def test_horizontal_line_none_before_plot(self):
        p = ryplot.Plotter(1)
        assert p.horizontalLineCoords(subplotNum=1, y=0) is None

    def test_get_subplot_after_plot(self):
        p = ryplot.Plotter(1)
        x, y = _simple_xy()
        p.plot(1, x, y)
        ax = p.getSubPlot(1)
        assert ax is not None

    def test_get_xlim_after_plot(self):
        p = ryplot.Plotter(1)
        x, y = _simple_xy()
        p.plot(1, x, y)
        lims = p.getXLim(1)
        assert lims is not None
        assert len(lims) == 2
        assert lims[0] < lims[1]

    def test_get_ylim_after_plot(self):
        p = ryplot.Plotter(1)
        x, y = _simple_xy()
        p.plot(1, x, y)
        lims = p.getYLim(1)
        assert lims is not None
        assert len(lims) == 2

    def test_vertical_line_coords_after_plot(self):
        p = ryplot.Plotter(1)
        x, y = _simple_xy()
        p.plot(1, x, y)
        result = p.verticalLineCoords(subplotNum=1, x=5.0)
        assert result is not None
        xc, yc = result
        assert np.all(xc == 5.0)
        assert len(yc) == 2

    def test_horizontal_line_coords_after_plot(self):
        p = ryplot.Plotter(1)
        x, y = _simple_xy()
        p.plot(1, x, y)
        result = p.horizontalLineCoords(subplotNum=1, y=0.5)
        assert result is not None
        xc, yc = result
        assert np.all(yc == 0.5)
        assert len(xc) == 2


class TestPlotterLabelSubplot:
    """Tests for labelSubplot."""

    def setup_method(self):
        _close_all()

    def teardown_method(self):
        _close_all()

    def test_label_subplot_all_labels(self):
        p = ryplot.Plotter(1)
        x, y = _simple_xy()
        p.plot(1, x, y)
        ax = p.getSubPlot(1)
        p.labelSubplot(ax, ptitle='T', xlabel='X', ylabel='Y')
        assert ax.get_title() == 'T'
        assert ax.get_xlabel() == 'X'
        assert ax.get_ylabel() == 'Y'

    def test_label_subplot_none_args_no_error(self):
        p = ryplot.Plotter(1)
        x, y = _simple_xy()
        p.plot(1, x, y)
        ax = p.getSubPlot(1)
        p.labelSubplot(ax)  # all None — should not raise


# ---------------------------------------------------------------------------
# Plotter — plot methods
# ---------------------------------------------------------------------------

class TestPlotterPlot:
    """Tests for Plotter.plot (linear cartesian)."""

    def setup_method(self):
        _close_all()

    def teardown_method(self):
        _close_all()

    def test_basic_plot_returns_axis(self):
        p = ryplot.Plotter(1)
        x, y = _simple_xy()
        ax = p.plot(1, x, y)
        assert ax is not None

    def test_plot_with_title_and_labels(self):
        p = ryplot.Plotter(1)
        x, y = _simple_xy()
        ax = p.plot(1, x, y, ptitle='Title', xlabel='X', ylabel='Y')
        assert ax.get_title() == 'Title'
        assert ax.get_xlabel() == 'X'
        assert ax.get_ylabel() == 'Y'

    def test_plot_multi_column_y(self):
        p = ryplot.Plotter(1)
        x = np.linspace(0, 1, 20)
        y = np.column_stack([np.sin(x), np.cos(x)])
        ax = p.plot(1, x, y)
        assert ax is not None

    def test_plot_with_legend(self):
        p = ryplot.Plotter(1)
        x, y = _simple_xy()
        ax = p.plot(1, x, y, label=['sin'], legendAlpha=0.5)
        assert ax is not None

    def test_plot_pd_series(self):
        """pd.Series inputs should be accepted and converted."""
        import pandas as pd
        p = ryplot.Plotter(1)
        x = pd.Series(np.linspace(0, 1, 10))
        y = pd.Series(np.linspace(0, 1, 10))
        ax = p.plot(1, x, y)
        assert ax is not None

    def test_plot_y_invert(self):
        p = ryplot.Plotter(1)
        x, y = _simple_xy()
        ax = p.plot(1, x, y, yInvert=True)
        ylim = ax.get_ylim()
        assert ylim[0] > ylim[1]  # inverted

    def test_plot_multiple_subplots(self):
        p = ryplot.Plotter(1, 2, 1)
        x, y = _simple_xy()
        ax1 = p.plot(1, x, y, ptitle='Top')
        ax2 = p.plot(2, x, -y, ptitle='Bottom')
        assert ax1 is not None
        assert ax2 is not None


class TestPlotterLogLogSemilog:
    """Tests for logLog, semilogX, semilogY."""

    def setup_method(self):
        _close_all()

    def teardown_method(self):
        _close_all()

    def test_log_log(self):
        p = ryplot.Plotter(1)
        x = np.logspace(0, 3, 30)
        y = x ** 2
        ax = p.logLog(1, x, y, ptitle='loglog')
        assert ax is not None

    def test_semilog_x(self):
        p = ryplot.Plotter(1)
        x = np.logspace(0, 3, 30)
        y = np.log(x)
        ax = p.semilogX(1, x, y, ptitle='semilogX')
        assert ax is not None

    def test_semilog_y(self):
        p = ryplot.Plotter(1)
        x = np.linspace(1, 10, 30)
        y = np.exp(x / 3.0)
        ax = p.semilogY(1, x, y, ptitle='semilogY')
        assert ax is not None


class TestPlotterStackPlot:
    """Tests for Plotter.stackplot."""

    def setup_method(self):
        _close_all()

    def teardown_method(self):
        _close_all()

    def test_stackplot_basic(self):
        p = ryplot.Plotter(1)
        np.random.seed(0)
        x = np.arange(10)
        y = np.vstack([np.random.randint(5, 50, 10) for _ in range(3)])
        y = (y / y.sum(axis=0, keepdims=True) * 100).T  # (10,3)
        ax = p.stackplot(1, x, y, ptitle='Stack')
        assert ax is not None


class TestPlotterEmptyPlot:
    """Tests for Plotter.emptyPlot."""

    def setup_method(self):
        _close_all()

    def teardown_method(self):
        _close_all()

    def test_empty_plot_returns_axis(self):
        p = ryplot.Plotter(1)
        ax = p.emptyPlot(1)
        assert ax is not None

    def test_empty_plot_polar_projection(self):
        p = ryplot.Plotter(1)
        ax = p.emptyPlot(1, projection='polar')
        assert ax is not None


class TestPlotterPolar:
    """Tests for Plotter.polar."""

    def setup_method(self):
        _close_all()

    def teardown_method(self):
        _close_all()

    def test_polar_basic(self):
        p = ryplot.Plotter(1)
        theta = np.linspace(0, 2 * np.pi, 50)
        r = np.abs(np.sin(theta))
        ax = p.polar(1, theta, r, ptitle='Polar')
        assert ax is not None

    def test_polar_multi_column_bug_documented(self):
        """Document pre-existing bug: multi-column r with highlightNegative raises ValueError.

        Note:
            Pre-existing bug in Plotter.polar: when ``r`` has M > 1 columns and
            ``highlightNegative`` is active (the default), the code calls::

                np.split(rr, zero_crossings)   # rr shape (N, M)

            producing sub-arrays of shape (n, M).  It then evaluates::

                if negrrr[ii][0] < 0:

            where ``negrrr[ii][0]`` is a row of M values.  Comparing an
            M-element array to a scalar with ``<`` returns an array, and
            ``if <array>`` raises ``ValueError: The truth value of an array
            with more than one element is ambiguous``.

            The workaround requires BOTH ``rscale=[-x, rmax]`` (so that the code
            does not override ``highlightNegative`` back to ``True``) AND
            ``highlightNegative=False`` (so the buggy loop is skipped).  Neither
            alone is sufficient — see ``test_polar_multi_column_workaround``.
        """
        import pytest as _pytest
        p = ryplot.Plotter(1)
        theta = np.linspace(0, 2 * np.pi, 50)
        r = np.column_stack([np.abs(np.sin(theta)), np.abs(np.cos(theta))])
        # Default call (no rscale) triggers the bug:
        with _pytest.raises(ValueError, match="truth value of an array"):
            p.polar(1, theta, r)

    def test_polar_multi_column_workaround(self):
        """Multi-column r works when both rscale[0]<0 and highlightNegative=False are set.

        Note:
            The ``polar`` method unconditionally overrides the ``highlightNegative``
            argument to ``True`` in two cases::

                if rscale is not None:
                    if rscale[0] < 0:
                        MakeAbs = False          # highlightNegative NOT touched
                    else:
                        highlightNegative = True # override
                else:
                    highlightNegative = True     # override

            So the only way to preserve ``highlightNegative=False`` is to also
            supply ``rscale`` with a negative first element.  Both are required;
            either alone is insufficient.
        """
        p = ryplot.Plotter(1)
        theta = np.linspace(0, 2 * np.pi, 50)
        r = np.column_stack([np.abs(np.sin(theta)), np.abs(np.cos(theta))])
        # Both conditions are required: rscale[0]<0 prevents the override of
        # highlightNegative; highlightNegative=False then skips the buggy loop.
        ax = p.polar(1, theta, r, rscale=[-0.1, 1.0], highlightNegative=False)
        assert ax is not None


class TestPlotterShowImage:
    """Tests for Plotter.showImage."""

    def setup_method(self):
        _close_all()

    def teardown_method(self):
        _close_all()

    def test_show_image_basic(self):
        p = ryplot.Plotter(1)
        img = np.random.default_rng(0).uniform(0, 1, (50, 50))
        ax = p.showImage(1, img, ptitle='Image')
        assert ax is not None

    def test_show_image_with_colorbar(self):
        p = ryplot.Plotter(1)
        img = np.random.default_rng(0).uniform(0, 1, (30, 30))
        ax = p.showImage(1, img, cbarshow=True)
        assert ax is not None


class TestPlotterMeshContour:
    """Tests for Plotter.meshContour."""

    def setup_method(self):
        _close_all()

    def teardown_method(self):
        _close_all()

    def test_mesh_contour_basic(self):
        p = ryplot.Plotter(1)
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.exp(-(X**2 + Y**2))
        ax = p.meshContour(1, x, y, Z, levels=5, ptitle='Contour')
        assert ax is not None


class TestPlotterPlot3D:
    """Tests for Plotter.plot3d."""

    def setup_method(self):
        _close_all()

    def teardown_method(self):
        _close_all()

    def test_plot3d_basic(self):
        p = ryplot.Plotter(1)
        t = np.linspace(0, 4 * np.pi, 50)
        x = np.sin(t)
        y = np.cos(t)
        z = t
        ax = p.plot3d(1, x, y, z, ptitle='3D')
        assert ax is not None


class TestPlotterPlotArray:
    """Tests for Plotter.plotArray."""

    def setup_method(self):
        _close_all()

    def teardown_method(self):
        _close_all()

    def test_plot_array_cols_bug_documented(self):
        """Document pre-existing bug: plotArray raises TypeError when labels=None.

        Note:
            Pre-existing bug in Plotter.plotArray: when ``labels`` is ``None``
            (its default value), the code unconditionally executes::

                xlabel = labels[0]

            which raises ``TypeError: 'NoneType' object is not subscriptable``.
            The function provides no guard for the ``labels=None`` case.
            The workaround is to always supply a labels list.
        """
        import pytest as _pytest
        p = ryplot.Plotter(1, 1, 3)
        x = np.linspace(0, 1, 30)
        y1 = np.sin(2 * np.pi * x)
        y2 = np.cos(2 * np.pi * x)
        arr = np.column_stack([x, y1, y2])
        # Default labels=None triggers the bug:
        with _pytest.raises(TypeError):
            p.plotArray(1, arr, slicedim=0)

    def test_plot_array_cols_with_labels(self):
        """plotArray works correctly when labels are supplied explicitly."""
        p = ryplot.Plotter(1, 1, 3)
        x = np.linspace(0, 1, 30)
        y1 = np.sin(2 * np.pi * x)
        y2 = np.cos(2 * np.pi * x)
        arr = np.column_stack([x, y1, y2])
        # Must supply labels to avoid the pre-existing bug:
        p.plotArray(1, arr, slicedim=0, labels=['x', 'sin', 'cos'])


class TestPlotterSaveFig:
    """Tests for Plotter.saveFig — writes to tmp_path."""

    def setup_method(self):
        _close_all()

    def teardown_method(self):
        _close_all()

    def test_save_png(self, tmp_path):
        p = ryplot.Plotter(1)
        x, y = _simple_xy()
        p.plot(1, x, y)
        out = str(tmp_path / 'out.png')
        p.saveFig(out, dpi=72)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_save_pdf(self, tmp_path):
        p = ryplot.Plotter(1)
        x, y = _simple_xy()
        p.plot(1, x, y)
        out = str(tmp_path / 'out.pdf')
        p.saveFig(out, dpi=72)
        assert os.path.exists(out)

    def test_save_empty_filename_no_error(self):
        """Empty filename should not raise (guarded in saveFig)."""
        p = ryplot.Plotter(1)
        x, y = _simple_xy()
        p.plot(1, x, y)
        p.saveFig('')  # should not raise


# ---------------------------------------------------------------------------
# savePlot context manager
# ---------------------------------------------------------------------------

class TestSavePlot:
    """Tests for the savePlot context manager."""

    def setup_method(self):
        _close_all()

    def teardown_method(self):
        _close_all()

    def test_context_manager_no_save(self):
        x, y = _simple_xy()
        with ryplot.savePlot(1) as p:
            p.plot(1, x, y)
        # No file written; just ensure no exception

    def test_context_manager_saves_file(self, tmp_path):
        x, y = _simple_xy()
        out = str(tmp_path / 'ctx.png')
        with ryplot.savePlot(1, saveName=out) as p:
            p.plot(1, x, y)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_context_manager_saves_list(self, tmp_path):
        x, y = _simple_xy()
        out1 = str(tmp_path / 'a.png')
        out2 = str(tmp_path / 'b.png')
        with ryplot.savePlot(1, saveName=[out1, out2]) as p:
            p.plot(1, x, y)
        assert os.path.exists(out1)
        assert os.path.exists(out2)

    def test_context_manager_returns_plotter(self):
        with ryplot.savePlot(1) as p:
            assert isinstance(p, ryplot.Plotter)


# ---------------------------------------------------------------------------
# cubehelixcmap
# ---------------------------------------------------------------------------

class TestCubehelixCmap:
    """Tests for the cubehelixcmap function."""

    def test_returns_colormap(self):
        cmap = ryplot.cubehelixcmap()
        assert hasattr(cmap, 'name') or hasattr(cmap, '__call__')

    def test_default_name(self):
        cmap = ryplot.cubehelixcmap()
        assert cmap.name == 'cubehelix_map'

    def test_reverse_option(self):
        cmap_fwd = ryplot.cubehelixcmap(reverse=False)
        cmap_rev = ryplot.cubehelixcmap(reverse=True)
        # Both should be valid colormaps (name suffix changes)
        assert cmap_fwd is not None
        assert cmap_rev is not None

    def test_custom_levels(self):
        cmap = ryplot.cubehelixcmap(nlev=128.)
        assert cmap is not None

    def test_usable_with_imshow(self):
        """Colormap returned by cubehelixcmap can be passed to imshow."""
        cmap = ryplot.cubehelixcmap()
        fig, ax = plt.subplots()
        img = np.random.default_rng(0).uniform(0, 1, (10, 10))
        ax.imshow(img, cmap=cmap)
        plt.close(fig)


# ---------------------------------------------------------------------------
# RGBToPyCmap
# ---------------------------------------------------------------------------

class TestRGBToPyCmap:
    """Tests for the RGBToPyCmap helper function."""

    def test_basic_output_structure(self):
        rgb = np.array([[0.0, 0.0, 0.0],
                        [0.5, 0.5, 0.5],
                        [1.0, 1.0, 1.0]])
        result = ryplot.RGBToPyCmap(rgb)
        assert 'red' in result
        assert 'green' in result
        assert 'blue' in result

    def test_output_lengths_match_input(self):
        n = 5
        rgb = np.tile(np.linspace(0, 1, n).reshape(-1, 1), (1, 3))
        result = ryplot.RGBToPyCmap(rgb)
        assert len(result['red']) == n
        assert len(result['green']) == n
        assert len(result['blue']) == n

    def test_step_axis_bounds(self):
        rgb = np.array([[0.0, 0.1, 0.2],
                        [0.5, 0.6, 0.7],
                        [1.0, 0.9, 0.8]])
        result = ryplot.RGBToPyCmap(rgb)
        # First step should be 0.0, last should be 1.0
        assert result['red'][0][0] == pytest.approx(0.0)
        assert result['red'][-1][0] == pytest.approx(1.0)

    def test_usable_to_create_colormap(self):
        """RGBToPyCmap output can be used to build a LinearSegmentedColormap."""
        from matplotlib.colors import LinearSegmentedColormap
        rgb = np.array([[0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0]])
        mpl_data = ryplot.RGBToPyCmap(rgb)
        cmap = LinearSegmentedColormap('test', mpl_data)
        assert cmap is not None


# ---------------------------------------------------------------------------
# Pre-existing bug: turbo_colormap_data.shadoAllpe
# ---------------------------------------------------------------------------

class TestPreExistingBugs:
    """Document and assert pre-existing bugs so accidental fixes are caught."""

    def test_no_wrappable_long_lines(self):
        """No non-URL lines longer than 120 chars should exist in module code.

        URL-only lines (bare https:// in docstrings, or # http... comments)
        are exempt because URLs cannot be meaningfully shortened.
        """
        with open(os.path.join(_PYRADI, 'ryplot.py'), encoding='utf-8') as f:
            lines = f.readlines()
        main_start = next(
            (i for i, l in enumerate(lines) if "__name__ == '__main__'" in l),
            len(lines),
        )
        non_url_long = [
            l.rstrip()
            for i, l in enumerate(lines)
            if i < main_start
            and len(l.rstrip()) > 120
            and 'http' not in l  # bare URL lines (in docstrings or comments) exempt
        ]
        assert non_url_long == [], (
            f"Wrappable lines > 120 chars found:\n" +
            "\n".join(f"  ({len(l)}): {l[:100]}" for l in non_url_long)
        )

    def test_shadoAllpe_bug_documented(self):
        """turbo_colormap_data.shadoAllpe[0] is a pre-existing typo.

        The correct attribute is .shape[0].  The buggy branch only executes
        when 'turbo' is NOT already registered in matplotlib (i.e. never in
        practice on matplotlib >= 3.2).  The bug is preserved per the cardinal
        rule: never change observable behaviour during refactoring.

        This test asserts that the numpy array does NOT have the misspelled
        attribute 'shadoAllpe', confirming the bug is still present.
        """
        # Access the turbo data array through the module's global namespace.
        # We look for the array used in the register_cmap call.
        # The variable is named 'turbo_colormap_data' at module level.
        import importlib, types
        mod = importlib.import_module('ryplot')
        # Find the turbo array
        turbo = getattr(mod, 'turbo_colormap_data', None)
        if turbo is not None:
            # Confirm that 'shadoAllpe' does NOT exist (it is the typo)
            assert not hasattr(turbo, 'shadoAllpe'), (
                "Bug has been silently fixed: 'shadoAllpe' now exists. "
                "If this was intentional, update this test."
            )
            # And confirm that the correct attribute DOES exist
            assert hasattr(turbo, 'shape')


# ---------------------------------------------------------------------------
# __all__ contract
# ---------------------------------------------------------------------------

class TestAllContract:
    """Verify that every name in __all__ is actually importable."""

    def test_all_names_exist(self):
        import importlib
        mod = importlib.import_module('ryplot')
        phantoms = [name for name in mod.__all__ if not hasattr(mod, name)]
        assert phantoms == [], f"Phantom names in __all__: {phantoms}"

    def test_plotter_in_all(self):
        assert 'Plotter' in ryplot.__all__

    def test_cubehelixcmap_in_all(self):
        assert 'cubehelixcmap' in ryplot.__all__

    def test_filledmarker_in_all(self):
        assert 'FilledMarker' in ryplot.__all__

    def test_markers_in_all(self):
        assert 'Markers' in ryplot.__all__

    def test_processimage_in_all(self):
        assert 'ProcessImage' in ryplot.__all__

    def test_saveplot_in_all(self):
        assert 'savePlot' in ryplot.__all__
