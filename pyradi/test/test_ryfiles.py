"""Regression tests for pyradi/ryfiles.py.

Run with:
    cd K:/WorkN/pyradi/pyradi/pyradi
    pytest test/test_ryfiles.py -v
"""

import io
import os
import sys
import gzip
import tarfile
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ryfiles


# ---------------------------------------------------------------------------
# latex_escape
# ---------------------------------------------------------------------------
class TestLatexEscape:
    def test_ampersand(self):
        assert ryfiles.latex_escape('a&b') == r'a\&b'

    def test_percent(self):
        assert ryfiles.latex_escape('50%') == r'50\%'

    def test_dollar(self):
        assert ryfiles.latex_escape('$1') == r'\$1'

    def test_hash(self):
        assert ryfiles.latex_escape('#tag') == r'\#tag'

    def test_underscore(self):
        assert ryfiles.latex_escape('a_b') == r'a\_b'

    def test_braces(self):
        assert ryfiles.latex_escape('{x}') == r'\{x\}'

    def test_backslash(self):
        assert ryfiles.latex_escape('a\\b') == r'a\textbackslash{}b'

    def test_greater_less(self):
        assert ryfiles.latex_escape('a>b<c') == r'a\textgreater{}b\textless{}c'

    def test_plain_text_unchanged(self):
        assert ryfiles.latex_escape('hello world') == 'hello world'

    def test_empty_string(self):
        assert ryfiles.latex_escape('') == ''

    def test_mixed(self):
        result = ryfiles.latex_escape('a&b%c$d#e_f{g}h~i^j\\k>l<m')
        assert '\\&' in result
        assert '\\%' in result
        assert '\\$' in result


# ---------------------------------------------------------------------------
# cleanFilename
# ---------------------------------------------------------------------------
class TestCleanFilename:
    def test_removes_default_chars(self):
        result = ryfiles.cleanFilename('aa bb%cc:dd/ee,ff.gg')
        assert ' ' not in result
        assert '%' not in result
        assert ':' not in result
        assert '/' not in result
        assert ',' not in result

    def test_custom_removestring(self):
        result = ryfiles.cleanFilename('hello world', ' ')
        assert result == 'helloworld'

    def test_empty_removestring_keeps_all(self):
        s = 'aa bb%cc'
        assert ryfiles.cleanFilename(s, '') == s

    def test_plain_alphanum_unchanged(self):
        assert ryfiles.cleanFilename('abc123', '') == 'abc123'


# ---------------------------------------------------------------------------
# saveHeaderArrayTextFile / loadColumnTextFile / loadHeaderTextFile
# ---------------------------------------------------------------------------
class TestSaveLoadTextFile:
    def test_roundtrip_no_header(self, tmp_path):
        f = str(tmp_path / 'data.txt')
        arr = np.outer(np.arange(1, 4), np.arange(1, 4)).astype(float)
        ryfiles.saveHeaderArrayTextFile(f, arr)
        loaded = ryfiles.loadColumnTextFile(f, loadCol=[1, 2])
        assert loaded.shape == (3, 2)
        np.testing.assert_allclose(loaded, arr[:, 1:3])

    def test_roundtrip_with_header_and_comment(self, tmp_path):
        f = str(tmp_path / 'data.txt')
        arr = np.eye(3)
        ryfiles.saveHeaderArrayTextFile(
            f, arr, header='line1\nline2', comment='%', delimiter=' '
        )
        loaded = ryfiles.loadColumnTextFile(
            f, loadCol=[1, 2], comment='%', delimiter=' '
        )
        assert loaded.shape == (3, 2)

    def test_abscissa_interpolation(self, tmp_path):
        f = str(tmp_path / 'interp.txt')
        x = np.arange(0, 5, 1.0).reshape(-1, 1)
        y = x * 2.0
        arr = np.hstack([x, y])
        ryfiles.saveHeaderArrayTextFile(f, arr)
        x_out = np.array([0.5, 1.5, 2.5]).reshape(-1, 1)
        result = ryfiles.loadColumnTextFile(f, loadCol=[1], abscissaOut=x_out)
        np.testing.assert_allclose(result.flatten(), [1.0, 3.0, 5.0], rtol=1e-5)

    def test_normalize(self, tmp_path):
        f = str(tmp_path / 'norm.txt')
        arr = np.column_stack([np.arange(1, 6), np.arange(2, 11, 2)])
        ryfiles.saveHeaderArrayTextFile(f, arr)
        result = ryfiles.loadColumnTextFile(f, loadCol=[1], normalize=1)
        assert float(np.max(result)) == pytest.approx(1.0)

    def test_return_abscissa(self, tmp_path):
        f = str(tmp_path / 'absc.txt')
        arr = np.column_stack([np.arange(5), np.arange(5, 10)])
        ryfiles.saveHeaderArrayTextFile(f, arr)
        x_out = np.array([1.0, 2.0]).reshape(-1, 1)
        vals, absc = ryfiles.loadColumnTextFile(
            f, loadCol=[1], abscissaOut=x_out, returnAbscissa=True
        )
        assert absc.shape[0] == 2

    def test_loadheader(self, tmp_path):
        f = str(tmp_path / 'hdr.txt')
        with open(f, 'w') as fh:
            fh.write('%col0,col1,col2\n1 2 3\n4 5 6\n')
        result = ryfiles.loadHeaderTextFile(f, loadCol=[1, 2], comment='%')
        assert result == ['col1', 'col2']


# ---------------------------------------------------------------------------
# arrayToLaTex
# ---------------------------------------------------------------------------
class TestArrayToLaTex:
    def test_basic_table_created(self, tmp_path):
        f = str(tmp_path / 'table.txt')
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        ryfiles.arrayToLaTex(f, arr)
        content = open(f).read()
        assert '\\begin{tabular}' in content
        assert '\\end{tabular}' in content
        assert '\\hline' in content

    def test_with_header(self, tmp_path):
        f = str(tmp_path / 'table.txt')
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        ryfiles.arrayToLaTex(f, arr, header='A & B')
        content = open(f).read()
        assert 'A & B' in content

    def test_with_left_col(self, tmp_path):
        f = str(tmp_path / 'table.txt')
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        ryfiles.arrayToLaTex(f, arr, leftCol=['X', 'Row1', 'Row2'])
        content = open(f).read()
        assert 'Row1' in content
        assert 'Row2' in content

    def test_append_mode(self, tmp_path):
        f = str(tmp_path / 'table.txt')
        arr = np.array([[1.0, 2.0]])
        ryfiles.arrayToLaTex(f, arr)
        ryfiles.arrayToLaTex(f, arr, filemode='a')
        content = open(f).read()
        assert content.count('\\begin{tabular}') == 2

    def test_formatstring(self, tmp_path):
        f = str(tmp_path / 'table.txt')
        arr = np.array([[1.0, 2.0]])
        ryfiles.arrayToLaTex(f, arr, formatstring='%.1f')
        content = open(f).read()
        assert '1.0' in content


# ---------------------------------------------------------------------------
# epsLaTexFigure
# ---------------------------------------------------------------------------
class TestEpsLaTexFigure:
    def test_creates_figure_block(self, tmp_path):
        f = str(tmp_path / 'fig.tex')
        ryfiles.epsLaTexFigure(f, 'myfig.eps', 'My caption', scale=0.8)
        content = open(f).read()
        assert '\\begin{figure}' in content
        assert '\\end{figure}' in content
        assert 'My caption' in content
        assert 'myfig.eps' in content

    def test_no_scale_uses_plain_includegraphics(self, tmp_path):
        f = str(tmp_path / 'fig.tex')
        ryfiles.epsLaTexFigure(f, 'myfig.eps', 'caption')
        content = open(f).read()
        assert 'width=' not in content
        assert '\\includegraphics{myfig.eps}' in content

    def test_append_mode(self, tmp_path):
        f = str(tmp_path / 'fig.tex')
        ryfiles.epsLaTexFigure(f, 'a.eps', 'cap1', filemode='w')
        ryfiles.epsLaTexFigure(f, 'b.eps', 'cap2', filemode='a')
        content = open(f).read()
        assert content.count('\\begin{figure}') == 2


# ---------------------------------------------------------------------------
# listFiles
# ---------------------------------------------------------------------------
class TestListFiles:
    def test_finds_py_files(self, tmp_path):
        (tmp_path / 'a.py').write_text('x')
        (tmp_path / 'b.txt').write_text('y')
        result = ryfiles.listFiles(str(tmp_path), patterns='*.py', recurse=0)
        assert len(result) == 1
        assert result[0].endswith('a.py')

    def test_recurse(self, tmp_path):
        sub = tmp_path / 'sub'
        sub.mkdir()
        (sub / 'c.py').write_text('z')
        result = ryfiles.listFiles(str(tmp_path), patterns='*.py', recurse=1)
        assert any('c.py' in r for r in result)

    def test_no_recurse(self, tmp_path):
        sub = tmp_path / 'sub'
        sub.mkdir()
        (sub / 'c.py').write_text('z')
        result = ryfiles.listFiles(str(tmp_path), patterns='*.py', recurse=0)
        assert not any('c.py' in r for r in result)

    def test_multi_pattern(self, tmp_path):
        (tmp_path / 'a.py').write_text('x')
        (tmp_path / 'b.txt').write_text('y')
        result = ryfiles.listFiles(str(tmp_path), patterns='*.py;*.txt', recurse=0)
        assert len(result) == 2

    def test_regex_mode(self, tmp_path):
        (tmp_path / 'ryfiles.py').write_text('x')
        (tmp_path / 'other.py').write_text('y')
        result = ryfiles.listFiles(
            str(tmp_path), patterns=r'^ry.*\.py$', recurse=0, useRegex=True
        )
        assert len(result) == 1
        assert result[0].endswith('ryfiles.py')

    def test_return_folders(self, tmp_path):
        sub = tmp_path / 'mydir'
        sub.mkdir()
        result = ryfiles.listFiles(str(tmp_path), patterns='*', recurse=0, return_folders=1)
        assert any('mydir' in r for r in result)


# ---------------------------------------------------------------------------
# writeRawFrames / readRawFrames
# ---------------------------------------------------------------------------
class TestRawFrames:
    def _make_frames(self):
        rng = np.random.default_rng(42)
        return rng.integers(0, 255, size=(5, 32, 32), dtype=np.uint8)

    def test_roundtrip_all_frames(self, tmp_path):
        f = str(tmp_path / 'raw.bin')
        img = self._make_frames()
        ryfiles.writeRawFrames(f, img, np.uint8)
        frames, loaded = ryfiles.readRawFrames(f, 32, 32, np.uint8)
        assert frames == 5
        np.testing.assert_array_equal(loaded, img)

    def test_roundtrip_selected_frames(self, tmp_path):
        f = str(tmp_path / 'raw.bin')
        img = self._make_frames()
        ryfiles.writeRawFrames(f, img, np.uint8)
        frames, loaded = ryfiles.readRawFrames(f, 32, 32, np.uint8, loadFrames=[0, 2, 4])
        assert frames == 3
        np.testing.assert_array_equal(loaded[0], img[0])
        np.testing.assert_array_equal(loaded[1], img[2])
        np.testing.assert_array_equal(loaded[2], img[4])

    def test_write_selected_frames(self, tmp_path):
        f = str(tmp_path / 'raw.bin')
        img = self._make_frames()
        ryfiles.writeRawFrames(f, img, np.uint8, writeFrames=[1, 3])
        frames, loaded = ryfiles.readRawFrames(f, 32, 32, np.uint8)
        assert frames == 2

    def test_2d_input_promoted_to_3d(self, tmp_path):
        f = str(tmp_path / 'raw2d.bin')
        img = np.ones((16, 16), dtype=np.uint16) * 42
        ryfiles.writeRawFrames(f, img, np.uint16)
        frames, loaded = ryfiles.readRawFrames(f, 16, 16, np.uint16)
        assert frames == 1
        assert loaded[0, 0, 0] == 42

    def test_missing_file_returns_zeros(self, tmp_path):
        frames, loaded = ryfiles.readRawFrames(
            str(tmp_path / 'missing.bin'), 10, 10, np.uint8
        )
        assert frames == 0
        assert loaded is None

    def test_bad_rank_returns_message(self, tmp_path):
        f = str(tmp_path / 'bad.bin')
        img = np.ones((2, 3, 4, 5), dtype=np.uint8)
        result = ryfiles.writeRawFrames(f, img, np.uint8)
        assert result is not None and 'inappropriate' in result


# ---------------------------------------------------------------------------
# rawFrameToImageFile
# ---------------------------------------------------------------------------
class TestRawFrameToImageFile:
    def test_writes_png(self, tmp_path):
        f = str(tmp_path / 'frame.png')
        img = np.arange(256, dtype=np.float32).reshape(16, 16)
        ryfiles.rawFrameToImageFile(img, f)
        assert os.path.isfile(f)
        assert os.path.getsize(f) > 0


# ---------------------------------------------------------------------------
# unzipGZipfile / untarTarfile
# ---------------------------------------------------------------------------
class TestArchiveOps:
    def test_unzip_gzip(self, tmp_path):
        gz = str(tmp_path / 'test.gz')
        out = str(tmp_path / 'test.txt')
        with gzip.open(gz, 'wb') as fh:
            fh.write(b'hello pyradi')
        result = ryfiles.unzipGZipfile(gz, out)
        assert result == out
        assert open(out, 'rb').read() == b'hello pyradi'

    def test_unzip_bad_file_returns_none(self, tmp_path):
        bad = str(tmp_path / 'bad.gz')
        open(bad, 'wb').write(b'not gzip data')
        result = ryfiles.unzipGZipfile(bad, str(tmp_path / 'out.txt'))
        assert result is None

    def test_untar(self, tmp_path):
        # Build a small tar archive
        tar_path = str(tmp_path / 'archive.tar')
        content_file = tmp_path / 'hello.txt'
        content_file.write_text('hi')
        with tarfile.open(tar_path, 'w') as tf:
            tf.add(str(content_file), arcname='hello.txt')
        dest = str(tmp_path / 'extracted')
        result = ryfiles.untarTarfile(tar_path, dest)
        assert 'hello.txt' in result
        assert os.path.isfile(os.path.join(dest, 'hello.txt'))


# ---------------------------------------------------------------------------
# read2DLookupTable
# ---------------------------------------------------------------------------
class TestRead2DLookupTable:
    def test_parses_correctly(self, tmp_path):
        f = str(tmp_path / 'lut.txt')
        content = (
            'xname yname zname\n'
            '0 1.0 2.0 3.0\n'
            '10.0 0.1 0.2 0.3\n'
            '20.0 0.4 0.5 0.6\n'
        )
        open(f, 'w').write(content)
        xVec, yVec, data, xl, yl, title = ryfiles.read2DLookupTable(f)
        np.testing.assert_allclose(xVec, [10.0, 20.0])
        np.testing.assert_allclose(yVec, [1.0, 2.0, 3.0])
        assert xl == 'xname'
        assert yl == 'yname'
        assert title == 'zname'
        assert data.shape == (2, 3)


# ---------------------------------------------------------------------------
# mergeDFS
# ---------------------------------------------------------------------------
class TestMergeDFS:
    def _make_df(self, t, vals):
        import pandas as pd
        return pd.DataFrame({'time': t, 'v': vals})

    def test_equal_time_merge(self):
        df1 = self._make_df([0.0, 1.0, 2.0], [10.0, 20.0, 30.0])
        df2 = self._make_df([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
        result = ryfiles.mergeDFS(df1, df2)
        assert 'v_x' in result.columns or 'v_y' in result.columns or result.shape[1] >= 3

    def test_interpolation_merge(self):
        import pandas as pd
        df1 = self._make_df([0.0, 1.0, 2.0, 3.0], [0.0, 10.0, 20.0, 30.0])
        df2 = self._make_df([0.0, 1.5, 3.0], [0.0, 5.0, 10.0])
        result = ryfiles.mergeDFS(df1, df2)
        assert result is not None
        assert result.shape[0] > 0

    def test_single_row_same_time(self):
        import pandas as pd
        df1 = pd.DataFrame({'time': [1.0], 'a': [10.0]})
        df2 = pd.DataFrame({'time': [1.0], 'b': [20.0]})
        result = ryfiles.mergeDFS(df1, df2)
        assert result is not None

    def test_prefix_columns(self):
        df1 = self._make_df([0.0, 1.0], [1.0, 2.0])
        df2 = self._make_df([0.0, 1.0], [3.0, 4.0])
        result = ryfiles.mergeDFS(df1, df2, leftPre='L', rightPre='R')
        assert any(c.startswith('L_') for c in result.columns)
        assert any(c.startswith('R_') for c in result.columns)
