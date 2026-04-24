# -*- coding: utf-8 -*-
#
# pyradi Sphinx configuration
#

import sys
import os

# The pyradi package lives one level above this doc/ directory.
# Add its parent so that 'import pyradi' resolves correctly.
sys.path.insert(0, os.path.abspath('..'))   # the pyradi package itself
sys.path.insert(0, os.path.abspath('../..'))  # parent so 'import pyradi' works

# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',      # Google/NumPy docstring support
    'sphinx.ext.viewcode',      # [source] links
    'sphinx.ext.todo',
    'sphinx.ext.imgmath',
    'sphinx.ext.intersphinx',
]

# Mock heavy C-extension packages that may not be available at doc-build time.
# scipy is intentionally NOT mocked — it is installed in the pyradi conda env
# and ryplanck.py uses scipy.constants at module level (const.h must be a real
# float, not a Mock, or PlanckConstants.__init__ raises TypeError).
autodoc_mock_imports = [
    'h5py',
    'PIL',
    'Pillow',
    'imageio',
    'skimage',
    'sklearn',
    'cv2',
    'transformations',
    'mayavi',
    'tvtk',
    'pyface',
    'traits',
]

# autodoc settings — do NOT set 'members': True here.
# Each RST file uses explicit .. autofunction:: / .. autoclass:: directives to
# document individual members.  Adding 'members': True would make every
# .. automodule:: also pull in all members, causing hundreds of
# "duplicate object description" warnings.
autodoc_default_options = {
    'undoc-members': False,
    'show-inheritance': True,
}
autodoc_member_order = 'bysource'
add_module_names = False          # keep names short in the output

# Napoleon settings (for Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# intersphinx: cross-link to Python, NumPy, SciPy docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy':  ('https://numpy.org/doc/stable', None),
}

templates_path = ['_templates']
source_suffix  = '.rst'
master_doc     = 'index'

project   = u'pyradi'
copyright = u'2012-2026, CJ Willers'
author    = u'CJ Willers'

version = '0.1'
release = '0.1.0'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = 'sphinx'

todo_include_todos = True

# -- HTML output ----------------------------------------------------------

html_theme = 'alabaster'          # clean built-in theme; no extra install needed
                                  # fall back to 'default' if alabaster is missing

html_theme_options = {
    'description':  'General-purpose Python radiometry toolkit',
    'github_user':  'NelisW',
    'github_repo':  'pyradi',
    'fixed_sidebar': True,
}

html_static_path  = ['_static']
html_logo         = '_images/pyradi.png'
htmlhelp_basename = 'pyradidoc'

# -- LaTeX output ---------------------------------------------------------

latex_elements = {
    'papersize': 'a4paper',
}

latex_documents = [
    ('index', 'pyradi.tex', u'pyradi Documentation', u'CJ Willers', 'manual'),
]
