"""
Setup script for pyradi.

pyradi is distributed via GitHub only:
    https://github.com/NelisW/pyradi

Install from source
-------------------
Clone the repository and install into the active conda environment:

    conda activate pyradi
    git clone https://github.com/NelisW/pyradi.git
    cd pyradi
    pip install -e .

Use the conda environment defined in environment.yml to obtain the required
Python version and all dependencies:

    conda env create -f environment.yml
    conda activate pyradi

Release cycle when updating pyradi
-----------------------------------
1.  Make the code changes in the git master branch and test.

2.  Bump the version in pyradi/__init__.py.

3.  Build the Sphinx documentation:

        conda activate pyradi
        K:\\WorkN\\pyradi\\pyradi-docs\\tools\\06_build_docs.bat

    This writes HTML and PDF output directly to the pyradi-docs repo.
    Commit and push the pyradi-docs repo (gh-pages branch).

4.  Commit and push the pyradi code to the pyradi repo (master branch).

5.  Update and commit any changed IPython notebooks; push to GitHub.

some of the helper code in this script comes from
https://github.com/paltman/python-setup-template/blob/master/setup.py
"""
import codecs
import os
import sys

from distutils.util import convert_path
from fnmatch import fnmatchcase
from setuptools import setup, find_packages


def read(fname):
    return codecs.open(os.path.join(os.path.dirname(__file__), fname)).read()


# Provided as an attribute, so you can append to these instead
# of replicating them:
standard_exclude = ['*.py', '*.pyc', '*$py.class', '*~', '.*', '*.bak']
standard_exclude_directories = [
    '.*', 'CVS', '_darcs', './build', './dist', 'EGG-INFO', '*.egg-info',
    'pyradi/__pycache__','pyradi/othercode'
]


# (c) 2005 Ian Bicking and contributors; written for Paste (http://pythonpaste.org)
# Licensed under the MIT license: http://www.opensource.org/licenses/mit-license.php
# Note: you may want to copy this into your setup.py file verbatim, as
# you can't import this from another package, when you don't know if
# that package is installed yet.
def find_package_data(
    where='.',
    package='',
    exclude=standard_exclude,
    exclude_directories=standard_exclude_directories,
    only_in_packages=True,
    show_ignored=False):
    """
    Return a dictionary suitable for use in ``package_data``
    in a distutils ``setup.py`` file.

    The dictionary looks like::

        {"package": [files]}

    Where ``files`` is a list of all the files in that package that
    don"t match anything in ``exclude``.

    If ``only_in_packages`` is true, then top-level directories that
    are not packages won"t be included (but directories under packages
    will).

    Directories matching any pattern in ``exclude_directories`` will
    be ignored; by default directories with leading ``.``, ``CVS``,
    and ``_darcs`` will be ignored.

    If ``show_ignored`` is true, then all the files that aren"t
    included in package data are shown on stderr (for debugging
    purposes).

    Note patterns use wildcards, or can be exact paths (including
    leading ``./``), and all searching is case-insensitive.
    """
    out = {}
    stack = [(convert_path(where), '', package, only_in_packages)]
    while stack:
        where, prefix, package, only_in_packages = stack.pop(0)
        for name in os.listdir(where):
            fn = os.path.join(where, name)
            if os.path.isdir(fn):
                bad_name = False
                for pattern in exclude_directories:
                    if (fnmatchcase(name, pattern)
                        or fn.lower() == pattern.lower()):
                        bad_name = True
                        if show_ignored:
                            print('Directory {} ignored by pattern {}'.format(fn, pattern))
                        break
                if bad_name:
                    continue
                if (os.path.isfile(os.path.join(fn, '__init__.py'))
                    and not prefix):
                    if not package:
                        new_package = name
                    else:
                        new_package = package + '.' + name
                    stack.append((fn, '', new_package, False))
                else:
                    stack.append((fn, prefix + name + '/', package, only_in_packages))
            elif package or not only_in_packages:
                # is a file
                bad_name = False
                for pattern in exclude:
                    if (fnmatchcase(name, pattern)
                        or fn.lower() == pattern.lower()):
                        bad_name = True
                        if show_ignored:
                            print('File {} ignored by pattern {}'.format(fn, pattern))
                        break
                if bad_name:
                    continue
                out.setdefault(package, []).append(prefix+name)
    return out

PACKAGE = 'pyradi'
NAME = 'pyradi'
DESCRIPTION = 'The PyRadi toolkit provides utilities for radiometry (flux flow) calculations, supporting electro-optical and infrared system design.'
AUTHOR = 'Nelis Willers'
AUTHOR_EMAIL = 'neliswillers@gmail.com'
URL = 'https://github.com/NelisW/pyradi/'
VERSION = __import__(PACKAGE).__version__

print(sys.prefix)

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=read('pyradi/README.rst'),
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license='Mozilla Public License 1.1',
    url=URL,
    packages=[PACKAGE,],
    include_package_data=True, # this will use MANIFEST.in during install where we specify additional files
    package_dir = {'pyradi': 'pyradi'},
    test_suite='nose.collector',
    tests_require=['nose'],    
    keywords='radiometry, electro-optical infrared planck',
#    install_requires=['matplotlib', 'numpy', 'scipy', 'scikit-image'],
    classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Environment :: Console',
          'Environment :: Other Environment',
          'Intended Audience :: Education',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Mozilla Public License 1.1 (MPL 1.1)',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.12',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Visualization',
          ],
    platforms='any, binary data in little-endian format',
    zip_safe=False
)

################################################################################
# Useful references
################################################################################
"""
setuptools documentation:
https://setuptools.pypa.io/en/latest/

packaging guide:
https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/

project structure:
https://docs.python-guide.org/writing/structure/
https://github.com/audreyr/cookiecutter-pypackage

setup.py classifiers list:
https://pypi.org/classifiers/

get version number from code:
https://packaging.python.org/en/latest/guides/single-sourcing-package-version/

install locally for development:
pip install -e .

including data files:
https://setuptools.pypa.io/en/latest/userguide/datafiles.html

testing:
https://docs.pytest.org/en/latest/
"""
