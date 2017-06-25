"""
Setup script for pyradi. This procedure must be followed only once so set up.

0. Make a list of all the folders included and excluded in the PyPi package in MANIFEST.in

1.  Set up a user account with pyPI or testpyPI. The passwords may only contain 
    letters and numbers, not other symbols.

2.  On Windows create a HOME environmental variable to your home directory (C:\\Users\\username)

3.  Place the .pypirc file in your home directory (to where HOME points). The format
    of this file is as follows (no spaces in last two lines):
    ---------------------------------------------------------
    [distutils]
    index-servers =
        pypi

    [pypi]
    username:yourPyPIusername
    password:yourPyPIpassword    
    ---------------------------------------------------------
    you can remove the password, it will be prompted.

4.  Write the setup.py script for project (this file)

5.  Write and test the long description, this goes onto PyPI project page.

6. If not already existing, mkdir ../pyradi-docs/_build. Alternatively clone the
   gh-pages branch from https://github.com/NelisW/pyradi-docs/tree/gh-pages
   To be on the same level as the main pyradi directory:In other words,
   create a new folder on the same level as where pyradi root is:

    ..
    +-pyradi  [https://github.com/NelisW/pyradi]
      +-.git
      |-setup.py (this file)
      | ...
      +-pyradi
        + ... all the pyradi files

    +-pyradi-docs [https://github.com/NelisW/pyradi-data] (this is an optional clone) 
      +-.git
      +-_build

    +-pyradi-data [https://github.com/NelisW/pyradi-docs] (this is an optional clone) 
      +-.git
      +-regression
      +-images

7.  On every new release:  Register the meta data with PyPI    
    cd to where setup.py is located and run the following command:
    python setup.py register   [[this will register new release on pyPI]]

8.  Create the package files zip and installers (not uploading)
    python setup.py sdist bdist_wininst [[ built in dist directory ]]

9.  Test locally by unzipping the package and running
    python setup.py install  [[ will install to site-packages]]
    test this installation prior to uploading to pyPI

10.  Upload the packages to PyPIy
    python setup.py sdist bdist_wininst upload [[ built in dist directory ]]

11. Installing a package from the server to local pyPI
    pip install --upgrade pyradi

See the bottom of this file for additional notes.

Release cycle when updating PyRadi

1.  Make the code changes required in the git master branch.
    Test as necessary.

2.  Raise the software version one count in __init()__.py

3.  Build the Python Sphinx documentation by changing to the 
    /pyradi/pyradi/doc directory.
    Make sure that the gh-pages branch is checked out in the 
    /pyradi/pyradi/doc/../../../pyradi-docs directory.

    Create both the HTML and PDF docs.
    See that the documentation is correct (spot checks).
    cd to /pyradi/pyradi/doc
    make html
    make latex
    cd to /pyradi/pyradi/doc/_build/latex
    pdflatex pyradi.tex

    This procedure will create the docs in the build directory
    /pyradi/pyradi/doc/../../../pyradi-docs/_build.
    which should contain a clone of the gh-pages repo at 
    https://github.com/NelisW/pyradi-docs/tree/gh-pages.
    Commit the changes to the pyradi-docs repo.

4a. Commit the pyradi code to the pyradi repo, master branch.

4b. Commit pyradi-docs to the pyradi-docs repo, gh-pages branch.

4c. Commit the pyradi-data repo as and if required.

5.  Update the IPython notebook documentation in the local git clone.
    Do a complete rebuild of the notebooks: restart kernel and run all cells.
    Check that all cells are rendered.

6.  If a new notebook name is used, get the nbviewer link to the html rendering.

7.  Update README.md with the new notebook name.

8.  Commit the ipynb files to the local git repo.

9. Push the local git repo changes to github repo.

--  NB make sure the local IE proxy is not set in IE.  Must be 'no proxy'

10. Register the new version on PyPI by one of the two following methods:

10a. cd to root pyradi directory and do in a command window:
    python setup.py register   [[this will register new release on pyPI]]

10b. Log in on the PyPi website and manually register the new release, 
     to create a new index (release).    

11. Create the package files zip and installers (not uploading)
    python setup.py sdist bdist_wininst [[ built in dist directory ]]

12. Test locally by unzipping the package and running
    python setup.py install  [[ will install to site-packages]]
    test this installation prior to uploading to pyPI

13 Upload the packages to PyPIy, by one of two methods:

13a. python setup.py sdist bdist_wininst upload [[ built in dist directory ]]

13b. Log in to the PyPi website and manually upload the files against new 
     release that was registered in 10b above.
     
    
some of the code in this script below comes from 
https://github.com/paltman/python-setup-template/blob/master/setup.py
I am just not using much of it :-((
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
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Physics',
          'Topic :: Scientific/Engineering :: Visualization',
          ],
    platforms='any, binary data in little-endian format',
    zip_safe=False
)

################################################################################
################################################################################
################################################################################
"""
General:
http://www.jeffknupp.com/blog/2013/08/16/open-sourcing-a-python-project-the-right-way/
http://docs.python-guide.org/en/latest/writing/structure/
https://github.com/audreyr/cookiecutter-pypackage
https://coderwall.com/p/lt2kew
http://www.scotttorborg.com/python-packaging/
http://pythonhosted.org/setuptools/setuptools.html#development-mode
http://www.ibm.com/developerworks/library/os-pythonpackaging/index.html?ca=rec_art
http://svn.python.org/projects/sandbox/trunk/setuptools/setuptools.txt
http://stackoverflow.com/questions/6344076/differences-between-distribute-distutils-and-setuptools

use setuptools: 
https://stackoverflow.com/questions/6344076/differences-between-distribute-distutils-setuptools-and-distutils2
http://lucumr.pocoo.org/2012/6/22/hate-hate-hate-everywhere/

.pypirc
https://stackoverflow.com/questions/1569315/setup-py-upload-is-failing-with-upload-failed-401-you-must-be-identified-t/1569331#1569331'
https://pythonhosted.org/an_example_pypi_project/setuptools.html
https://wiki.python.org/moin/TestPyPI
http://guide.python-distribute.org/contributing.html

setup.py script 
http://docs.python.org/2.7/distutils/setupscript.html#
http://docs.python.org/2.7/distutils/introduction.html#distutils-simple-example
https://pypi.python.org/pypi?:action=list_classifiers
http://docs.python.org/2/distutils/sourcedist.html#manifest
http://docs.python.org/2/distutils/setupscript.html#meta-data

check the long description by
python setup.py --long-description | rst2html.py > output.html
http://docs.python.org/2.7/distutils/packageindex.html#uploading-packages

register the meta data with the server listed in -r
python setup.py register -r https://testpypi.python.org/pypi
https://wiki.python.org/moin/TestPyPI
http://docs.python.org/2.7/distutils/packageindex.html#registering-packages

build the packages for local testing
python setup.py sdist bdist_wininst upload

build and upload the package files to server at -r
python setup.py sdist  upload -r https://testpypi.python.org/pypi
python setup.py sdist bdist_wininst upload -r https://testpypi.python.org/pypi
python setup.py register sdist upload -r https://testpypi.python.org/pypi

upload previously built to server at -r
python setup.py upload -r https://testpypi.python.org/pypi
http://docs.python.org/2.7/distutils/packageindex.html#uploading-packages

Uploading source and installers with a batch script
https://pythonhosted.org/an_example_pypi_project/setuptools.html

Upload for each version of python you want to distribute
https://pythonhosted.org/an_example_pypi_project/setuptools.html

get version number from the code files
http://www.ibm.com/developerworks/library/os-pythonpackaging/index.html?ca=rec_art
http://legacy.python.org/dev/peps/pep-0386/

9) Installing a package from server at i
pip install --upgrade -i https://testpypi.python.org/pypi pyradi2
http://www.pip-installer.org/en/latest/reference/pip_wheel.html#allow-external

install locally on your pc
$ python setup.py install <will install from here to python directory>

including data
include_package_data
http://svn.python.org/projects/sandbox/trunk/setuptools/setuptools.txt
http://www.scotttorborg.com/python-packaging/metadata.html

using manifest file
http://danielsokolowski.blogspot.com/2012/08/setuptools-includepackagedata-option.html
http://docs.python.org/2/distutils/sourcedist.html#manifest-template 
  
testing
http://www.scotttorborg.com/python-packaging/testing.html

requirements
https://caremad.io/blog/setup-vs-requirement/
"""
