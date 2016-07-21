# How to create the docs

## NB!

To build the documentation the module files must be reachable via the `PYTHONPATH`.
The easiest way to to this is to place a file with the module name and extension `.pth`
in the `/Lib/site-packages` directory.  The `.pth` file must contain a path to the location 
where the module files are kept, such that the module directory name must not be 
in the specified path, i.e., enter a path one level higher than the module directory.

## Create the documentation

The pyradi documentation build directory is currently set up to build in an external directory
outside of the main pyradi directory. To set up this directory

1. Building the documentation requires the sphinx documentation toolset.

1. clone the `gh-pages` branch from `https://github.com/NelisW/pyradi-docs/tree/gh-pages`, 
to be on the same level as the main pyradi directory:  
		+-pyradi
		  +-.git
		  |-setup.py (this file)
		  | ...
		  +-pyradi
		    + ... all the pyradi files
		+-pyradi-docs
		  +-.git
		  +-_build

1.  To build the html documentation open a command window in this directory
and execute the command `make html`.

1. To build the pdf documentation open a command window in this directory
and execute the command `make latex`, and then build the pdf from the latex
files in `pyradi/doc/_build/latex`.

1. Commit all files to the `gh-pages` branch.

