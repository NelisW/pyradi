# How to create the docs

## NB!

To build the documentation the pyradi module files must be reachable via the `PYTHONPATH`. The easiest way to to this is to place a file with the module name and extension `.pth` in the `/Lib/site-packages` directory.  The `.pth` file must contain a path to the location 
where the module files are kept, such that the module directory name must not be  in the specified path, i.e., enter a path one level higher than the module directory.


The present version of pyradi requires Python 2.7 to build the documentation, because the code only runs under Python 2.7.  If your Python 2.7 is in an environment, activate the environment.

## Create the documentation

The pyradi documentation build directory is currently set up to build in an external directory outside of the main pyradi directory. To set up this directory

1. Building the documentation requires the sphinx documentation toolset.

1. Be careful to clone the <font color="red"> `gh-pages` branch </font> from `https://github.com/NelisW/pyradi-docs/tree/gh-pages`, 
to be on the same level as the main pyradi directory:  

		+-pyradi
		  +-.git
		  |-setup.py
		  | ...
		  +-pyradi
		    + ... all the pyradi files
		+-pyradi-docs
		  +-.git
		  +-_build

1.  To build the html documentation open a command window in the `doc`  directory
and execute the command `make html`.

1. To build the pdf documentation open a command window in the `doc` directory
and execute the command `make latex`, and then build the pdf from the latex
files in `pyradi-docs/_build/latex`.

1. Commit all files to the `gh-pages` branch.

## References

<https://help.github.com/articles/creating-project-pages-manually/>  
<https://help.github.com/articles/creating-pages-with-the-automatic-generator/>  
<https://help.github.com/articles/user-organization-and-project-pages/>  
