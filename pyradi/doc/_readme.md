# How to create the docs

## NB!

To build the documentation the pyradi module files must be reachable via the `PYTHONPATH`. The easiest way to to this is to place a file with the module name and extension `.pth` in the `/Lib/site-packages` directory.  The `.pth` file must contain a path to the location 
where the module files are kept, such that the module directory name must not be  in the specified path, i.e., enter a path one level higher than the module directory.


The present version of pyradi requires Python 3 to build the documentation, because the code only runs under Python 3.  

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

To build the html and PDF documentation run the command

	K:\WorkN\pyradi\pyradi-docs\tools\06_build_docs.bat.

and follow the instructions at the end of the batch file. 
   

## References

<https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site>  
<https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site>  
<https://docs.github.com/en/pages/getting-started-with-github-pages/about-github-pages>  
