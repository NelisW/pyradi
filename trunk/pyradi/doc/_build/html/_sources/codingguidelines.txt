Coding Guidelines
******************

.. include global.rst

Broadly speaking we adhere to the Google Python Style Guide, but not always.
The style guide is available at
http://google-styleguide.googlecode.com/svn/trunk/pyguide.html. This style is 
based on Python's PEP 8 http://www.python.org/dev/peps/pep-0008/.

Naming Rules
-----------------

We deviate from PEP 8 / Google's naming rules as shown here. Essentially we avoid
underscores inside names, and prefer to Capitalise words to highlight.  The primary
motivation is (in our opinion) improved readability: it better binds the words
into a single entity. Underscores tend to break the name visually into separate
sub-names.  


===========================  ====================================  ================================================================= ======================
Type                          Public                                Internal                                                           PEP 8
===========================  ====================================  ================================================================= ======================
Packages                     lowerwordslater                                                                                          lower_with_under 
Modules                      lowerwordslater                        _lowerwordslater                                                  lower_with_under
Classes                      CapWordsLater                          _CapWordsLater                                                    CapWords
Exceptions                   CapWordsLater                                                                                            CapWords
Functions                    lowerWordsLater()                      _lowerWordsLater()                                                lower_with_under()
Global/Class Constants       CAPS_WITH_UNDER                        _CAPS_WITH_UNDER                                                  CAPS_WITH_UNDER
Global/Class Variables       lowerWordsLater                        _lowerWordsLater                                                  lower_with_under
Instance Variables           lowerWordsLater                        _lowerWordsLater (protected) or __lowerWordsLater (private)       lower_with_under
Method Names                 lowerWordsLater()                      _lowerWordsLater() (protected) or __lowerWordsLater() (private)   lower_with_under()
Function/Method Parameters   lowerWordsLater                                                                                          lower_with_under
Local Variables              lowerWordsLater                                                                                          lower_with_under
===========================  ====================================  ================================================================= ======================



