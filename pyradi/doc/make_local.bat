@echo off
REM Simpler one-liner alternative (no clean):
REM   conda run -n pyradi sphinx-build -b html . _build\html
conda run -n pyradi sphinx-build -b html -d _build\doctrees . _build\html
echo.
echo Open: %CD%\_build\html\index.html
