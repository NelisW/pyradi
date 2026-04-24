@echo off
REM Build pyradi Sphinx HTML documentation
REM Run this from the doc\ folder inside the conda pyradi environment:
REM   conda activate pyradi
REM   cd K:\WorkN\pyradi\pyradi\pyradi\doc
REM   build_docs.bat

setlocal

REM Output folder (inside doc\_build\html)
set BUILDDIR=_build

REM Check sphinx-build is available
where sphinx-build >nul 2>&1
if errorlevel 1 (
    echo sphinx-build not found. Installing...
    pip install sphinx
)

REM Clean previous build
if exist %BUILDDIR% rmdir /s /q %BUILDDIR%

REM Build HTML
sphinx-build -b html -d %BUILDDIR%\doctrees . %BUILDDIR%\html

if errorlevel 1 (
    echo.
    echo BUILD FAILED. Check the output above for errors.
    exit /b 1
)

echo.
echo ============================================================
echo Build complete.
echo Open: %CD%\_build\html\index.html
echo ============================================================
