@echo off
REM Recreate the pyradi conda environment from environment.yml and run all tests.
REM Run this from the pyradi/ directory:
REM     cd K:\WorkN\pyradi\pyradi\pyradi
REM     recreate_env_and_test.bat

setlocal

set ENV_NAME=pyradi
set SCRIPT_DIR=%~dp0

echo.
echo === Step 1: Remove existing environment (if any) ===
call conda env remove -n %ENV_NAME% -y 2>nul

echo.
echo === Step 2: Create fresh environment from environment.yml ===
call conda env create -f "%SCRIPT_DIR%environment.yml"
if errorlevel 1 (
    echo ERROR: conda env create failed.
    exit /b 1
)

echo.
echo === Step 3: Run full test suite ===
call conda run -n %ENV_NAME% --no-capture-output ^
    pytest ^
    test\test_ryutils.py ^
    test\test_ryfiles.py ^
    test\test_ry3dnoise.py ^
    test\test_rychroma.py ^
    test\test_rydetector.py ^
    test\test_rylookup.py ^
    test\test_rymodtran.py ^
    test\test_ryhdf5.py ^
    test\test_rypflux.py ^
    test\test_ryplanck.py ^
    test\test_ryprob.py ^
    test\test_ryplotspherical.py ^
    test\test_ryptw.py ^
    test\test_ryscene.py ^
    test\test_rystare.py ^
    test\test_rytarggen.py ^
    -v --tb=short

endlocal
