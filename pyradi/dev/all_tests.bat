@echo off
REM Recreate the pyradi conda environment from environment.yml and run all tests.
REM Run this from anywhere — the script locates paths relative to itself.
REM
REM Usage:
REM     K:\WorkN\pyradi\pyradi\pyradi\dev\all_tests.bat

setlocal

set ENV_NAME=pyradi
REM SCRIPT_DIR = dev\  ->  PYRADI_DIR = one level up (pyradi\pyradi\pyradi\)
set SCRIPT_DIR=%~dp0
set PYRADI_DIR=%SCRIPT_DIR%..\

echo.
echo === Run full test suite ===
call conda run -n %ENV_NAME% --no-capture-output ^
    pytest ^
    "%PYRADI_DIR%test\test_ryutils.py" ^
    "%PYRADI_DIR%test\test_ryfiles.py" ^
    "%PYRADI_DIR%test\test_ry3dnoise.py" ^
    "%PYRADI_DIR%test\test_rychroma.py" ^
    "%PYRADI_DIR%test\test_rydetector.py" ^
    "%PYRADI_DIR%test\test_rylookup.py" ^
    "%PYRADI_DIR%test\test_rymodtran.py" ^
    "%PYRADI_DIR%test\test_ryhdf5.py" ^
    "%PYRADI_DIR%test\test_rypflux.py" ^
    "%PYRADI_DIR%test\test_ryplanck.py" ^
    "%PYRADI_DIR%test\test_ryprob.py" ^
    "%PYRADI_DIR%test\test_ryplotspherical.py" ^
    "%PYRADI_DIR%test\test_ryplot.py" ^
    "%PYRADI_DIR%test\test_ryptw.py" ^
    "%PYRADI_DIR%test\test_ryscene.py" ^
    "%PYRADI_DIR%test\test_rystare.py" ^
    "%PYRADI_DIR%test\test_rytarggen.py" ^
    -v --tb=short

endlocal
