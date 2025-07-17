@echo off
setlocal enabledelayedexpansion

if "%1"=="" (
    echo Usage: run_experiment.bat [number_of_runs]
    exit /b 1
)

set n=%1
echo Running genetic algorithm %n% times...

for /l %%i in (1,1,%n%) do (
    echo.
    echo Run %%i of %n%
    python genetic_algorithm.py --config config_COTS.json
    if errorlevel 1 (
        echo Run %%i failed
    ) else (
        echo Run %%i completed
    )
)

echo.
echo All runs completed
