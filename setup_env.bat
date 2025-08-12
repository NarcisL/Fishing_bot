@echo off
echo Activating Python environment...
call .venv\Scripts\activate.bat

echo.
echo Environment activated! You can now run:
echo   python fishing.py       (for the main fishing bot)
echo   python initc_fish.py    (for the initialization fishing bot)
echo.
echo To deactivate the environment, type: deactivate
echo.

cmd /k
