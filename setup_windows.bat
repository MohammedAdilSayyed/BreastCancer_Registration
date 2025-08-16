@echo off
echo ========================================
echo Multimodal Breast Cancer Analysis Setup
echo ========================================
echo.

echo Checking Python installation...
echo.

REM Try different Python commands
echo Testing Python commands:
echo.

echo 1. Testing 'python'...
python --version 2>nul
if %errorlevel% equ 0 (
    echo ✓ Python found: 
    python --version
) else (
    echo ✗ 'python' command not found
)

echo.
echo 2. Testing 'py'...
py --version 2>nul
if %errorlevel% equ 0 (
    echo ✓ Python launcher found:
    py --version
) else (
    echo ✗ 'py' command not found
)

echo.
echo 3. Testing 'python3'...
python3 --version 2>nul
if %errorlevel% equ 0 (
    echo ✓ Python3 found:
    python3 --version
) else (
    echo ✗ 'python3' command not found
)

echo.
echo 4. Testing 'python311'...
python311 --version 2>nul
if %errorlevel% equ 0 (
    echo ✓ Python311 found:
    python311 --version
) else (
    echo ✗ 'python311' command not found
)

echo.
echo ========================================
echo Python Installation Status
echo ========================================

REM Check if any Python is available
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ Python is available
    echo.
    echo Testing pip...
    pip --version 2>nul
    if %errorlevel% equ 0 (
        echo ✓ pip is available
        echo.
        echo Would you like to:
        echo 1. Install project dependencies
        echo 2. Create virtual environment
        echo 3. Run project setup
        echo 4. Exit
        echo.
        set /p choice="Enter your choice (1-4): "
        
        if "%choice%"=="1" goto install_deps
        if "%choice%"=="2" goto create_venv
        if "%choice%"=="3" goto run_setup
        if "%choice%"=="4" goto end
        goto end
    ) else (
        echo ✗ pip not found
        echo Please install pip or use a different Python installation
    )
) else (
    echo ✗ No Python installation found
    echo.
    echo Please install Python from one of these sources:
    echo 1. Official Python website: https://www.python.org/downloads/
    echo 2. Microsoft Store: Search for "Python"
    echo 3. Anaconda: https://www.anaconda.com/products/distribution
    echo.
    echo After installation, restart this script.
)

goto end

:install_deps
echo.
echo Installing project dependencies...
pip install -r requirements.txt
if %errorlevel% equ 0 (
    echo ✓ Dependencies installed successfully
) else (
    echo ✗ Failed to install dependencies
    echo Please check the error messages above
)
goto end

:create_venv
echo.
echo Creating virtual environment...
python -m venv breast_cancer_env
if %errorlevel% equ 0 (
    echo ✓ Virtual environment created
    echo.
    echo To activate the environment, run:
    echo breast_cancer_env\Scripts\activate
    echo.
    echo Then install dependencies:
    echo pip install -r requirements.txt
) else (
    echo ✗ Failed to create virtual environment
)
goto end

:run_setup
echo.
echo Running project setup...
python setup_project.py
if %errorlevel% equ 0 (
    echo ✓ Project setup completed
) else (
    echo ✗ Project setup failed
    echo Please check the error messages above
)
goto end

:end
echo.
echo ========================================
echo Setup Complete
echo ========================================
echo.
echo For more help, see TROUBLESHOOTING.md
echo.
pause 