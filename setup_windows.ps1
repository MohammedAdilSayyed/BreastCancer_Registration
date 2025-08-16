# Multimodal Breast Cancer Analysis Setup Script
# PowerShell version with detailed diagnostics

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Multimodal Breast Cancer Analysis Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Checking Python installation..." -ForegroundColor Yellow
Write-Host ""

# Function to test Python command
function Test-PythonCommand {
    param([string]$Command)
    
    try {
        $result = & $Command --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ $Command found:" -ForegroundColor Green
            Write-Host "  $result" -ForegroundColor White
            return $true
        } else {
            Write-Host "✗ $Command command not found" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "✗ $Command command not found" -ForegroundColor Red
        return $false
    }
}

# Test different Python commands
Write-Host "Testing Python commands:" -ForegroundColor Yellow
Write-Host ""

$pythonCommands = @("python", "py", "python3", "python311")
$pythonFound = $false

foreach ($cmd in $pythonCommands) {
    if (Test-PythonCommand -Command $cmd) {
        $pythonFound = $true
        $workingCommand = $cmd
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Python Installation Status" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($pythonFound) {
    Write-Host "✓ Python is available" -ForegroundColor Green
    Write-Host ""
    
    # Test pip
    Write-Host "Testing pip..." -ForegroundColor Yellow
    try {
        $pipResult = & $workingCommand -m pip --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ pip is available" -ForegroundColor Green
            Write-Host "  $pipResult" -ForegroundColor White
            Write-Host ""
            
            # Show menu
            Write-Host "Would you like to:" -ForegroundColor Yellow
            Write-Host "1. Install project dependencies" -ForegroundColor White
            Write-Host "2. Create virtual environment" -ForegroundColor White
            Write-Host "3. Run project setup" -ForegroundColor White
            Write-Host "4. Show system diagnostics" -ForegroundColor White
            Write-Host "5. Exit" -ForegroundColor White
            Write-Host ""
            
            $choice = Read-Host "Enter your choice (1-5)"
            
            switch ($choice) {
                "1" { Install-Dependencies }
                "2" { Create-VirtualEnvironment }
                "3" { Run-ProjectSetup }
                "4" { Show-SystemDiagnostics }
                "5" { exit }
                default { Write-Host "Invalid choice" -ForegroundColor Red }
            }
        } else {
            Write-Host "✗ pip not found" -ForegroundColor Red
            Write-Host "Please install pip or use a different Python installation" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "✗ pip not found" -ForegroundColor Red
        Write-Host "Please install pip or use a different Python installation" -ForegroundColor Yellow
    }
} else {
    Write-Host "✗ No Python installation found" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python from one of these sources:" -ForegroundColor Yellow
    Write-Host "1. Official Python website: https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "2. Microsoft Store: Search for 'Python'" -ForegroundColor White
    Write-Host "3. Anaconda: https://www.anaconda.com/products/distribution" -ForegroundColor White
    Write-Host ""
    Write-Host "After installation, restart this script." -ForegroundColor Yellow
}

# Function to install dependencies
function Install-Dependencies {
    Write-Host ""
    Write-Host "Installing project dependencies..." -ForegroundColor Yellow
    try {
        & $workingCommand -m pip install -r requirements.txt
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
        } else {
            Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
            Write-Host "Please check the error messages above" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "✗ Failed to install dependencies" -ForegroundColor Red
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Function to create virtual environment
function Create-VirtualEnvironment {
    Write-Host ""
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    try {
        & $workingCommand -m venv breast_cancer_env
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Virtual environment created" -ForegroundColor Green
            Write-Host ""
            Write-Host "To activate the environment, run:" -ForegroundColor Yellow
            Write-Host "breast_cancer_env\Scripts\Activate.ps1" -ForegroundColor White
            Write-Host ""
            Write-Host "Then install dependencies:" -ForegroundColor Yellow
            Write-Host "pip install -r requirements.txt" -ForegroundColor White
        } else {
            Write-Host "✗ Failed to create virtual environment" -ForegroundColor Red
        }
    } catch {
        Write-Host "✗ Failed to create virtual environment" -ForegroundColor Red
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Function to run project setup
function Run-ProjectSetup {
    Write-Host ""
    Write-Host "Running project setup..." -ForegroundColor Yellow
    try {
        & $workingCommand setup_project.py
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Project setup completed" -ForegroundColor Green
        } else {
            Write-Host "✗ Project setup failed" -ForegroundColor Red
            Write-Host "Please check the error messages above" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "✗ Project setup failed" -ForegroundColor Red
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Function to show system diagnostics
function Show-SystemDiagnostics {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "System Diagnostics" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    # OS Information
    Write-Host "Operating System:" -ForegroundColor Yellow
    $os = Get-WmiObject -Class Win32_OperatingSystem
    Write-Host "  $($os.Caption) $($os.OSArchitecture)" -ForegroundColor White
    Write-Host ""
    
    # Python Information
    Write-Host "Python Information:" -ForegroundColor Yellow
    try {
        $pythonVersion = & $workingCommand --version 2>$null
        Write-Host "  Version: $pythonVersion" -ForegroundColor White
        
        $pythonPath = (Get-Command $workingCommand).Source
        Write-Host "  Path: $pythonPath" -ForegroundColor White
    } catch {
        Write-Host "  Could not get Python information" -ForegroundColor Red
    }
    Write-Host ""
    
    # PATH Information
    Write-Host "PATH Environment Variable:" -ForegroundColor Yellow
    $paths = $env:PATH -split ';'
    $pythonPaths = $paths | Where-Object { $_ -like "*Python*" }
    if ($pythonPaths) {
        foreach ($path in $pythonPaths) {
            Write-Host "  $path" -ForegroundColor White
        }
    } else {
        Write-Host "  No Python paths found in PATH" -ForegroundColor Red
    }
    Write-Host ""
    
    # Available Python installations
    Write-Host "Available Python Installations:" -ForegroundColor Yellow
    $possiblePaths = @(
        "$env:LOCALAPPDATA\Programs\Python",
        "C:\Python*",
        "C:\Program Files\Python*",
        "C:\Program Files (x86)\Python*"
    )
    
    foreach ($basePath in $possiblePaths) {
        if (Test-Path $basePath) {
            $pythonDirs = Get-ChildItem -Path $basePath -Directory -ErrorAction SilentlyContinue
            foreach ($dir in $pythonDirs) {
                $pythonExe = Join-Path $dir.FullName "python.exe"
                if (Test-Path $pythonExe) {
                    Write-Host "  $($dir.FullName)" -ForegroundColor White
                }
            }
        }
    }
    Write-Host ""
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "For more help, see TROUBLESHOOTING.md" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to exit" 