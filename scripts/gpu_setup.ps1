# PowerShell script to set GPU persistence mode and lock clock speed for maximum performance.
# Note: Must be run as Administrator.

Write-Output "=== GPU performance locking script ==="

# Check for admin privileges
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Warning "This script is not running as Administrator. GPU clock locking and persistence mode require admin rights."
    Write-Warning "Please re-run this script in an Administrator PowerShell prompt."
    # We will still try to query nvidia-smi
}

# Find nvidia-smi path
$nvidiaSmiPath = "nvidia-smi"
# Check common paths if not in PATH
if (-not (Get-Command $nvidiaSmiPath -ErrorAction SilentlyContinue)) {
    $commonPath = "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    if (Test-Path $commonPath) {
        $nvidiaSmiPath = $commonPath
    } else {
        Write-Error "nvidia-smi was not found on your system. Make sure NVIDIA Drivers are installed."
        exit 1
    }
}

Write-Output "Found nvidia-smi at: $nvidiaSmiPath"

# 1. Enable Persistence Mode (if supported on Geforce RTX 5060)
Write-Output "Attempting to enable GPU persistence mode..."
& $nvidiaSmiPath -pm 1 2>&1 | Out-String | ForEach-Object {
    if ($_ -match "Failed" -or $_ -match "not supported") {
        Write-Warning "Persistence mode is not supported on this GPU or driver: $_"
    } else {
        Write-Output "Persistence mode enabled successfully."
    }
}

# 2. Query Supported Clocks
Write-Output "Querying supported GPU graphics clocks..."
$clocks = & $nvidiaSmiPath --query-gpu=clocks.max.graphics --format=csv,noheader,nounits 2>&1
if ($clocks -match "Failed" -or $clocks -match "error") {
    Write-Warning "Could not query max graphics clock: $clocks"
    $maxClock = 1455 # Fallback based on RTX 5060 mobile specs
} else {
    $maxClock = [int]$clocks.Trim()
    Write-Output "Detected maximum graphics clock: $maxClock MHz"
}

# 3. Lock GPU Clocks
# We lock it to the max graphics clock
Write-Output "Attempting to lock GPU graphics clock to $maxClock MHz..."
$lockCmd = & $nvidiaSmiPath --lock-gpu-clocks=$maxClock,$maxClock 2>&1 | Out-String
if ($lockCmd -match "Failed" -or $lockCmd -match "not support" -or $lockCmd -match "Permission") {
    Write-Warning "Failed to lock GPU clocks: $lockCmd"
    Write-Warning "Common causes: not running as admin, or GeForce cards might require a newer driver / may restrict clock locking."
} else {
    Write-Output "Successfully locked GPU clocks to $maxClock MHz."
}

# 4. Show current status
Write-Output "=== Current GPU Status ==="
& $nvidiaSmiPath --query-gpu=name,power.draw,clocks.gr,clocks.mem,temp.gpu --format=csv
