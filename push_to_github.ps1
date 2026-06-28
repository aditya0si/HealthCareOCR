# push_to_github.ps1
# Script to initialize git repository and push the clean codebase to GitHub.

# Ensure git is installed
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Error "Git CLI is not installed or not in PATH. Please install Git and try again."
    exit 1
}

# 1. Initialize local repository if not already done
if (!(Test-Path .git)) {
    Write-Host "Initializing local Git repository..."
    git init
} else {
    Write-Host "Local Git repository already initialized."
}

# 2. Configure remote origin
$remoteUrl = "https://github.com/aditya0si/HealthCareOCR.git"
$remotes = git remote
if ($remotes -contains "origin") {
    Write-Host "Setting remote origin URL to $remoteUrl..."
    git remote set-url origin $remoteUrl
} else {
    Write-Host "Adding remote origin $remoteUrl..."
    git remote add origin $remoteUrl
}

# 3. Rename branch to main
Write-Host "Setting main branch..."
git branch -M main

# 4. Stage files
Write-Host "Staging files (respecting .gitignore)..."
git add .

# Print status of staged files to verify no sensitive data is staged
Write-Host "`nStaged Files Summary:"
git status

# 5. Commit
Write-Host "`nCommitting files..."
git commit -m "Initial commit: clean HealthCareOCR codebase without patient data"

# 6. Push
Write-Host "`nPushing to GitHub..."
Write-Host "Note: If this fails or prompts for credentials, please configure your GitHub credentials / PAT."
git push -u origin main
