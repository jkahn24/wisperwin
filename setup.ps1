$ErrorActionPreference = "Stop"
$ToolDir = $PSScriptRoot
# Python 3.10+ from the py launcher, else python on PATH. To pick one: set the PYTHON env var to its full path.
$Python = if ($env:PYTHON) { $env:PYTHON } elseif (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }

Write-Host "Setting up WisperWin venv..."
& $Python -m venv "$ToolDir\venv"
& "$ToolDir\venv\Scripts\python.exe" -m pip install --upgrade pip
& "$ToolDir\venv\Scripts\pip.exe" install -r "$ToolDir\requirements.txt"
Write-Host "Done. Launcher will use venv\Scripts\pythonw.exe automatically."
