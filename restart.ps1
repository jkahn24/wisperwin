# WisperWin restart — only kills WisperWin processes, leaves other Python apps alone.
# Note: the CommandLine filter matches BOTH wisperwin.py and launcher.py (the
# folder path contains "wisperwin"), so both are killed — and both must be
# restarted. The old version only restarted wisperwin.py, silently leaving the
# OS-level hotkey (launcher) dead.
$procs = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -like "*python*" -and $_.CommandLine -like "*wisperwin*"
}
foreach ($p in $procs) {
    Write-Output "Stopping WisperWin PID $($p.ProcessId)"
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep 1

# Prefer the tool's venv pythonw so both processes run in the isolated env.
$pythonw = Join-Path $PSScriptRoot "venv\Scripts\pythonw.exe"
if (-not (Test-Path $pythonw)) { $pythonw = "pythonw" }

Start-Process $pythonw -ArgumentList "`"$PSScriptRoot\launcher.py`"" -WorkingDirectory $PSScriptRoot
Start-Process $pythonw -ArgumentList "`"$PSScriptRoot\wisperwin.py`"" -WorkingDirectory $PSScriptRoot
Write-Output "WisperWin launcher + app started"
