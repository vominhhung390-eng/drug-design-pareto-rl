param(
    [string]$Python = 'python',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Output) { $Output = Join-Path $Root 'results\paper_tables' }
& $Python (Join-Path $PSScriptRoot 'build_paper_tables.py') --output $Output
if ($LASTEXITCODE -ne 0) { throw "Paper table build failed with exit code $LASTEXITCODE" }
