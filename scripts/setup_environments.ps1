param(
    [string]$Config = '',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Config) { $Config = Join-Path $Root 'config\reproduction_pipeline.json' }
$Protocol = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
function Resolve-ProjectPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return $Value }
    return [IO.Path]::GetFullPath((Join-Path $Root $Value))
}
function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    if ($DryRun) { Write-Output ("DRYRUN " + $Executable + ' ' + ($Arguments -join ' ')); return }
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Environment command failed with exit code ${LASTEXITCODE}: $Executable" }
}
function New-CondaEnvironment([string]$PythonPath, [string]$PythonVersion) {
    if (Test-Path -LiteralPath $PythonPath) { return }
    $Prefix = Split-Path -Parent (Split-Path -Parent $PythonPath)
    Invoke-Checked ([string]$Protocol.executables.conda) @('create', '-y', '--prefix', $Prefix, "python=$PythonVersion", 'pip')
}
function New-CleanRequirements([string]$Source, [string]$Name, [string[]]$ExcludePatterns) {
    $Output = Join-Path $Root "results\reproduction\setup\$Name"
    if ($DryRun) { return $Output }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Output) | Out-Null
    $Lines = Get-Content -LiteralPath $Source | Where-Object {
        $Line = $_
        -not (@($ExcludePatterns | Where-Object { $Line -match $_ }).Count)
    }
    $Lines | Set-Content -LiteralPath $Output -Encoding UTF8
    return $Output
}

$Common = Resolve-ProjectPath ([string]$Protocol.executables.clover_python)
$MoTorch = Resolve-ProjectPath ([string]$Protocol.executables.mo_lso_python)
$MoGp = Resolve-ProjectPath ([string]$Protocol.executables.mo_lso_gp_python)
$Docking = Resolve-ProjectPath ([string]$Protocol.executables.docking_python)
New-CondaEnvironment $Common '3.10.20'
New-CondaEnvironment $MoTorch '3.10.20'
New-CondaEnvironment $MoGp '3.10.20'
New-CondaEnvironment $Docking '3.10.20'

$CommonReq = New-CleanRequirements (Join-Path $Root 'environments\requirements_snapshot_clover_mol.txt') 'common_requirements.txt' @('^polygon==')
Invoke-Checked $Common @('-m', 'pip', 'install', '--extra-index-url', 'https://download.pytorch.org/whl/cu128', '-r', $CommonReq)
$GraphReq = New-CleanRequirements (Join-Path $Root 'environments\requirements_snapshot_graphpareto_nsga2.txt') 'graphpareto_requirements.txt' @('^polygon==')
Invoke-Checked $Common @('-m', 'pip', 'install', '--extra-index-url', 'https://download.pytorch.org/whl/cu128', '-r', $GraphReq)
Invoke-Checked $Common @('-m', 'pip', 'install', '--editable', (Join-Path $Root 'vendor\polygon-main'))

$MoReq = New-CleanRequirements (Join-Path $Root 'environments\requirements_snapshot_mo_lso_torch.txt') 'mo_lso_torch_requirements.txt' @('^polygon==')
Invoke-Checked $MoTorch @('-m', 'pip', 'install', '--extra-index-url', 'https://download.pytorch.org/whl/cu128', '-r', $MoReq)
Invoke-Checked $MoGp @('-m', 'pip', 'install', '-r', (Join-Path $Root 'environments\requirements_snapshot_mo_lso_gpflow.txt'))
Invoke-Checked $Docking @('-m', 'pip', 'install', '-r', (Join-Path $Root 'environments\requirements_snapshot_docking.txt'))

if (-not (Test-Path -LiteralPath (Resolve-ProjectPath ([string]$Protocol.executables.reinvent_python)))) {
    Invoke-Checked $Common @('-m', 'pip', 'install', 'uv')
    $Uv = Join-Path (Split-Path -Parent $Common) 'uv.exe'
    Invoke-Checked $Uv @('sync', '--project', (Join-Path $Root 'baselines\reinvent4_adapter'), '--frozen')
}
Write-Output 'ENVIRONMENT_SETUP_COMPLETE'
