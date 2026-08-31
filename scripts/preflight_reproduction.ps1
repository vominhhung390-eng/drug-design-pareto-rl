param(
    [string]$Config = '',
    [switch]$AllowRecoveredEgfrVegfr2,
    [switch]$SkipExecutableChecks,
    [switch]$Strict
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Config) { $Config = Join-Path $Root 'config\reproduction_pipeline.json' }
$Protocol = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
function Resolve-ProjectPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return $Value }
    return [IO.Path]::GetFullPath((Join-Path $Root $Value))
}
$Checks = @()
function Add-Check([string]$Name, [bool]$Pass, [string]$Detail, [string]$Severity = 'error') {
    $script:Checks += [pscustomobject]@{ name = $Name; pass = $Pass; severity = $Severity; detail = $Detail }
}
$Dataset = Resolve-ProjectPath ([string]$Protocol.dataset.path)
Add-Check 'dataset_exists' (Test-Path -LiteralPath $Dataset) $Dataset
if (Test-Path -LiteralPath $Dataset) {
    $Hash = (Get-FileHash -LiteralPath $Dataset -Algorithm SHA256).Hash.ToLowerInvariant()
    Add-Check 'dataset_sha256' ($Hash -eq ([string]$Protocol.dataset.sha256).ToLowerInvariant()) "actual=$Hash"
}
foreach ($Relative in @(
    'vendor\polygon-main\polygon\run.py',
    'method\ablation\run_wc_two_targets_multiexplore.py',
    'baselines\polygon_adapter\adapter_optimize_dual_oracle.py',
    'baselines\reinvent4_adapter\adapter_optimize_dual_oracle.py',
    'baselines\drugex_v2_adapter\adapter_optimize_dual_oracle.py',
    'baselines\mo_lso_adapter\adapter_optimize_dual_oracle.py',
    'baselines\graphpareto_nsga2_adapter\adapter_optimize_dual_oracle.py',
    'docking\scripts\run_seed_top10_vina.py'
)) {
    $Path = Join-Path $Root $Relative
    Add-Check "required_$Relative" (Test-Path -LiteralPath $Path) $Path
}
foreach ($Relative in @(
    'data\predictor_target_pairs\01_EGFR_VEGFR2_第一组_恢复相关数据_NOT_EXACT_ORIGINAL\EGFR_P00533_BindingDB_API_snapshot_20260712.json',
    'data\predictor_target_pairs\01_EGFR_VEGFR2_第一组_恢复相关数据_NOT_EXACT_ORIGINAL\VEGFR2_P35968_BindingDB_API_snapshot_20260712.json',
    'data\predictor_target_pairs\02_PARP1_BRD4_第二组_当前正式预测器数据_ChEMBL37\PARP1_CHEMBL3105_train_through_2023_n2538.csv',
    'data\predictor_target_pairs\02_PARP1_BRD4_第二组_当前正式预测器数据_ChEMBL37\BRD4_CHEMBL1163125_train_through_2023_n5245.csv'
)) {
    $Path = Join-Path $Root $Relative
    Add-Check "predictor_data_$Relative" (Test-Path -LiteralPath $Path) $Path
}
Add-Check 'exact_historical_egfr_vegfr2_rows' $AllowRecoveredEgfrVegfr2 `
    'Exact historical rows are unavailable; explicit recovered-data opt-in is required.' `
    ($(if ($AllowRecoveredEgfrVegfr2) { 'warning' } else { 'error' }))

$VinaConfigured = if ($env:VINA_EXECUTABLE) { [string]$env:VINA_EXECUTABLE } else { [string]$Protocol.docking.vina_executable_default }
$VinaPath = Resolve-ProjectPath $VinaConfigured
$VinaOnPath = Get-Command 'vina' -ErrorAction SilentlyContinue
if ($SkipExecutableChecks) {
    Add-Check 'external_vina_executable' $true 'skipped during dry-run; historical Vina 1.1.2 must be supplied separately' 'warning'
} else {
    $VinaExists = (Test-Path -LiteralPath $VinaPath) -or ($null -ne $VinaOnPath)
    $VinaDetail = if (Test-Path -LiteralPath $VinaPath) { $VinaPath } elseif ($VinaOnPath) { $VinaOnPath.Source } else { "missing; set VINA_EXECUTABLE or place binary at $VinaPath" }
    Add-Check 'external_vina_executable' $VinaExists $VinaDetail
}

if (-not $SkipExecutableChecks) {
    foreach ($Property in $Protocol.executables.PSObject.Properties) {
        if ($Property.Name -in @('conda', 'powershell')) { continue }
        $Value = [string]$Property.Value
        $Path = Resolve-ProjectPath $Value
        $Exists = (Test-Path -LiteralPath $Path) -or ($null -ne (Get-Command $Value -ErrorAction SilentlyContinue))
        Add-Check "executable_$($Property.Name)" $Exists $Path
    }
}
$Errors = @($Checks | Where-Object { -not $_.pass -and $_.severity -eq 'error' })
$Warnings = @($Checks | Where-Object { -not $_.pass -and $_.severity -eq 'warning' })
$Report = [ordered]@{
    generated_at = (Get-Date).ToString('o')
    project_root = $Root
    passed = ($Errors.Count -eq 0)
    errors = $Errors.Count
    warnings = $Warnings.Count
    checks = $Checks
}
$ReportPath = Join-Path $Root 'results\reproduction\preflight.json'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ReportPath) | Out-Null
$Report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ReportPath -Encoding UTF8
$Checks | Format-Table -AutoSize
Write-Output "PREFLIGHT passed=$($Report.passed) errors=$($Errors.Count) warnings=$($Warnings.Count) report=$ReportPath"
if ($Strict -and $Errors.Count -gt 0) { throw "Preflight found $($Errors.Count) blocking error(s)." }
