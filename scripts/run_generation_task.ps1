param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('own_method', 'polygon_shared_augmented_vae', 'reinvent4', 'drugex_v2', 'mo_lso', 'graphpareto_nsga2')]
    [string]$Method,
    [Parameter(Mandatory = $true)]
    [ValidateSet('EGFR_VEGFR2', 'PARP1_BRD4')]
    [string]$TargetPair,
    [int]$Seed,
    [int]$Budget = 10240,
    [string]$Config = '',
    [switch]$UseRecoveredEgfrVegfr2
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Config) { $Config = Join-Path $Root 'config\reproduction_pipeline.json' }
$Protocol = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
function Resolve-ProjectPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return $Value }
    return [IO.Path]::GetFullPath((Join-Path $Root $Value))
}
$PredictorRoot = if ($TargetPair -eq 'EGFR_VEGFR2' -and -not $UseRecoveredEgfrVegfr2) {
    Resolve-ProjectPath ([string]$Protocol.predictors.historical_first_pair_model_dir)
} else {
    Resolve-ProjectPath ([string]$Protocol.predictors.output_dir)
}
Write-Output "ORACLE_SOURCE pair=$TargetPair root=$PredictorRoot recovered_first_pair=$UseRecoveredEgfrVegfr2"
if ($TargetPair -eq 'EGFR_VEGFR2') {
    if ($Method -eq 'own_method') {
        $Output = Join-Path $Root "results\own_method_v4\common_seeds_42_51_10240\v4_b_raw_mean_seed$Seed"
    } else {
        $Folder = if ($Method -eq 'polygon_shared_augmented_vae') { 'polygon_original' } else { $Method }
        $Output = Join-Path $Root "results\baselines\$Folder\formal_${Budget}_seed$Seed"
    }
} else {
    $Base = Join-Path $Root 'results\target_pairs\parp1_brd4_egfr_vegfr2_aligned_20260827'
    if ($Method -eq 'own_method') {
        $Output = Join-Path $Base "own_method\formal_${Budget}_seed$Seed"
    } else {
        $Folder = if ($Method -eq 'polygon_shared_augmented_vae') { 'polygon_original' } else { $Method }
        $Output = Join-Path $Base "baselines\$Folder\formal_${Budget}_seed$Seed"
    }
}

if ($Method -eq 'own_method') {
    $Python = Resolve-ProjectPath ([string]$Protocol.executables.clover_python)
    & (Join-Path $PSScriptRoot 'run_own_method_formal.ps1') -TargetPair $TargetPair -Seed $Seed `
        -Budget $Budget -Python $Python -PredictorRoot $PredictorRoot -OutputDirectory $Output
    exit $LASTEXITCODE
}
$MethodArgument = if ($Method -eq 'polygon_shared_augmented_vae') { 'polygon_original' } else { $Method }
$PythonKey = switch ($Method) {
    'polygon_shared_augmented_vae' { 'polygon_python' }
    'reinvent4' { 'reinvent_python' }
    'drugex_v2' { 'drugex_python' }
    'mo_lso' { 'mo_lso_python' }
    'graphpareto_nsga2' { 'graphpareto_python' }
}
$Python = Resolve-ProjectPath ([string]$Protocol.executables.$PythonKey)
$Arguments = @{
    Method = $MethodArgument
    TargetPair = $TargetPair
    Budget = $Budget
    Seed = $Seed
    Python = $Python
    OutputDirectory = $Output
    PredictorRoot = $PredictorRoot
    OracleThreads = 4
    Evaluate = $true
}
if ($Method -eq 'mo_lso') {
    $Arguments.MoLsoGpPython = Resolve-ProjectPath ([string]$Protocol.executables.mo_lso_gp_python)
}
& (Join-Path $Root 'run_baseline_generation.ps1') @Arguments
exit $LASTEXITCODE
