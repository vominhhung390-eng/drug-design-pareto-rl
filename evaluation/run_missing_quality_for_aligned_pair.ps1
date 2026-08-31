param(
    [string]$ExperimentRoot = '',
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
if (-not $ExperimentRoot) {
    $ExperimentRoot = Join-Path $project 'results\target_pairs\parp1_brd4_egfr_vegfr2_aligned_20260827'
}
$qualityScript = Join-Path $PSScriptRoot 'quality_constrained_metrics.py'
$logDirectory = Join-Path $ExperimentRoot 'logs\quality_constrained'
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$log = Join-Path $logDirectory 'run.log'

$runs = @()
$runs += Get-ChildItem -LiteralPath (Join-Path $ExperimentRoot 'own_method') -Directory -Filter 'formal_10240_seed*' -ErrorAction SilentlyContinue
$runs += Get-ChildItem -LiteralPath (Join-Path $ExperimentRoot 'baselines') -Directory -Recurse -Filter 'formal_10240_seed*' -ErrorAction SilentlyContinue
$runs = @($runs | Sort-Object FullName -Unique)

$completed = 0
$skipped = 0
$missing = 0
foreach ($run in $runs) {
    $evaluation = Join-Path $run.FullName 'anytime\budget_10240'
    $input = Join-Path $evaluation 'standardized_molecules.csv'
    $output = Join-Path $evaluation 'quality_constrained'
    $summary = Join-Path $output 'quality_constrained_summary.json'
    if (Test-Path -LiteralPath $summary) {
        $skipped++
        continue
    }
    if (-not (Test-Path -LiteralPath $input)) {
        $missing++
        continue
    }
    "START $($run.FullName) $(Get-Date -Format o)" | Out-File -LiteralPath $log -Append -Encoding utf8
    & $Python $qualityScript $input $output *>> $log
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $summary)) {
        throw "Quality evaluation failed for $($run.FullName)"
    }
    $completed++
    "COMPLETE $($run.FullName) $(Get-Date -Format o)" | Out-File -LiteralPath $log -Append -Encoding utf8
}

[pscustomobject]@{
    discovered_runs = $runs.Count
    completed_now = $completed
    already_complete = $skipped
    missing_final_evaluation = $missing
    log = $log
}
