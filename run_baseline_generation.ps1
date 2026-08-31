param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('polygon_original', 'reinvent4', 'drugex_v2', 'mo_lso', 'graphpareto_nsga2')]
    [string]$Method,
    [ValidateSet('EGFR_VEGFR2', 'PARP1_BRD4')]
    [string]$TargetPair = 'EGFR_VEGFR2',
    [int]$Budget = 10240,
    [int]$Seed = 42,
    [string]$Python = 'python',
    [string]$OutputDirectory = '',
    [int]$OracleThreads = 4,
    [string]$PredictorRoot = '',
    [string]$MoLsoGpPython = '',
    [switch]$CpuOnly,
    [switch]$Evaluate
)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $PredictorRoot) {
    $PredictorRoot = if ($TargetPair -eq 'EGFR_VEGFR2') {
        Join-Path $project 'models\oracles'
    } else {
        Join-Path $project 'models\reproduced_oracles'
    }
}
if ($TargetPair -eq 'EGFR_VEGFR2') {
    $model1 = Join-Path $PredictorRoot 'target_EGFR_model.pkl'
    $model2 = Join-Path $PredictorRoot 'target_VEGFR2_model.pkl'
    $target1 = 'EGFR'
    $target2 = 'VEGFR2'
}
else {
    $model1 = Join-Path $PredictorRoot 'target_PARP1_model.pkl'
    $model2 = Join-Path $PredictorRoot 'target_BRD4_model.pkl'
    $target1 = 'PARP1'
    $target2 = 'BRD4'
}

foreach ($path in @($model1, $model2)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing predictor: $path" }
}

$env:DUAL_TARGET_MODEL_1 = $model1
$env:DUAL_TARGET_MODEL_2 = $model2
$env:DUAL_TARGET_NAME_1 = $target1
$env:DUAL_TARGET_NAME_2 = $target2
$env:CUDA_VISIBLE_DEVICES = if ($CpuOnly) { '-1' } else { '0' }
$env:OMP_NUM_THREADS = [string]$OracleThreads
$env:MKL_NUM_THREADS = [string]$OracleThreads
$env:OPENBLAS_NUM_THREADS = [string]$OracleThreads
$env:PYTHONHASHSEED = [string]$Seed

if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $project "results\reproduced\$TargetPair\$Method\formal_${Budget}_seed$Seed"
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$workingDirectory = Join-Path $project "baselines\${Method}_adapter"
switch ($Method) {
    'polygon_original' {
        $workingDirectory = Join-Path $project 'baselines\polygon_adapter'
        $arguments = @('adapter_optimize_dual_oracle.py', '--checkpoint', "$project\models\polygon_vae_best_valid_novel_stable_020.pt", '--output-dir', $OutputDirectory, '--budget', "$Budget", '--seed', "$Seed")
        if ($Budget -eq 10240) { $arguments += @('--batch-size', '1024', '--keep-top', '512', '--finetune-epochs', '2', '--finetune-batch-size', '256') }
    }
    'reinvent4' {
        $workingDirectory = Join-Path $project 'baselines\reinvent4_adapter'
        $arguments = @('adapter_optimize_dual_oracle.py', '--prior', "$project\results\baselines\reinvent4\models\reinvent4_common_dataset.model", '--output-dir', $OutputDirectory, '--budget', "$Budget", '--seed', "$Seed")
    }
    'drugex_v2' {
        $workingDirectory = Join-Path $project 'baselines\drugex_v2_adapter'
        $arguments = @('adapter_optimize_dual_oracle.py', '--prior', "$project\results\baselines\drugex_v2\models\drugex_v2_common_dataset_best.pkg", '--vocabulary', "$project\results\baselines\drugex_v2\data\common_voc.txt", '--output-dir', $OutputDirectory, '--budget', "$Budget", '--seed', "$Seed")
    }
    'mo_lso' {
        $workingDirectory = Join-Path $project 'baselines\mo_lso_adapter'
        $arguments = @('adapter_optimize_dual_oracle.py', '--model', "$project\results\baselines\mo_lso\models\best.pt", '--output-dir', $OutputDirectory, '--budget', "$Budget", '--seed', "$Seed")
        if ($MoLsoGpPython) { $arguments += @('--gp-python', $MoLsoGpPython) }
        if ($CpuOnly) { $arguments += @('--device', 'cpu') }
    }
    'graphpareto_nsga2' {
        $workingDirectory = Join-Path $project 'baselines\graphpareto_nsga2_adapter'
        $arguments = @('adapter_optimize_dual_oracle.py', '--output-dir', $OutputDirectory, '--budget', "$Budget", '--seed', "$Seed", '--oracle-threads', "$OracleThreads", '--resume')
    }
}

Push-Location -LiteralPath $workingDirectory
try {
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "$Method generation failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }

if ($Evaluate) {
    & $Python "$project\evaluation\evaluate_anytime.py" "$OutputDirectory\generated.csv" "$OutputDirectory\anytime"
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed with exit code $LASTEXITCODE" }
}

Write-Output "COMPLETE method=$Method pair=$TargetPair budget=$Budget seed=$Seed output=$OutputDirectory"
