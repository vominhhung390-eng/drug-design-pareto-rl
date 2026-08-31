param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(42, 51)]
    [int]$Seed,
    [int]$Budget = 10240,
    [int]$OracleThreads = 1
)

$ErrorActionPreference = 'Stop'
$adapter = Split-Path -Parent $MyInvocation.MyCommand.Path
$project = Split-Path -Parent (Split-Path -Parent $adapter)
$python = if ($env:GRAPHPARETO_PYTHON) { $env:GRAPHPARETO_PYTHON } else { 'python' }
$runner = Join-Path $adapter 'adapter_optimize_dual_oracle.py'
$evaluator = Join-Path $project 'evaluation\evaluate_anytime.py'
$evalPython = if ($env:CLOVER_PYTHON) { $env:CLOVER_PYTHON } else { 'python' }
$output = Join-Path $project "results\baselines\graphpareto_nsga2\formal_${Budget}_seed$Seed"
$runLog = Join-Path $output 'run.log'
$metadataPath = Join-Path $output 'metadata.json'
$workerLog = Join-Path $project 'logs\baselines\graphpareto_nsga2\workers.log'

New-Item -ItemType Directory -Force -Path $output | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $workerLog) | Out-Null
$env:PYTHONHASHSEED = "$Seed"
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

function Write-WorkerLog([string]$Message) {
    "$Message $(Get-Date -Format o)" | Out-File $workerLog -Append -Encoding utf8
}

function Test-OptimizationComplete {
    if (-not (Test-Path -LiteralPath $metadataPath)) { return $false }
    try {
        $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
        return [bool]$metadata.complete -and ([int]$metadata.used -eq $Budget)
    }
    catch { return $false }
}

function Test-BaseAnytimeComplete {
    foreach ($checkpoint in @(1024, 2048, 5120, 10240)) {
        $summary = Join-Path $output "anytime\budget_$checkpoint\evaluation_summary.json"
        if (-not (Test-Path -LiteralPath $summary)) { return $false }
        try {
            $payload = Get-Content -LiteralPath $summary -Raw | ConvertFrom-Json
            if ([int]$payload.generated_rows -ne $checkpoint) { return $false }
        }
        catch { return $false }
    }
    return $true
}

function Test-QualityAnytimeComplete {
    foreach ($checkpoint in @(1024, 2048, 5120, 10240)) {
        $quality = Join-Path $output "anytime\budget_$checkpoint\quality_constrained\quality_annotated_molecules.csv"
        $summary = Join-Path $output "anytime\budget_$checkpoint\quality_constrained\quality_constrained_summary.json"
        if (-not (Test-Path -LiteralPath $quality) -or -not (Test-Path -LiteralPath $summary)) {
            return $false
        }
    }
    return $true
}

Write-WorkerLog "SEED_START seed=$Seed budget=$Budget"
if (-not (Test-OptimizationComplete)) {
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $python $runner --output-dir $output --budget "$Budget" --seed "$Seed" --oracle-threads "$OracleThreads" --resume *>> $runLog
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($exitCode -ne 0 -or -not (Test-OptimizationComplete)) {
        Write-WorkerLog "SEED_FAILED seed=$Seed exit=$exitCode"
        throw "GraphPareto seed $Seed failed or did not reach the exact budget"
    }
}
Write-WorkerLog "OPTIMIZATION_COMPLETE seed=$Seed"

if (-not (Test-BaseAnytimeComplete)) {
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $evalPython $evaluator (Join-Path $output 'generated.csv') (Join-Path $output 'anytime') *>> $runLog
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $savedPreference
    if ($exitCode -ne 0 -or -not (Test-BaseAnytimeComplete)) {
        Write-WorkerLog "EVALUATION_FAILED seed=$Seed exit=$exitCode"
        throw "GraphPareto evaluation for seed $Seed failed"
    }
}

if (-not (Test-QualityAnytimeComplete)) {
    foreach ($checkpoint in @(1024, 2048, 5120, 10240)) {
        $checkpointRoot = Join-Path $output "anytime\budget_$checkpoint"
        $qualityRoot = Join-Path $checkpointRoot 'quality_constrained'
        $qualitySummary = Join-Path $qualityRoot 'quality_constrained_summary.json'
        if (Test-Path -LiteralPath $qualitySummary) { continue }
        $savedPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $evalPython (Join-Path $project 'evaluation\quality_constrained_metrics.py') `
            (Join-Path $checkpointRoot 'standardized_molecules.csv') $qualityRoot *>> $runLog
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $savedPreference
        if ($exitCode -ne 0) {
            Write-WorkerLog "QUALITY_EVALUATION_FAILED seed=$Seed checkpoint=$checkpoint exit=$exitCode"
            throw "GraphPareto quality evaluation for seed $Seed checkpoint $checkpoint failed"
        }
    }
}
if (-not (Test-QualityAnytimeComplete)) {
    throw "GraphPareto quality evaluation for seed $Seed is incomplete"
}
Write-WorkerLog "SEED_COMPLETE seed=$Seed"
