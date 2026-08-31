param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('EGFR_VEGFR2', 'PARP1_BRD4')]
    [string]$TargetPair,
    [int]$Seed = 42,
    [int]$Budget = 10240,
    [string]$Python = 'python',
    [string]$PredictorRoot = '',
    [string]$VaeModel = '',
    [string]$OutputDirectory = '',
    [int]$Threads = 4,
    [switch]$CpuOnly
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not $PredictorRoot) {
    $PredictorRoot = if ($TargetPair -eq 'EGFR_VEGFR2') {
        Join-Path $Root 'models\oracles'
    } else {
        Join-Path $Root 'models\reproduced_oracles'
    }
}
if (-not $VaeModel) { $VaeModel = Join-Path $Root 'models\polygon_vae_best_valid_novel_stable_020.pt' }
if ($TargetPair -eq 'EGFR_VEGFR2') {
    $Target1 = 'EGFR'; $Target2 = 'VEGFR2'
    $DefaultOutput = Join-Path $Root "results\own_method_v4\common_seeds_42_51_10240\v4_b_raw_mean_seed$Seed"
} else {
    $Target1 = 'PARP1'; $Target2 = 'BRD4'
    $DefaultOutput = Join-Path $Root "results\target_pairs\parp1_brd4_egfr_vegfr2_aligned_20260827\own_method\formal_${Budget}_seed$Seed"
}
if (-not $OutputDirectory) { $OutputDirectory = $DefaultOutput }
$Model1 = Join-Path $PredictorRoot "target_${Target1}_model.pkl"
$Model2 = Join-Path $PredictorRoot "target_${Target2}_model.pkl"
foreach ($Path in @($Model1, $Model2, $VaeModel)) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing required model: $Path" }
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$Generated = Join-Path $OutputDirectory 'all_generated_molecules.csv'
$Rows = if (Test-Path -LiteralPath $Generated) {
    [math]::Max(0, (Get-Content -LiteralPath $Generated | Measure-Object -Line).Lines - 1)
} else { 0 }
$env:PYTHONHASHSEED = [string]$Seed
$env:OMP_NUM_THREADS = [string]$Threads
$env:MKL_NUM_THREADS = [string]$Threads
$env:OPENBLAS_NUM_THREADS = [string]$Threads
$env:NUMEXPR_NUM_THREADS = [string]$Threads

if ($Rows -ne $Budget) {
    $Runner = Join-Path $Root 'method\ablation\run_wc_two_targets_multiexplore.py'
    $Epochs = [int]($Budget / 64)
    $Device = if ($CpuOnly) { 'cpu' } else { 'cuda' }
    $Arguments = @(
        $Runner, '--model', $VaeModel, '--output', $OutputDirectory,
        '--epochs', [string]$Epochs, '--oracle-budget', [string]$Budget, '--batch', '64',
        '--device', $Device, '--trajectory-length', '1', '--trajectory-step-normalization', 'sqrt',
        '--archive-seed-fraction', '0.0', '--archive-seed-noise', '0.10',
        '--archive-seed-noise-end', '0.05', '--archive-seed-start', '0.30',
        '--archive-seed-ramp-end', '0.70', '--archive-seed-selection', 'uniform',
        '--archive-hvc-weight', '0.7', '--archive-balance-weight', '0.3',
        '--archive-selection-temperature', '0.25', '--archive-uniform-mix', '0.10',
        '--archive-stagnation-window', '0', '--archive-stagnation-delta', '0.002',
        '--archive-stagnation-noise', '0.0', '--actor-mode', 'train',
        '--generator-finetune-interval', '16', '--generator-finetune-epochs', '2',
        '--generator-finetune-top', '512', '--generator-finetune-batch-size', '512',
        '--generator-finetune-lr', '0.0003', '--generator-elite-strategy', 'raw_mean',
        '--sample-preference-mode', 'shared', '--sample-preference-blend', '0.0',
        '--sample-preference-start', '0.30', '--sample-preference-ramp-end', '0.70',
        '--pareto-reward-start', '0.30', '--pareto-reward-ramp-end', '0.70',
        '--hvc-reward-weight', '0.0', '--crowding-reward-weight', '0.0',
        '--balanced-reward-weight', '0.0', '--pareto-actor-coef', '0.0',
        '--weight-mode', 'dynamic', '--critic-mode', 'multi',
        '--controller-variant', 'ours_full_corrected', '--channel-mode', 'adaptive',
        '--exploration-mode', 'multiscale', '--protocol-config', (Join-Path $Root 'config\formal_experiments.json'),
        '--egfr-model', $Model1, '--vegfr2-model', $Model2, '--oracle-system', 'original_rf',
        '--log-interval', '10', '--checkpoint-interval', '20', '--seed', [string]$Seed
    )
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "CLOVER-Mol generation failed with exit code $LASTEXITCODE" }
}

& $Python (Join-Path $Root 'evaluation\evaluate_anytime.py') $Generated (Join-Path $OutputDirectory 'anytime')
if ($LASTEXITCODE -ne 0) { throw "CLOVER-Mol evaluation failed with exit code $LASTEXITCODE" }
& $Python (Join-Path $Root 'evaluation\evaluate_experiment.py') $Generated (Join-Path $OutputDirectory 'evaluation')
if ($LASTEXITCODE -ne 0) { throw "CLOVER-Mol quality evaluation failed with exit code $LASTEXITCODE" }
[ordered]@{
    method = 'CLOVER-Mol V4-B raw-mean'
    target_pair = @($Target1, $Target2)
    compatibility_columns = [ordered]@{ egfr = $Target1; vegfr2 = $Target2 }
    target_1_model = $Model1
    target_1_sha256 = (Get-FileHash -LiteralPath $Model1 -Algorithm SHA256).Hash.ToLowerInvariant()
    target_2_model = $Model2
    target_2_sha256 = (Get-FileHash -LiteralPath $Model2 -Algorithm SHA256).Hash.ToLowerInvariant()
    generator = $VaeModel
    generator_sha256 = (Get-FileHash -LiteralPath $VaeModel -Algorithm SHA256).Hash.ToLowerInvariant()
    budget = $Budget
    seed = $Seed
    complete = $true
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'experiment_metadata.json') -Encoding UTF8
Write-Output "COMPLETE method=CLOVER-Mol pair=$TargetPair seed=$Seed output=$OutputDirectory"
