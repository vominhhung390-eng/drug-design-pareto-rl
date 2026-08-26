param(
    [int]$OracleBudget = 2048,
    [int]$Seed = 42,
    [ValidateSet(1, 3, 5)]
    [int]$TrajectoryLength = 1,
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda"
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path (Split-Path -Parent $ProjectRoot) "code\.conda-envs\drug-pareto-rl\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

& $Python (Join-Path $ProjectRoot "method\ablation\run_wc_two_targets_multiexplore.py") `
    --oracle-budget $OracleBudget `
    --seed $Seed `
    --trajectory-length $TrajectoryLength `
    --device $Device `
    --output (Join-Path $ProjectRoot "logs\ours_full_k${TrajectoryLength}_seed_$Seed")
