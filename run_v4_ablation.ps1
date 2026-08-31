param(
    [int]$MaxConcurrent = 4,
    [int]$ThreadsPerProcess = 1,
    [int]$LaunchStaggerSeconds = 2,
    [string]$ConfigPath = "config\v4_formal_ablation_10240.json",
    [string]$ResultFolder = "results\own_method_v4\formal_ablation_10240",
    [int]$OracleBudgetOverride = 0,
    [int[]]$SeedOverride = @(),
    [int]$MaxAttempts = 2,
    [string]$Python = ''
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Python) {
    $Python = Join-Path (Split-Path -Parent $ProjectRoot) "code\.conda-envs\drug-pareto-rl\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }
}
$Runner = Join-Path $ProjectRoot "method\ablation\run_wc_two_targets_multiexplore.py"
$Protocol = Join-Path $ProjectRoot "config\formal_experiments.json"
$MatrixPath = if ([System.IO.Path]::IsPathRooted($ConfigPath)) { $ConfigPath } else { Join-Path $ProjectRoot $ConfigPath }
$Matrix = Get-Content -LiteralPath $MatrixPath -Raw | ConvertFrom-Json
$ResultRoot = if ([System.IO.Path]::IsPathRooted($ResultFolder)) { $ResultFolder } else { Join-Path $ProjectRoot $ResultFolder }
$LogRoot = Join-Path $ResultRoot "launcher_logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$OracleBudget = if ($OracleBudgetOverride -gt 0) { $OracleBudgetOverride } else { [int]$Matrix.oracle_budget }
$Seeds = if ($SeedOverride.Count -gt 0) { @($SeedOverride) } else { @($Matrix.seeds | ForEach-Object { [int]$_ }) }
$env:OMP_NUM_THREADS = [string]$ThreadsPerProcess
$env:MKL_NUM_THREADS = [string]$ThreadsPerProcess
$env:OPENBLAS_NUM_THREADS = [string]$ThreadsPerProcess
$env:NUMEXPR_NUM_THREADS = [string]$ThreadsPerProcess

function Merge-Variant([object]$Base, [object]$Override) {
    $merged = [ordered]@{}
    foreach ($property in $Base.PSObject.Properties) { $merged[$property.Name] = $property.Value }
    foreach ($property in $Override.PSObject.Properties) {
        if ($property.Name -notin @("label", "purpose")) { $merged[$property.Name] = $property.Value }
    }
    return $merged
}

$queue = [System.Collections.Generic.Queue[object]]::new()
foreach ($seed in $Seeds) {
    foreach ($variantProperty in $Matrix.variants.PSObject.Properties) {
        $queue.Enqueue([pscustomobject]@{
            Variant = $variantProperty.Name
            Config = Merge-Variant $Matrix.base_variant $variantProperty.Value
            Seed = [int]$seed
            Attempt = 1
        })
    }
}

$launchPlan = [ordered]@{
    experiment_id = $Matrix.experiment_id
    config = $MatrixPath
    oracle_budget = $OracleBudget
    seeds = $Seeds
    variants = @($Matrix.variants.PSObject.Properties.Name)
    requested_max_concurrent = $MaxConcurrent
    threads_per_process = $ThreadsPerProcess
    started_at = (Get-Date).ToString("o")
}
$launchPlan | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $ResultRoot "launch_plan.json") -Encoding UTF8

$running = @()
$failed = @()
$completed = @()
$CurrentLimit = $MaxConcurrent
while ($queue.Count -gt 0 -or $running.Count -gt 0) {
    while ($queue.Count -gt 0 -and $running.Count -lt $CurrentLimit) {
        $job = $queue.Dequeue()
        $outDir = Join-Path $ResultRoot ("{0}_seed{1}" -f $job.Variant, $job.Seed)
        $summary = Join-Path $outDir "summary.csv"
        if (Test-Path -LiteralPath $summary) {
            $row = Import-Csv -LiteralPath $summary | Select-Object -First 1
            if ([int][double]$row.oracle_budget -eq $OracleBudget) {
                Write-Output ("SKIP complete {0} seed {1}" -f $job.Variant, $job.Seed)
                $completed += $job
                continue
            }
        }
        New-Item -ItemType Directory -Force -Path $outDir | Out-Null
        $suffix = if ($job.Attempt -gt 1) { ".attempt$($job.Attempt)" } else { "" }
        $stdout = Join-Path $LogRoot ("{0}_seed{1}{2}.out.log" -f $job.Variant, $job.Seed, $suffix)
        $stderr = Join-Path $LogRoot ("{0}_seed{1}{2}.err.log" -f $job.Variant, $job.Seed, $suffix)
        $c = $Matrix.common
        $v = $job.Config
        $arguments = @(
            $Runner,
            "--oracle-budget", [string]$OracleBudget,
            "--batch", [string]$Matrix.batch_size,
            "--seed", [string]$job.Seed,
            "--device", "cuda",
            "--trajectory-length", [string]$c.trajectory_length,
            "--controller-variant", [string]$c.controller_variant,
            "--protocol-config", $Protocol,
            "--archive-seed-fraction", [string]$v.archive_seed_fraction,
            "--archive-seed-noise", [string]$c.archive_seed_noise,
            "--archive-seed-noise-end", [string]$c.archive_seed_noise_end,
            "--archive-seed-start", [string]$c.archive_seed_start,
            "--archive-seed-ramp-end", [string]$c.archive_seed_ramp_end,
            "--archive-seed-selection", [string]$v.archive_seed_selection,
            "--archive-hvc-weight", [string]$v.archive_hvc_weight,
            "--archive-balance-weight", [string]$v.archive_balance_weight,
            "--archive-selection-temperature", [string]$v.archive_selection_temperature,
            "--archive-uniform-mix", [string]$v.archive_uniform_mix,
            "--archive-stagnation-window", [string]$v.archive_stagnation_window,
            "--archive-stagnation-delta", [string]$v.archive_stagnation_delta,
            "--archive-stagnation-noise", [string]$v.archive_stagnation_noise,
            "--hvc-reward-weight", [string]$v.hvc_reward_weight,
            "--crowding-reward-weight", [string]$v.crowding_reward_weight,
            "--balanced-reward-weight", [string]$v.balanced_reward_weight,
            "--pareto-actor-coef", [string]$v.pareto_actor_coef,
            "--actor-mode", [string]$v.actor_mode,
            "--generator-finetune-interval", [string]$v.generator_finetune_interval,
            "--generator-finetune-epochs", [string]$v.generator_finetune_epochs,
            "--generator-finetune-top", [string]$v.generator_finetune_top,
            "--generator-finetune-batch-size", [string]$v.generator_finetune_batch_size,
            "--generator-finetune-lr", [string]$v.generator_finetune_lr,
            "--generator-elite-strategy", [string]$v.generator_elite_strategy,
            "--weight-mode", [string]$v.weight_mode,
            "--critic-mode", [string]$v.critic_mode,
            "--channel-mode", [string]$v.channel_mode,
            "--exploration-mode", [string]$v.exploration_mode,
            "--pareto-reward-start", [string]$c.pareto_reward_start,
            "--pareto-reward-ramp-end", [string]$c.pareto_reward_ramp_end,
            "--sample-preference-mode", [string]$v.sample_preference_mode,
            "--sample-preference-blend", [string]$v.sample_preference_blend,
            "--sample-preference-start", [string]$v.sample_preference_start,
            "--sample-preference-ramp-end", [string]$v.sample_preference_ramp_end,
            "--log-interval", "20",
            "--checkpoint-interval", "40",
            "--output", $outDir
        )
        $process = Start-Process -FilePath $Python -ArgumentList $arguments -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
        $running += [pscustomobject]@{Process=$process;Job=$job;Summary=$summary;Stdout=$stdout;Stderr=$stderr}
        Write-Output ("START pid={0} {1} seed {2} attempt={3} active={4}/{5}" -f $process.Id, $job.Variant, $job.Seed, $job.Attempt, $running.Count, $CurrentLimit)
        if ($LaunchStaggerSeconds -gt 0) { Start-Sleep -Seconds $LaunchStaggerSeconds }
    }

    Start-Sleep -Seconds 2
    $stillRunning = @()
    foreach ($entry in $running) {
        if (-not $entry.Process.HasExited) { $stillRunning += $entry; continue }
        if (Test-Path -LiteralPath $entry.Summary) {
            $row = Import-Csv -LiteralPath $entry.Summary | Select-Object -First 1
            if ([int][double]$row.oracle_budget -eq $OracleBudget) {
                Write-Output ("DONE {0} seed {1} HV={2} validity={3}" -f $entry.Job.Variant, $entry.Job.Seed, $row.hv_final, $row.valid_rate)
                $completed += $entry.Job
                continue
            }
        }
        $errorText = if (Test-Path -LiteralPath $entry.Stderr) { Get-Content -LiteralPath $entry.Stderr -Raw } else { "" }
        $isOom = $errorText -match "CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|illegal memory access|cudaErrorIllegalAddress"
        if ($isOom -and $CurrentLimit -gt 4) {
            $CurrentLimit = [Math]::Max(4, $CurrentLimit - 2)
            Write-Output ("RESOURCE_BACKOFF new_concurrency={0}" -f $CurrentLimit)
        }
        if ($entry.Job.Attempt -lt $MaxAttempts) {
            $entry.Job.Attempt += 1
            $queue.Enqueue($entry.Job)
            Write-Output ("REQUEUE {0} seed {1} attempt={2}" -f $entry.Job.Variant, $entry.Job.Seed, $entry.Job.Attempt)
        } else {
            Write-Output ("FAIL {0} seed {1}; see {2}" -f $entry.Job.Variant, $entry.Job.Seed, $entry.Stderr)
            $failed += $entry.Job
        }
    }
    $running = $stillRunning
}

$launcherSummary = [ordered]@{
    experiment_id = $Matrix.experiment_id
    oracle_budget = $OracleBudget
    completed = $completed.Count
    failed = $failed.Count
    final_concurrency = $CurrentLimit
    finished_at = (Get-Date).ToString("o")
}
$launcherSummary | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ResultRoot "launcher_summary.json") -Encoding UTF8
Write-Output ("ABLATION_FINISHED completed={0} failed={1} final_concurrency={2}" -f $completed.Count, $failed.Count, $CurrentLimit)
if ($failed.Count -gt 0) { exit 1 }
