param(
    [int]$MaxParallel = 4,
    [int]$OracleBudget = 10240
)

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Root ".venvs\chemprop\Scripts\python.exe"
$Runner = Join-Path $Root "method\ablation\run_wc_two_targets_multiexplore.py"
$Protocol = Join-Path $Root "config\predictor_reward_comparison.json"
$Model = Join-Path $Root "models\polygon_vae_best_valid_novel_stable_020.pt"
$OutputRoot = Join-Path $Root "results\own_method_v4\predictor_reward_generation_20260802"
$LogRoot = Join-Path $OutputRoot "launcher_logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$tasks = [System.Collections.Generic.List[object]]::new()
foreach ($seed in 42..51) {
    foreach ($oracle in @("v41", "original_rf")) {
        $tasks.Add([pscustomobject]@{ Oracle = $oracle; Seed = $seed })
    }
}

$running = [System.Collections.Generic.List[object]]::new()
$completed = 0
$failed = 0

while ($tasks.Count -gt 0 -or $running.Count -gt 0) {
    while ($tasks.Count -gt 0 -and $running.Count -lt $MaxParallel) {
        $task = $tasks[0]
        $tasks.RemoveAt(0)
        $name = "$($task.Oracle)_seed$($task.Seed)"
        $output = Join-Path (Join-Path $OutputRoot $task.Oracle) "seed$($task.Seed)"
        $summary = Join-Path $output "summary.csv"
        if (Test-Path -LiteralPath $summary) {
            Write-Output "SKIP complete $name"
            $completed++
            continue
        }
        New-Item -ItemType Directory -Force -Path $output | Out-Null
        $stdout = Join-Path $LogRoot "$name.out.log"
        $stderr = Join-Path $LogRoot "$name.err.log"
        $arguments = @(
            "-X", "utf8", $Runner,
            "--model", $Model,
            "--output", $output,
            "--protocol-config", $Protocol,
            "--oracle-budget", "$OracleBudget",
            "--batch", "64",
            "--seed", "$($task.Seed)",
            "--device", "cuda",
            "--oracle-system", $task.Oracle,
            "--trajectory-length", "1",
            "--controller-variant", "ours_full_corrected",
            "--actor-mode", "train",
            "--generator-finetune-interval", "16",
            "--generator-finetune-epochs", "2",
            "--generator-finetune-top", "512",
            "--generator-finetune-batch-size", "512",
            "--generator-finetune-lr", "0.0003",
            "--generator-elite-strategy", "raw_mean",
            "--archive-seed-fraction", "0",
            "--archive-seed-noise", "0.1",
            "--archive-seed-noise-end", "0.05",
            "--archive-seed-start", "0.3",
            "--archive-seed-ramp-end", "0.7",
            "--sample-preference-mode", "shared",
            "--sample-preference-blend", "0",
            "--sample-preference-start", "0.3",
            "--sample-preference-ramp-end", "0.7",
            "--pareto-reward-start", "0.3",
            "--pareto-reward-ramp-end", "0.7",
            "--checkpoint-interval", "20"
        )
        $process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        $running.Add([pscustomobject]@{
            Name = $name
            Process = $process
            Output = $output
            Stdout = $stdout
            Stderr = $stderr
            Started = Get-Date
        })
        Write-Output "START $name pid=$($process.Id) active=$($running.Count)"
    }

    Start-Sleep -Seconds 5
    for ($index = $running.Count - 1; $index -ge 0; $index--) {
        $job = $running[$index]
        if ($job.Process.HasExited) {
            $job.Process.Refresh()
            $elapsed = [math]::Round(((Get-Date) - $job.Started).TotalMinutes, 2)
            $summaryPath = Join-Path $job.Output "summary.csv"
            if (Test-Path -LiteralPath $summaryPath) {
                Write-Output "DONE $($job.Name) minutes=$elapsed"
                $completed++
            } else {
                Write-Output "FAIL $($job.Name) exit=$($job.Process.ExitCode) minutes=$elapsed log=$($job.Stderr)"
                $failed++
            }
            $running.RemoveAt($index)
        }
    }
    Write-Output "STATUS completed=$completed failed=$failed queued=$($tasks.Count) active=$($running.Count)"
}

if ($failed -gt 0) {
    throw "$failed predictor-reward generation runs failed"
}
Write-Output "ALL COMPLETE completed=$completed"
