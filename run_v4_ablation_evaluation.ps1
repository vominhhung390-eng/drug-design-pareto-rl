param(
    [int]$MaxConcurrent = 8,
    [string]$ResultFolder = "results\own_method_v4\formal_ablation_10240",
    [string]$Python = ''
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Python) {
    $Python = Join-Path (Split-Path -Parent $ProjectRoot) "code\.conda-envs\drug-pareto-rl\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }
}
$Evaluator = Join-Path $ProjectRoot "evaluation\evaluate_ablation_run.py"
$ResultRoot = if ([System.IO.Path]::IsPathRooted($ResultFolder)) { $ResultFolder } else { Join-Path $ProjectRoot $ResultFolder }
$LogRoot = Join-Path $ResultRoot "evaluation_logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"

$queue = [System.Collections.Generic.Queue[System.IO.DirectoryInfo]]::new()
Get-ChildItem -LiteralPath $ResultRoot -Directory -Filter "*_seed*" | Sort-Object Name | ForEach-Object { $queue.Enqueue($_) }
$running = @()
$completed = @()
$failed = @()
while ($queue.Count -gt 0 -or $running.Count -gt 0) {
    while ($queue.Count -gt 0 -and $running.Count -lt $MaxConcurrent) {
        $run = $queue.Dequeue()
        $source = Join-Path $run.FullName "all_generated_molecules.csv"
        $quality = Join-Path $run.FullName "evaluation\quality_constrained\quality_constrained_summary.json"
        $anytime = Join-Path $run.FullName "anytime\budget_10240\evaluation_summary.json"
        if ((Test-Path -LiteralPath $quality) -and (Test-Path -LiteralPath $anytime)) {
            $completed += $run
            continue
        }
        if (-not (Test-Path -LiteralPath $source)) {
            Write-Output ("MISSING input {0}" -f $source)
            $failed += $run
            continue
        }
        $stdout = Join-Path $LogRoot ($run.Name + ".out.log")
        $stderr = Join-Path $LogRoot ($run.Name + ".err.log")
        $process = Start-Process -FilePath $Python -ArgumentList @($Evaluator, $run.FullName) -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
        $running += [pscustomobject]@{Process=$process;Run=$run;Quality=$quality;Anytime=$anytime;Stderr=$stderr}
        Write-Output ("EVAL_START pid={0} {1}" -f $process.Id, $run.Name)
    }
    Start-Sleep -Seconds 1
    $stillRunning = @()
    foreach ($entry in $running) {
        if (-not $entry.Process.HasExited) { $stillRunning += $entry; continue }
        if ((Test-Path -LiteralPath $entry.Quality) -and (Test-Path -LiteralPath $entry.Anytime)) {
            $completed += $entry.Run
            Write-Output ("EVAL_DONE {0}" -f $entry.Run.Name)
        } else {
            $failed += $entry.Run
            Write-Output ("EVAL_FAIL {0}; see {1}" -f $entry.Run.Name, $entry.Stderr)
        }
    }
    $running = $stillRunning
}
@{completed=$completed.Count;failed=$failed.Count;finished_at=(Get-Date).ToString("o")} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ResultRoot "evaluation_launcher_summary.json") -Encoding UTF8
Write-Output ("EVALUATION_FINISHED completed={0} failed={1}" -f $completed.Count, $failed.Count)
if ($failed.Count -gt 0) { exit 1 }
