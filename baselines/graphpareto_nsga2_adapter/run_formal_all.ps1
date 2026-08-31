param(
    [int[]]$Seeds = (42..51),
    [int]$OracleThreads = 1
)

$ErrorActionPreference = 'Stop'
$adapter = Split-Path -Parent $MyInvocation.MyCommand.Path
$project = Split-Path -Parent (Split-Path -Parent $adapter)
$worker = Join-Path $adapter 'run_formal_seed.ps1'
$logRoot = Join-Path $project 'logs\baselines\graphpareto_nsga2'
$launcherLog = Join-Path $logRoot 'launcher.log'
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$launched = @()
foreach ($seed in $Seeds) {
    $existing = @(Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match 'run_formal_seed\.ps1' -and
        $_.CommandLine -match "-Seed\s+$seed(?:\s|$)"
    })
    if ($existing.Count -gt 0) {
        $launched += [pscustomobject]@{ Seed = $seed; Pid = $existing[0].ProcessId; Status = 'already-running' }
        continue
    }
    $stdout = Join-Path $logRoot "seed_${seed}.stdout.log"
    $stderr = Join-Path $logRoot "seed_${seed}.stderr.log"
    $process = Start-Process -FilePath 'powershell.exe' `
        -ArgumentList @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$worker`"",
            '-Seed', "$seed", '-Budget', '10240', '-OracleThreads', "$OracleThreads"
        ) `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru
    "LAUNCHED seed=$seed pid=$($process.Id) $(Get-Date -Format o)" |
        Out-File $launcherLog -Append -Encoding utf8
    $launched += [pscustomobject]@{ Seed = $seed; Pid = $process.Id; Status = 'launched' }
}
$launched | Format-Table -AutoSize
