param([int]$RefreshSeconds = 10)

$project = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$resultRoot = Join-Path $project 'results\baselines\graphpareto_nsga2'
$budget = 10240
$seeds = 42..51

function Get-Bar([double]$Percent, [int]$Width = 20) {
    $filled = [math]::Min($Width, [math]::Max(0, [math]::Floor($Percent * $Width / 100)))
    return ('#' * $filled) + ('-' * ($Width - $filled))
}

while ($true) {
    $processes = @(Get-CimInstance Win32_Process)
    $rows = @()
    foreach ($seed in $seeds) {
        $run = Join-Path $resultRoot "formal_${budget}_seed$seed"
        $metadataPath = Join-Path $run 'metadata.json'
        $used = 0
        $complete = $false
        if (Test-Path -LiteralPath $metadataPath) {
            try {
                $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
                $used = [int]$metadata.used
                $complete = [bool]$metadata.complete -and $used -eq $budget
            }
            catch { }
        }
        $evaluation = (Test-Path -LiteralPath (Join-Path $run 'anytime\budget_10240\evaluation_summary.json')) -and
            (Test-Path -LiteralPath (Join-Path $run 'anytime\budget_10240\quality_constrained\quality_constrained_summary.json'))
        $running = @($processes | Where-Object {
            $_.CommandLine -match 'run_formal_seed\.ps1' -and
            $_.CommandLine -match "-Seed\s+$seed(?:\s|$)"
        }).Count -gt 0
        $status = if ($evaluation) { 'done' } elseif ($complete -and $running) { 'evaluating' } elseif ($complete) { 'generated' } elseif ($running) { 'running' } else { 'waiting' }
        $percent = 100.0 * $used / $budget
        $rows += [pscustomobject]@{
            Seed = $seed
            Bar = '[' + (Get-Bar $percent) + ']'
            Progress = ('{0,5:N1}%' -f $percent)
            Used = "$used/$budget"
            Status = $status
        }
    }
    $totalUsed = ($rows | ForEach-Object { [int](($_.Used -split '/')[0]) } | Measure-Object -Sum).Sum
    $done = @($rows | Where-Object Status -eq 'done').Count
    $totalPercent = 100.0 * $totalUsed / ($budget * $seeds.Count)
    Clear-Host
    Write-Host ("GraphPareto total {0:N1}%  {1}/{2}  done {3}/10" -f $totalPercent, $totalUsed, ($budget * 10), $done) -ForegroundColor Cyan
    Write-Host ''
    $rows | Format-Table -AutoSize
    Write-Host ("Refresh every {0}s. Closing this window does not stop experiments." -f $RefreshSeconds) -ForegroundColor DarkGray
    Start-Sleep -Seconds $RefreshSeconds
}
