param(
    [ValidateSet('All', 'Setup', 'Preflight', 'Predictors', 'VAE', 'BottomModels', 'Generation', 'Ablations', 'Docking', 'Tables')]
    [string]$Stage = 'All',
    [string]$Config = '',
    [switch]$AllowRecoveredEgfrVegfr2,
    [string]$VinaExecutable = '',
    [switch]$SkipEnvironmentSetup,
    [int]$MaxParallel = 0,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Config) { $Config = Join-Path $Root 'config\reproduction_pipeline.json' }
$Protocol = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
function Resolve-ProjectPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return $Value }
    return [IO.Path]::GetFullPath((Join-Path $Root $Value))
}
function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    if ($DryRun) { Write-Output ("DRYRUN " + $Executable + ' ' + ($Arguments -join ' ')); return }
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Executable" }
}
function Includes([string]$Name) { return ($Stage -eq 'All' -or $Stage -eq $Name) }

if ($VinaExecutable) {
    $env:VINA_EXECUTABLE = Resolve-ProjectPath $VinaExecutable
}

$LogRoot = Join-Path $Root 'results\reproduction\logs'
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
if ((Includes 'Setup') -and -not $SkipEnvironmentSetup) {
    & (Join-Path $PSScriptRoot 'setup_environments.ps1') -Config $Config -DryRun:$DryRun
}
if (Includes 'Preflight') {
    $Preflight = @{
        Config = $Config
        AllowRecoveredEgfrVegfr2 = $AllowRecoveredEgfrVegfr2
        Strict = $true
        SkipExecutableChecks = $DryRun
    }
    & (Join-Path $PSScriptRoot 'preflight_reproduction.ps1') @Preflight
}

$PredictorPython = Resolve-ProjectPath ([string]$Protocol.executables.predictor_python)
if (Includes 'Predictors') {
    $Arguments = @((Join-Path $PSScriptRoot 'train_four_rf_predictors.py'), '--output', (Resolve-ProjectPath ([string]$Protocol.predictors.output_dir)))
    if ($AllowRecoveredEgfrVegfr2) { $Arguments += '--allow-recovered-egfr-vegfr2' }
    if ($AllowRecoveredEgfrVegfr2) {
        Write-Output 'PREDICTOR_MODE optional recovered EGFR/VEGFR2 plus formal PARP1/BRD4 retraining'
    } else {
        Write-Output 'PREDICTOR_MODE bundled historical EGFR/VEGFR2 plus formal PARP1/BRD4 retraining'
    }
    Invoke-Checked $PredictorPython $Arguments
}
if (Includes 'VAE') {
    & (Join-Path $PSScriptRoot 'train_shared_polygon_vae.ps1') -Python (Resolve-ProjectPath ([string]$Protocol.executables.polygon_python)) -Config $Config -DryRun:$DryRun
}
if (Includes 'BottomModels') {
    if ($DryRun) {
        Write-Output 'DRYRUN train REINVENT4, DrugEx v2 and MO-LSO bottom models from data/train_smiles_only.txt'
    } else {
        & (Join-Path $Root 'train_baseline_bottom_model.ps1') -Method reinvent4 `
            -Python (Resolve-ProjectPath ([string]$Protocol.executables.reinvent_python)) `
            -ReinventExecutable (Resolve-ProjectPath ([string]$Protocol.executables.reinvent_executable)) -Workers 8
        & (Join-Path $Root 'train_baseline_bottom_model.ps1') -Method drugex_v2 `
            -Python (Resolve-ProjectPath ([string]$Protocol.executables.drugex_python)) -Workers 8 -RebuildPreprocessedData
        & (Join-Path $Root 'train_baseline_bottom_model.ps1') -Method mo_lso `
            -Python (Resolve-ProjectPath ([string]$Protocol.executables.mo_lso_python)) -Workers 8 -RebuildPreprocessedData
    }
}

if (Includes 'Generation') {
    if ($MaxParallel -le 0) { $MaxParallel = [int]$Protocol.formal_protocol.max_parallel_generation_tasks }
    $ShellValue = [string]$Protocol.executables.powershell
    $Shell = if ([IO.Path]::IsPathRooted($ShellValue)) { $ShellValue } else { (Get-Command $ShellValue -ErrorAction Stop).Source }
    $Tasks = foreach ($Pair in @($Protocol.formal_protocol.target_pairs)) {
        foreach ($Method in @($Protocol.formal_protocol.methods)) {
            foreach ($Seed in @($Protocol.formal_protocol.seeds)) {
                [pscustomobject]@{ pair = [string]$Pair; method = [string]$Method; seed = [int]$Seed }
            }
        }
    }
    $Running = @()
    $Failures = @()
    foreach ($Task in $Tasks) {
        while (@($Running | Where-Object { -not $_.process.HasExited }).Count -ge $MaxParallel) {
            Start-Sleep -Seconds 2
            foreach ($Entry in @($Running | Where-Object { $_.process.HasExited -and -not $_.collected })) {
                $Entry.process.WaitForExit(); $Entry.collected = $true
                if ($Entry.process.ExitCode -ne 0) { $Failures += $Entry }
            }
            if ($Failures.Count) { break }
        }
        if ($Failures.Count) { break }
        $Name = "$($Task.pair)_$($Task.method)_seed$($Task.seed)"
        $OracleMode = if ($Task.pair -eq 'EGFR_VEGFR2' -and -not $AllowRecoveredEgfrVegfr2) { 'bundled-historical' } else { 'reproduced' }
        if ($DryRun) { Write-Output "DRYRUN generation $Name oracle=$OracleMode"; continue }
        $Stdout = Join-Path $LogRoot "$Name.out.log"
        $Stderr = Join-Path $LogRoot "$Name.err.log"
        $Arguments = @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'run_generation_task.ps1'),
            '-Method', $Task.method, '-TargetPair', $Task.pair, '-Seed', [string]$Task.seed,
            '-Budget', [string]$Protocol.formal_protocol.oracle_budget_per_seed, '-Config', $Config
        )
        if ($AllowRecoveredEgfrVegfr2) { $Arguments += '-UseRecoveredEgfrVegfr2' }
        $Process = Start-Process -FilePath $Shell -ArgumentList $Arguments -WorkingDirectory $Root `
            -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
        $Running += [pscustomobject]@{ name = $Name; process = $Process; stdout = $Stdout; stderr = $Stderr; collected = $false }
        Write-Output "START $Name pid=$($Process.Id)"
    }
    if (-not $DryRun) {
        foreach ($Entry in $Running) {
            if (-not $Entry.collected) { $Entry.process.WaitForExit(); $Entry.collected = $true }
            if ($Entry.process.ExitCode -ne 0) { $Failures += $Entry }
        }
        if ($Failures.Count) {
            $Names = ($Failures | ForEach-Object { $_.name }) -join ', '
            throw "Generation failures: $Names. Inspect $LogRoot"
        }
    }
}

if (Includes 'Ablations') {
    $CloverPython = Resolve-ProjectPath ([string]$Protocol.executables.clover_python)
    $AblationRoot = Join-Path $Root 'results\own_method_v4\formal_ablation_10240'
    if ($DryRun) {
        Write-Output 'DRYRUN run registered V4-B ablation matrix, evaluate every seed, and build ablation report'
    } else {
        & (Join-Path $Root 'run_v4_ablation.ps1') -MaxConcurrent ([math]::Max(1, $MaxParallel)) -ThreadsPerProcess 1 -Python $CloverPython
        & (Join-Path $Root 'run_v4_ablation_evaluation.ps1') -MaxConcurrent 8 -ResultFolder $AblationRoot -Python $CloverPython
        Invoke-Checked $CloverPython @(
            (Join-Path $Root 'evaluation\summarize_v4_ablation.py'), $AblationRoot,
            '--config', (Join-Path $Root 'config\v4_formal_ablation_10240.json')
        )
    }
}

if (Includes 'Docking') {
    $DockingPython = Resolve-ProjectPath ([string]$Protocol.executables.docking_python)
    $env:MEEKO_PREP_LIGAND = Join-Path (Split-Path -Parent $DockingPython) 'mk_prepare_ligand.exe'
    foreach ($Script in @(
        'select_seed_top10_docking_panel.py',
        'prepare_seed_top10_ligands.py',
        'run_seed_top10_vina.py',
        'summarize_seed_top10_docking.py',
        'analyze_predictor_docking_consistency.py'
    )) {
        Invoke-Checked $DockingPython @((Join-Path $Root "docking\scripts\$Script"))
    }
}
if (Includes 'Tables') {
    if ($DryRun) {
        Write-Output 'DRYRUN build both target-pair paper tables from 120 formal seed results'
    } else {
        & (Join-Path $PSScriptRoot 'build_paper_tables.ps1') -Python $PredictorPython -Output (Join-Path $Root 'results\paper_tables')
    }
}
Write-Output "REPRODUCTION_STAGE_COMPLETE stage=$Stage root=$Root"
