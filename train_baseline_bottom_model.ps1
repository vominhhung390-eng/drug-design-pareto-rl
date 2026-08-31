param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('polygon_original', 'reinvent4', 'drugex_v2', 'mo_lso', 'graphpareto_nsga2')]
    [string]$Method,
    [string]$Python = 'python',
    [string]$ReinventExecutable = 'reinvent',
    [int]$Workers = 8,
    [switch]$RebuildPreprocessedData
)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONHASHSEED = '42'
$env:OMP_NUM_THREADS = [string]$Workers
$env:MKL_NUM_THREADS = [string]$Workers

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments, [string]$WorkingDirectory)
    Push-Location -LiteralPath $WorkingDirectory
    try {
        & $Executable @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
    }
    finally { Pop-Location }
}

switch ($Method) {
    'polygon_original' {
        & (Join-Path $project 'scripts\train_shared_polygon_vae.ps1') -Python $Python
        if ($LASTEXITCODE -ne 0) { throw 'Shared augmented POLYGON VAE training failed.' }
    }
    'graphpareto_nsga2' {
        Write-Output 'GraphPareto-NSGA-II has no bottom generative-model training stage; it performs direct graph evolutionary search.'
    }
    'reinvent4' {
        $root = Join-Path $project 'baselines\reinvent4_adapter'
        Invoke-Checked $Python @('-m', 'reinvent.runmodes.create_model.create_reinvent', 'local_configs\create_common_dataset_prior.toml') $root
        Invoke-Checked $ReinventExecutable @('local_configs\train_common_dataset_prior.toml', '--log-level', 'info', '-s', '42') $root
    }
    'drugex_v2' {
        $root = Join-Path $project 'baselines\drugex_v2_adapter'
        if ($RebuildPreprocessedData) {
            Invoke-Checked $Python @('adapter_prepare_common_data.py', '--input', "$project\data\train_smiles_only.txt", '--output-dir', "$project\results\baselines\drugex_v2\data") $root
        }
        Invoke-Checked $Python @('adapter_train_common_prior.py', '--data-dir', "$project\results\baselines\drugex_v2\data", '--output-dir', "$project\results\baselines\drugex_v2\models", '--epochs', '20', '--batch-size', '512', '--workers', "$Workers", '--patience', '3', '--seed', '42') $root
    }
    'mo_lso' {
        $root = Join-Path $project 'baselines\mo_lso_adapter'
        if ($RebuildPreprocessedData) {
            Invoke-Checked $Python @('adapter_prepare_common_data.py', '--input', "$project\data\train_smiles_only.txt", '--output-dir', "$project\results\baselines\mo_lso\data\tensors_train", '--workers', "$Workers", '--shard-size', '5000') $root
        }
        Invoke-Checked $Python @('adapter_train_common_jtvae.py', '--source', "$project\data\train_smiles_only.txt", '--data-dir', "$project\results\baselines\mo_lso\data", '--output-dir', "$project\results\baselines\mo_lso\models", '--epochs', '30', '--batch-size', '32', '--workers', "$Workers", '--patience', '3', '--seed', '42', '--device', 'cuda:0') $root
    }
}
