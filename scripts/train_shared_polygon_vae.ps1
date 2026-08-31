param(
    [string]$Python = '',
    [string]$Config = '',
    [switch]$SkipAugmentation,
    [switch]$SkipSelection,
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
if (-not $Python) { $Python = Resolve-ProjectPath ([string]$Protocol.executables.polygon_python) }
$Vae = $Protocol.shared_polygon_vae
$Source = Resolve-ProjectPath ([string]$Vae.source)
$Train = Resolve-ProjectPath ([string]$Vae.raw_training_data)
$Randomized = Resolve-ProjectPath ([string]$Vae.randomized_training_data)
$Output = Resolve-ProjectPath ([string]$Vae.output_dir)
$ModelBase = Join-Path $Output 'polygon_vae_best_valid_novel_stable.pt'
$Vocab = Join-Path $Output 'polygon_vae_best_valid_novel_stable_vocab.pkl'
$Log = Join-Path $Output 'train_log.csv'
$Quality = Join-Path $Output 'quality_checkpoints'
$TrainCache = Join-Path $Output 'train_canonical_cache.txt'
$SelectedTarget = Join-Path $Root 'models\polygon_vae_best_valid_novel_stable_020.pt'
$env:PYTHONPATH = $Source
$env:OMP_NUM_THREADS = '24'
$env:MKL_NUM_THREADS = '24'
$env:NUMEXPR_NUM_THREADS = '24'

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    if ($DryRun) { Write-Output ("DRYRUN " + $Executable + ' ' + ($Arguments -join ' ')); return }
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Executable" }
}

New-Item -ItemType Directory -Force -Path $Output, $Quality, (Split-Path -Parent $SelectedTarget) | Out-Null
if (-not $SkipAugmentation -and -not (Test-Path -LiteralPath $Randomized)) {
    Invoke-Checked $Python @(
        (Join-Path $PSScriptRoot 'make_randomized_smiles_training_set.py'),
        '--input', $Train, '--output', $Randomized,
        '--n-random', [string]$Vae.augmentation.canonical_plus_randomized,
        '--seed', [string]$Vae.augmentation.seed,
        '--workers', [string]$Vae.augmentation.workers,
        '--chunk-size', [string]$Vae.augmentation.chunk_size
    )
}
if (-not $DryRun -and -not (Test-Path -LiteralPath $Randomized)) { throw "Missing augmented training data: $Randomized" }

$T = $Vae.training
if (-not (Test-Path -LiteralPath $ModelBase)) {
    Invoke-Checked $Python @(
        '-m', 'polygon.run', 'train',
        '--train_data', $Randomized, '--model_save', $ModelBase, '--vocab_save', $Vocab,
        '--log_file', $Log, '--save_frequency', [string]$T.save_frequency,
        '--n_epoch', [string]$T.epochs, '--device', [string]$T.device, '--seed', [string]$T.seed,
        '--n_batch', [string]$T.batch_size, '--q_cell', 'gru', '--q_bidir',
        '--q_d_h', [string]$T.encoder_hidden, '--q_n_layers', [string]$T.encoder_layers,
        '--q_dropout', [string]$T.encoder_dropout, '--d_cell', 'gru',
        '--d_n_layers', [string]$T.decoder_layers, '--d_dropout', [string]$T.decoder_dropout,
        '--d_z', [string]$T.latent_dimension, '--d_d_h', [string]$T.decoder_hidden,
        '--lr_start', [string]$T.learning_rate_start, '--lr_end', [string]$T.learning_rate_end,
        '--kl_start', [string]$T.kl_start_epoch, '--kl_w_start', [string]$T.kl_weight_start,
        '--kl_w_end', [string]$T.kl_weight_end, '--clip_grad', [string]$T.gradient_clip,
        '--n_workers', [string]$T.data_loader_workers, '--lr_n_period', [string]$T.sgdr_period,
        '--lr_n_restarts', [string]$T.sgdr_restarts, '--lr_n_mult', [string]$T.sgdr_multiplier,
        '--m_dropout', [string]$T.middle_dropout, '--n_mid_layers', [string]$T.middle_layers,
        '--batchnorm_conv', [string]$T.batchnorm_conv, '--batchnorm_mid', [string]$T.batchnorm_middle,
        '--lambda_scale', [string]$T.lambda_scale, '--debug'
    )
}

if (-not $SkipSelection) {
    $S = $Vae.selection
    $Temps = (@($S.temperatures) | ForEach-Object { [string]$_ }) -join ','
    foreach ($Epoch in @($S.candidate_epochs)) {
        $Suffix = '{0:D3}' -f [int]$Epoch
        $Checkpoint = Join-Path $Output "polygon_vae_best_valid_novel_stable_$Suffix.pt"
        $CandidateOut = Join-Path $Quality "epoch_$Suffix"
        if ((Test-Path -LiteralPath $Checkpoint) -and -not (Test-Path -LiteralPath (Join-Path $CandidateOut 'vae_quality_by_temperature.csv'))) {
            Invoke-Checked $Python @(
                (Join-Path $PSScriptRoot 'evaluate_vae_quality.py'), '--model', $Checkpoint,
                '--train-data', $Train, '--train-cache', $TrainCache, '--output', $CandidateOut,
                '--device', [string]$T.device, '--samples', [string]$S.samples_per_candidate,
                '--batch-size', [string]$S.batch_size, '--max-len', [string]$S.max_smiles_length,
                '--seed', [string]$S.seed, '--temperatures', $Temps
            )
        }
    }
    if ((Test-Path -LiteralPath $ModelBase) -and -not (Test-Path -LiteralPath (Join-Path $Quality 'final\vae_quality_by_temperature.csv'))) {
        Invoke-Checked $Python @(
            (Join-Path $PSScriptRoot 'evaluate_vae_quality.py'), '--model', $ModelBase,
            '--train-data', $Train, '--train-cache', $TrainCache, '--output', (Join-Path $Quality 'final'),
            '--device', [string]$T.device, '--samples', [string]$S.samples_per_candidate,
            '--batch-size', [string]$S.batch_size, '--max-len', [string]$S.max_smiles_length,
            '--seed', [string]$S.seed, '--temperatures', $Temps
        )
    }
    $Table = Join-Path $Output 'quality_best_table.csv'
    Invoke-Checked $Python @((Join-Path $PSScriptRoot 'summarize_vae_quality.py'), '--quality-root', $Quality, '--output', $Table, '--top-k', '40')
    if (-not $DryRun) {
        $Winner = Import-Csv -LiteralPath $Table | Select-Object -First 1
        $WinnerPath = if ($Winner.model -eq 'final') { $ModelBase } else {
            $Epoch = [int]([string]$Winner.model -replace '^epoch_', '')
            Join-Path $Output ('polygon_vae_best_valid_novel_stable_{0:D3}.pt' -f $Epoch)
        }
        Copy-Item -LiteralPath $WinnerPath -Destination $SelectedTarget -Force
        [ordered]@{
            selected_model = $Winner.model
            temperature = [double]$Winner.temperature
            model_path = $WinnerPath
            installed_path = $SelectedTarget
            sha256 = (Get-FileHash -LiteralPath $SelectedTarget -Algorithm SHA256).Hash.ToLowerInvariant()
            selection_rule = [string]$S.rule
        } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Output 'selected_model.json') -Encoding UTF8
    }
}
Write-Output "VAE_STAGE_COMPLETE output=$Output installed=$SelectedTarget"
