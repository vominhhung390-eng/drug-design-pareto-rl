param([string]$PackageRoot = (Split-Path -Parent $PSScriptRoot))

$ErrorActionPreference = 'Stop'
$PackageRoot = (Resolve-Path -LiteralPath $PackageRoot).Path
$ManifestPath = Join-Path $PackageRoot 'MANIFEST_文件清单.csv'
$SourceMapPath = Join-Path $PackageRoot 'SOURCE_MAP_来源映射.csv'
$KeyHashPath = Join-Path $PackageRoot 'SHA256SUMS_关键文件.csv'
$AuditPath = Join-Path $PackageRoot 'PACKAGE_AUDIT.json'
$Generated = @($ManifestPath, $SourceMapPath, $KeyHashPath, $AuditPath)

function Relative-Path([string]$Path) {
    [IO.Path]::GetRelativePath($PackageRoot, $Path).Replace('\', '/')
}
function Category([string]$Relative) {
    switch -Regex ($Relative) {
        '^method/' { 'CLOVER-Mol source'; break }
        '^baselines/' { 'baseline source and adapter'; break }
        '^data/predictor_target_pairs/' { 'predictor training data'; break }
        '^data/' { 'shared generation data'; break }
        '^models/oracles/' { 'fixed historical first-pair oracle'; break }
        '^scripts/' { 'one-click reproduction entrypoint'; break }
        '^evaluation/|^analysis/' { 'evaluation and paper table code'; break }
        '^docking/' { 'docking source, receptor or protocol asset'; break }
        '^reference_results/' { 'compact verified reference summary'; break }
        '^config/' { 'machine-readable experiment parameters'; break }
        '^docs/' { 'reproduction documentation'; break }
        '^environments/' { 'environment version snapshot'; break }
        '^vendor/' { 'vendored training source'; break }
        '^tools/' { 'reproduction tool'; break }
        default { 'project entrypoint or metadata' }
    }
}

$SourceMap = @(
    [pscustomobject]@{PackagePath='method/';Source='CLOVER-Mol author workspace snapshot';Purpose='CLOVER-Mol V4-B method source';Included='yes'},
    [pscustomobject]@{PackagePath='baselines/';Source='Pinned upstream repositories plus fair-protocol adapters';Purpose='Five formal baseline sources and fair-protocol adapters';Included='yes'},
    [pscustomobject]@{PackagePath='vendor/polygon-main/';Source='Pinned POLYGON upstream source snapshot';Purpose='Historical shared VAE training implementation';Included='yes'},
    [pscustomobject]@{PackagePath='data/train_smiles_only.txt';Source='Author-provided shared SMILES dataset';Purpose='Shared bottom-model training data';Included='yes'},
    [pscustomobject]@{PackagePath='data/predictor_target_pairs/';Source='Recovered BindingDB plus formal ChEMBL 37 tables';Purpose='Optional first-pair retraining data plus formal second-pair RF training data';Included='yes'},
    [pscustomobject]@{PackagePath='models/oracles/';Source='Author-retained historical Polygon model snapshot';Purpose='Formal fixed EGFR/VEGFR2 oracle models; exact original training rows unavailable';Included='yes'},
    [pscustomobject]@{PackagePath='docking/';Source='CLOVER-Mol docking protocol snapshot';Purpose='Four receptors, protocol validation and Top10 docking scripts';Included='yes'},
    [pscustomobject]@{PackagePath='reference_results/';Source='Verified completed formal outputs';Purpose='Small paper tables and docking summaries only';Included='yes'},
    [pscustomobject]@{PackagePath='other models/, results/, outputs/ bulky contents';Source='Rebuilt by one-click pipeline';Purpose='Bottom-model weights, caches and historical full runs';Included='no'},
    [pscustomobject]@{PackagePath='retired baseline assets';Source='Retired method';Purpose='Excluded by user request';Included='no'}
)
$SourceMap | Export-Csv -LiteralPath $SourceMapPath -NoTypeInformation -Encoding UTF8

$KeyFiles = [ordered]@{
    pipeline_config = 'config/reproduction_pipeline.json'
    training_config_index = 'docs/training_configs/README.md'
    shared_smiles_dataset = 'data/train_smiles_only.txt'
    egfr_recovered_snapshot = 'data/predictor_target_pairs/01_EGFR_VEGFR2_第一组_恢复相关数据_NOT_EXACT_ORIGINAL/EGFR_P00533_BindingDB_API_snapshot_20260712.json'
    vegfr2_recovered_snapshot = 'data/predictor_target_pairs/01_EGFR_VEGFR2_第一组_恢复相关数据_NOT_EXACT_ORIGINAL/VEGFR2_P35968_BindingDB_API_snapshot_20260712.json'
    egfr_historical_oracle = 'models/oracles/target_EGFR_model.pkl'
    vegfr2_historical_oracle = 'models/oracles/target_VEGFR2_model.pkl'
    historical_oracle_metadata = 'models/oracles/HISTORICAL_MODEL_METADATA.json'
    parp1_formal_training = 'data/predictor_target_pairs/02_PARP1_BRD4_第二组_当前正式预测器数据_ChEMBL37/PARP1_CHEMBL3105_train_through_2023_n2538.csv'
    brd4_formal_training = 'data/predictor_target_pairs/02_PARP1_BRD4_第二组_当前正式预测器数据_ChEMBL37/BRD4_CHEMBL1163125_train_through_2023_n5245.csv'
    one_click = 'scripts/reproduce_all.ps1'
    predictor_training = 'scripts/train_four_rf_predictors.py'
    shared_vae_training = 'scripts/train_shared_polygon_vae.ps1'
    paper_tables = 'scripts/build_paper_tables.py'
    vina_install_notice = 'tools/autodock_vina_1.1.2/README.md'
    reference_main_table_a = 'reference_results/paper_tables/Table2_Table3_EGFR_VEGFR2.md'
    reference_main_table_b = 'reference_results/paper_tables/Table2_Table3_PARP1_BRD4.md'
    reference_docking = 'reference_results/docking_top10/docking_top10_compound_summary.csv'
    reference_docking_tasks = 'reference_results/docking_top10/docking_task_status.csv'
}
$KeyHashes = foreach ($Entry in $KeyFiles.GetEnumerator()) {
    $Path = Join-Path $PackageRoot $Entry.Value
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing key file: $($Entry.Value)" }
    $Item = Get-Item -LiteralPath $Path
    [pscustomobject]@{
        Asset = $Entry.Key
        RelativePath = $Entry.Value
        SizeBytes = $Item.Length
        SHA256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$KeyHashes | Export-Csv -LiteralPath $KeyHashPath -NoTypeInformation -Encoding UTF8

$DatasetHash = ($KeyHashes | Where-Object Asset -eq 'shared_smiles_dataset').SHA256
$LineCount = 0
$Reader = [IO.File]::OpenText((Join-Path $PackageRoot 'data\train_smiles_only.txt'))
try { while ($null -ne $Reader.ReadLine()) { $LineCount++ } } finally { $Reader.Dispose() }
$ReferenceDocking = Import-Csv -LiteralPath (Join-Path $PackageRoot 'reference_results\docking_top10\docking_top10_compound_summary.csv')
$ReferenceTasks = Import-Csv -LiteralPath (Join-Path $PackageRoot 'reference_results\docking_top10\docking_task_status.csv')
$RetiredPattern = '(?i)' + 'moth' + 'ra'
$NamedRetired = @(Get-ChildItem -LiteralPath $PackageRoot -Recurse -Force | Where-Object Name -Match $RetiredPattern | ForEach-Object { Relative-Path $_.FullName })
$AllowedPackagedWeights = @(
    'models/oracles/target_EGFR_model.pkl',
    'models/oracles/target_VEGFR2_model.pkl'
)
$UnexpectedWeights = @(
    Get-ChildItem -LiteralPath (Join-Path $PackageRoot 'models') -Recurse -File |
        Where-Object {
            $_.Extension -Match '^\.(pt|pkl|pkg|model|ckpt)$' -and
            (Relative-Path $_.FullName) -notin $AllowedPackagedWeights
        }
)
$EgfrHistoricalHash = ($KeyHashes | Where-Object Asset -eq 'egfr_historical_oracle').SHA256
$Vegfr2HistoricalHash = ($KeyHashes | Where-Object Asset -eq 'vegfr2_historical_oracle').SHA256
$Audit = [ordered]@{
    schema_version = 3
    package_root = '.'
    generated_at = (Get-Date).ToString('o')
    release_profile = 'lean source-data-config package with two required historical first-pair oracle models; other trained weights and bulky runs excluded'
    package_size_bytes = (Get-ChildItem -LiteralPath $PackageRoot -Recurse -File -Force | Measure-Object Length -Sum).Sum
    shared_dataset = [ordered]@{
        line_count = $LineCount
        sha256 = $DatasetHash
        verified = ($LineCount -eq 1584663 -and $DatasetHash -eq '4301e7f6118839465012eb93510328681ef4b7b24642e8748c4ad40971f4a304')
    }
    methods = @('CLOVER-Mol','POLYGON shared augmented VAE','REINVENT4','DrugEx v2','MO-LSO','GraphPareto-NSGA-II')
    formal_protocol = [ordered]@{ target_pairs=2; methods=6; seeds_per_cell=10; formal_runs=120; oracle_budget_per_seed=10240 }
    training_configuration = [ordered]@{
        method_documents = @(Get-ChildItem -LiteralPath (Join-Path $PackageRoot 'docs\training_configs') -Filter '*.md').Count
        machine_readable_pipeline = Test-Path -LiteralPath (Join-Path $PackageRoot 'config\reproduction_pipeline.json')
        one_click_entrypoint = Test-Path -LiteralPath (Join-Path $PackageRoot 'scripts\reproduce_all.ps1')
    }
    predictor_data = [ordered]@{
        parp1_brd4_formal_tables_present = $true
        egfr_vegfr2_exact_historical_rows_present = $false
        egfr_vegfr2_recovered_snapshots_present = $true
        historical_fixed_models_present = $true
        historical_fixed_model_hashes_verified = (
            $EgfrHistoricalHash -eq 'd57ba46d71c7a943c3e17a6a6a688d55d48d9cbed93fa1429d95dedb85ae03ab' -and
            $Vegfr2HistoricalHash -eq 'c2cfd492cfc0fa367dd8a262f5716b7eea4d1c2de3ff25acccdd96bae9eeeb94'
        )
        default_formal_first_pair_oracle = 'models/oracles fixed historical models'
        recovered_retraining_requires_explicit_opt_in = $true
    }
    polygon_identity = 'POLYGON adapter using the shared canonical+3 randomized-SMILES VAE; not strict no-augmentation original'
    docking_reference = [ordered]@{
        compound_rows = $ReferenceDocking.Count
        raw_tasks = $ReferenceTasks.Count
        successful_tasks = @($ReferenceTasks | Where-Object status -eq 'ok').Count
        expected_compounds = 1200
        expected_tasks = 2400
    }
    third_party_runtime = [ordered]@{
        historical_vina_version = '1.1.2'
        vina_binary_bundled = $false
        supply_via = 'VINA_EXECUTABLE or tools/autodock_vina_1.1.2/vina.exe'
    }
    compact_reference_tables_present = @(
        (Test-Path -LiteralPath (Join-Path $PackageRoot 'reference_results\paper_tables\Table2_Table3_EGFR_VEGFR2.md')),
        (Test-Path -LiteralPath (Join-Path $PackageRoot 'reference_results\paper_tables\Table2_Table3_PARP1_BRD4.md'))
    )
    unexpected_packaged_weights = @($UnexpectedWeights | ForEach-Object { Relative-Path $_.FullName })
    excluded_method_named_paths = $NamedRetired
    known_blockers = @(
        'Exact historical EGFR/VEGFR2 training rows are unavailable; formal output-level reproduction uses verified fixed historical models, while recovered BindingDB retraining is an explicit alternate condition.',
        'POLYGON must be reported as using the shared augmented VAE, not as strict no-augmentation original.',
        'The historical AutoDock Vina 1.1.2 executable is a user-supplied third-party runtime and is not redistributed.',
        'CLOVER-Mol top-level LICENSE and CITATION.cff require author choice and metadata before public release.'
    )
}
$Audit | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $AuditPath -Encoding UTF8

$Manifest = foreach ($File in Get-ChildItem -LiteralPath $PackageRoot -File -Recurse -Force | Sort-Object FullName) {
    if ($Generated -contains $File.FullName) { continue }
    $Relative = Relative-Path $File.FullName
    [pscustomobject]@{
        RelativePath = $Relative
        Category = Category $Relative
        SizeBytes = $File.Length
        SHA256 = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$Manifest | Export-Csv -LiteralPath $ManifestPath -NoTypeInformation -Encoding UTF8
Write-Output "INVENTORY_COMPLETE files=$($Manifest.Count) root=$PackageRoot"
