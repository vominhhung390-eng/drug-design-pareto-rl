param(
    [string]$OutputDir = "",
    [int]$PageSize = 2000,
    [int]$MaxParallel = 4
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
    $OutputDir = Join-Path $ProjectRoot "data\external\chembl\predictor_validation"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$targets = [ordered]@{
    EGFR = "CHEMBL203"
    VEGFR2 = "CHEMBL279"
}

$jobs = @()
foreach ($target in $targets.Keys) {
    $targetId = $targets[$target]
    $baseQuery = "target_chembl_id=$targetId&standard_type__in=IC50%2CKd&standard_relation=%3D&pchembl_value__isnull=false&assay_type=B&assay_variant_mutation__isnull=true"
    $probeUrl = "https://www.ebi.ac.uk/chembl/api/data/activity.json?$baseQuery&limit=1&offset=0"
    $probe = Invoke-RestMethod -Uri $probeUrl -TimeoutSec 240
    $total = [int]$probe.page_meta.total_count
    Write-Output "$target total=$total"

    for ($offset = 0; $offset -lt $total; $offset += $PageSize) {
        $output = Join-Path $OutputDir ("{0}_offset{1}_limit{2}.json" -f $target.ToLower(), $offset, $PageSize)
        if ((Test-Path -LiteralPath $output) -and (Get-Item -LiteralPath $output).Length -gt 1000) {
            Write-Output "cached $output"
            continue
        }
        while (($jobs | Where-Object { $_.State -eq "Running" }).Count -ge $MaxParallel) {
            $done = Wait-Job -Job $jobs -Any -Timeout 30
            if ($done) {
                Receive-Job -Job $done
                if ($done.State -eq "Failed") { throw "Download job failed: $($done.Name)" }
                Remove-Job -Job $done
                $jobs = @($jobs | Where-Object { $_.Id -ne $done.Id })
            }
        }
        $url = "https://www.ebi.ac.uk/chembl/api/data/activity.json?$baseQuery&limit=$PageSize&offset=$offset"
        $jobs += Start-Job -Name "$target-$offset" -ScriptBlock {
            param($Url, $Output)
            $ErrorActionPreference = "Stop"
            for ($attempt = 1; $attempt -le 6; $attempt++) {
                try {
                    Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 300 -OutFile $Output
                    $payload = Get-Content -Raw -LiteralPath $Output -Encoding UTF8 | ConvertFrom-Json
                    if ($null -eq $payload.activities) { throw "Missing activities array" }
                    Write-Output "downloaded $Output rows=$($payload.activities.Count)"
                    return
                } catch {
                    if ($attempt -eq 6) { throw }
                    Start-Sleep -Seconds ([Math]::Min(30, [Math]::Pow(2, $attempt)))
                }
            }
        } -ArgumentList $url, $output
    }
}

while ($jobs.Count -gt 0) {
    $done = Wait-Job -Job $jobs -Any -Timeout 30
    if ($done) {
        Receive-Job -Job $done
        if ($done.State -eq "Failed") { throw "Download job failed: $($done.Name)" }
        Remove-Job -Job $done
        $jobs = @($jobs | Where-Object { $_.Id -ne $done.Id })
    }
}

Write-Output "ChEMBL predictor-validation pages complete: $OutputDir"
