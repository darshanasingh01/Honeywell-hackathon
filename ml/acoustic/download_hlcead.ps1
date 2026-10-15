param(
    [string]$Destination = "ml/datasets/hl_cead"
)

$ErrorActionPreference = "Stop"
$expectedSizes = @(
    25165824, 25165824, 25165824, 25165824, 25165824,
    25165824, 25165824, 25165824, 15150862
)
$destinationPath = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null

for ($i = 1; $i -le 9; $i++) {
    $suffix = $i.ToString("000")
    $name = "MLVRG-Car-Engine-Audio.zip.$suffix"
    $url = "https://raw.githubusercontent.com/MachineLearningVisionRG/machine_biometrics/main/$name"
    $target = Join-Path $destinationPath $name
    $partial = "$target.part"
    $expected = [int64]$expectedSizes[$i - 1]

    if (Test-Path -LiteralPath $target) {
        $existing = (Get-Item -LiteralPath $target).Length
        if ($existing -ne $expected) {
            throw "$name exists with $existing bytes; expected $expected. Move it aside before retrying."
        }
        Write-Host "Verified existing $name ($existing bytes)."
        continue
    }

    if ((Test-Path -LiteralPath $partial) -and (Get-Item -LiteralPath $partial).Length -gt $expected) {
        throw "$partial is larger than expected ($expected bytes); refusing to resume."
    }

    $curlArgs = @("--location", "--fail", "--show-error", "--connect-timeout", "20", "--max-time", "900", "--retry", "0", "--progress-bar", "--output", $partial)
    if (Test-Path -LiteralPath $partial) {
        $curlArgs += @("--continue-at", "-")
    }
    $curlArgs += $url
    Write-Host "Downloading $name from $url"
    & curl.exe @curlArgs
    $curlExit = $LASTEXITCODE

    # curl exit 33 means the origin rejected a range resume. Restart this part
    # from byte zero only; never treat the failed partial response as complete.
    if ($curlExit -eq 33) {
        Write-Host "Range resume unavailable; restarting $name from byte zero."
        & curl.exe --location --fail --show-error --connect-timeout 20 --max-time 900 --retry 0 --progress-bar --output $partial $url
        $curlExit = $LASTEXITCODE
    }
    if ($curlExit -ne 0) {
        throw "Download failed for $url (curl exit $curlExit). Partial bytes retained at $partial."
    }

    $actual = (Get-Item -LiteralPath $partial).Length
    if ($actual -ne $expected) {
        throw "Size verification failed for $url`: received $actual bytes; expected $expected. Partial file retained at $partial."
    }
    Move-Item -LiteralPath $partial -Destination $target
    Write-Host "Verified $name ($actual bytes)."
}

$verified = 0
for ($i = 1; $i -le 9; $i++) {
    $name = "MLVRG-Car-Engine-Audio.zip.$($i.ToString('000'))"
    $path = Join-Path $destinationPath $name
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing archive part: $path" }
    $size = (Get-Item -LiteralPath $path).Length
    if ($size -ne $expectedSizes[$i - 1]) { throw "Size mismatch: $path has $size bytes; expected $($expectedSizes[$i - 1])." }
    $verified += $size
}
if ($verified -ne 216477454) { throw "Aggregate size mismatch: got $verified bytes; expected 216477454." }
Write-Host "All nine parts verified: $verified bytes total."
