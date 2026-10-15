<#
.SYNOPSIS
  Render test speech turns with the Windows SAPI voices (16 kHz, 16-bit, mono WAV).

.DESCRIPTION
  Reads a JSON array of turns: [{"voice": "Microsoft David Desktop", "text": "...", "file": "turn_00.wav"}, ...]
  and writes one WAV per turn into -OutDir. Used by apps/api/tests/fixtures/build_fixture.py, which
  concatenates the turns into the listener test fixture.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts/make_test_audio.ps1 -ScriptJson turns.json -OutDir out
#>
param(
    [Parameter(Mandatory = $true)][string]$ScriptJson,
    [Parameter(Mandatory = $true)][string]$OutDir,
    [int]$Rate = 0
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech

if (-not (Test-Path -LiteralPath $ScriptJson)) { throw "Turn script not found: $ScriptJson" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$turns = Get-Content -LiteralPath $ScriptJson -Raw -Encoding UTF8 | ConvertFrom-Json
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono
)

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $installed = $synth.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }
    $synth.Rate = $Rate
    foreach ($turn in $turns) {
        if ($installed -notcontains $turn.voice) { throw "SAPI voice not installed: $($turn.voice)" }
        $path = Join-Path $OutDir $turn.file
        $synth.SelectVoice($turn.voice)
        $synth.SetOutputToWaveFile($path, $format)
        $synth.Speak([string]$turn.text)
        $synth.SetOutputToNull()
        Write-Output "wrote $path"
    }
}
finally {
    $synth.Dispose()
}
