# Expose the local app (http://localhost:8000) on a public HTTPS URL with no account.
# Prefers cloudflared (trycloudflare.com); falls back to localhost.run over SSH.
param([int]$Port = 8000)

$ErrorActionPreference = "Stop"
try { Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 5 | Out-Null }
catch { Write-Error "Nothing is answering on http://127.0.0.1:$Port. Start the app first (uv run python -m wsw_brain --port $Port)."; exit 1 }

$cf = Get-Command cloudflared -ErrorAction SilentlyContinue
if ($cf) {
    Write-Host "Opening a Cloudflare quick tunnel. Copy the https://*.trycloudflare.com URL printed below."
    & $cf.Source tunnel --no-autoupdate --url "http://127.0.0.1:$Port"
    exit $LASTEXITCODE
}

$ssh = Get-Command ssh -ErrorAction SilentlyContinue
if (-not $ssh) { Write-Error "Neither cloudflared nor ssh is available. Install one: winget install Cloudflare.cloudflared"; exit 1 }
Write-Host "Opening a localhost.run tunnel over SSH. Copy the https://*.lhr.life URL printed below."
& $ssh.Source -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -R "80:127.0.0.1:$Port" nokey@localhost.run
