[CmdletBinding()]
param(
 [Parameter(Mandatory)][ValidateSet('Auth','Validate','Push','Status','Files','Output')][string]$Action,
 [string]$Kernel, [string]$Folder, [string]$OutputDirectory, [string]$ExpectedUsername,
 [string]$KaggleExe='C:\Users\Nandini.P\Tools\kaggle-cli\Scripts\kaggle.exe'
)
$ErrorActionPreference='Stop'
function Invoke-Kaggle([string[]]$Arguments) {
 $result = @(& $KaggleExe @Arguments 2>&1)
 if ($LASTEXITCODE -ne 0) { throw "Kaggle CLI failed (exit $LASTEXITCODE): $($result -join [Environment]::NewLine)" }
 return ($result -join [Environment]::NewLine)
}
function Validate-Metadata {
 if (-not $Folder) { throw 'Folder is required' }
 $m=Get-Content -Raw -LiteralPath (Join-Path $Folder 'kernel-metadata.json') | ConvertFrom-Json
 if ($m.id -notmatch '^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$') { throw 'Invalid notebook id' }
 if ($m.kernel_type -ne 'notebook' -or $m.language -ne 'python') { throw 'Python notebook required' }
 if ($m.is_private -ne $true -or $m.enable_gpu -ne $true -or $m.enable_internet -ne $true) { throw 'Private GPU notebook with internet required' }
 if ($m.code_file -ne 'SemanticAlgorithms_Master.ipynb') { throw 'Use the single master notebook' }
 if (-not (Test-Path -LiteralPath (Join-Path $Folder $m.code_file))) { throw 'Notebook missing' }
 # push uses the CLI accelerator flag; notebook also verifies two CUDA devices.
 return $m
}
if (-not (Test-Path -LiteralPath $KaggleExe)) { throw 'Kaggle CLI executable missing' }
if ($Action -eq 'Auth') { & $KaggleExe auth login --force; if ($LASTEXITCODE) { throw 'Interactive authentication failed' }; return }
if ($Action -eq 'Validate') { Validate-Metadata | ConvertTo-Json -Depth 20; return }
if ($Action -eq 'Push') {
 if (-not $ExpectedUsername -or $ExpectedUsername -notmatch '^[a-zA-Z0-9_-]+$') { throw 'ExpectedUsername must be supplied explicitly after user confirms the intended account' }
 $meta=Validate-Metadata
 if (($meta.id -split '/')[0] -cne $ExpectedUsername) { throw 'Metadata owner differs from explicitly expected account' }
 $python=Join-Path (Split-Path $KaggleExe -Parent) 'python.exe'
 $identity=@(& $python -c 'from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); print(a.config_values.get("username", ""))' 2>&1)
 if ($LASTEXITCODE -ne 0 -or $identity.Count -ne 1 -or $identity[0].ToString().Trim() -cne $ExpectedUsername) { throw 'Authenticated account identity could not be verified as expected; interactive login required' }
 Invoke-Kaggle @('kernels','push','-p',$Folder,'--accelerator','NvidiaTeslaT4'); return }
if ($Kernel -notmatch '^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$') { throw 'Kernel must be owner/slug; use a stable private notebook id' }
if ($Action -eq 'Status') { Invoke-Kaggle @('kernels','status',$Kernel); return }
if ($Action -eq 'Output') {
 if (-not $OutputDirectory) { throw 'OutputDirectory required' }
 if (Test-Path -LiteralPath $OutputDirectory) { throw 'Use a fresh output directory to preserve prior downloads' }
 New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
}
$token=''; $seen=@{}; $pages=@()
do {
 $arguments=@('kernels', $(if ($Action -eq 'Files') {'files'} else {'output'}),$Kernel,'--page-size','200')
 if ($Action -eq 'Files') { $arguments+=@('--format','json') } else { $arguments+=@('-p',$OutputDirectory) }
 if ($token) { $arguments+=@('--page-token',$token) }
 $page=Invoke-Kaggle $arguments; $pages+=$page
 $match=[regex]::Match($page,'(?im)^Next Page Token\s*=\s*(.+)$')
 $token=if ($match.Success) {$match.Groups[1].Value.Trim()} else {''}
 if ($token) { if ($seen.ContainsKey($token)) { throw 'Repeated pagination token' }; $seen[$token]=$true }
} while ($token)
if ($Action -eq 'Files') { $pages; return }
$manifests=@(Get-ChildItem -LiteralPath $OutputDirectory -Recurse -File -Filter 'artifact_sha256.json')
if ($manifests.Count -ne 1) { throw 'Exactly one run artifact_sha256.json required for remote hash verification' }
$manifest=Get-Content -Raw -LiteralPath $manifests[0].FullName | ConvertFrom-Json
$base=$manifests[0].Directory.FullName
$verified=@()
foreach ($property in $manifest.PSObject.Properties) {
 $path=[IO.Path]::GetFullPath((Join-Path $base $property.Name))
 if (-not $path.StartsWith($base+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Unsafe manifest path' }
 if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing downloaded artifact: $($property.Name)" }
 $digest=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
 if ($digest -ne $property.Value) { throw "SHA256 mismatch: $($property.Name)" }
 $verified+=[pscustomobject]@{path=$property.Name;sha256=$digest}
}
$verified | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputDirectory 'download_verification.json')
$verified
