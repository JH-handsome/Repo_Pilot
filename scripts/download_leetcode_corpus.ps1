$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/cnkyrpsgl/leetcode.git"
$TargetDir = Join-Path $PSScriptRoot "..\datasets\leetcode-python"

if (Test-Path $TargetDir) {
    Write-Host "Corpus already exists: $TargetDir"
    Write-Host "Patching local LeetCode imports..."
    python (Join-Path $PSScriptRoot "patch_leetcode_imports.py") $TargetDir
    exit 0
}

git clone --depth 1 $RepoUrl $TargetDir
python (Join-Path $PSScriptRoot "patch_leetcode_imports.py") $TargetDir

Write-Host ""
Write-Host "Downloaded corpus to: $TargetDir"
Write-Host "Try:"
Write-Host "python main.py datasets `"two sum hash map`" --top-k 5"
Write-Host "python main.py datasets `"binary tree level order null TreeNode`" --top-k 5"
