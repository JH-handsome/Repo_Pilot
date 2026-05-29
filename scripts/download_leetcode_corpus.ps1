$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/cnkyrpsgl/leetcode.git"
$TargetDir = Join-Path $PSScriptRoot "..\datasets\leetcode-python"

if (Test-Path $TargetDir) {
    Write-Host "语料库已存在: $TargetDir"
    Write-Host "正在补充本地 LeetCode 类型导入..."
    python (Join-Path $PSScriptRoot "patch_leetcode_imports.py") $TargetDir
    exit 0
}

git clone --depth 1 $RepoUrl $TargetDir
python (Join-Path $PSScriptRoot "patch_leetcode_imports.py") $TargetDir

Write-Host ""
Write-Host "语料库已下载到: $TargetDir"
Write-Host "可以试试:"
Write-Host "python main.py datasets `"two sum hash map`" --top-k 5"
Write-Host "python main.py datasets `"binary tree level order null TreeNode`" --top-k 5"
