#!/bin/bash
# Vendor hotfix patch builder (F-1 mamba COW + R-1 ragged cap).
# Produces git-apply patches in hotfix/ for each vendor tag.
set -euo pipefail
cd /workspace/repo
mkdir -p hotfix
git worktree prune >/dev/null 2>&1 || true
for TAG in v0.5.14 v0.5.15.post1 v0.5.17 v0.5.20; do
  WT=/tmp/hotfix-wt-$TAG
  git worktree remove -f $WT >/dev/null 2>&1 || true
  rm -rf $WT
  git worktree add -q --detach $WT $TAG
  python3 hotfix/patch_f1.py $WT $TAG >/dev/null
  cd $WT && git diff --binary --no-color --src-prefix=a/ --dst-prefix=b/ > /workspace/repo/hotfix/F1-mamba-cow.$TAG.patch
  cd /workspace/repo
  echo "F1-mamba-cow.$TAG.patch: $(wc -l < hotfix/F1-mamba-cow.$TAG.patch) lines"
done
echo done
