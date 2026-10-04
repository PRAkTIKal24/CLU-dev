#!/usr/bin/env bash
# Hub-only: install batmac ~/clu/pw2/live/{v2,style} back into the Head's two Desktop folders.
# Refuses if a Desktop folder changed since the last snapshot (the Head may be editing).
set -u
O=/Users/user/Desktop/naive-CLU-final-v2fina_subF; S=/Users/user/Desktop/naive-CLU-iclr-v2fina-stylesubF
ROOT=/Users/user/Desktop/CHLU/.claude/outputs/pw2-desktop-snapshots; LAST=$(cat /tmp/pw2-last-snap); TAG=${1:?tag}
for k in v2 style; do D=$O; [ $k = style ] && D=$S
  (cd $D && find . -type f ! -name .DS_Store -print0 | sort -z | xargs -0 shasum -a 256) > /tmp/pw2-now-$k.txt
  diff -q /tmp/pw2-now-$k.txt $LAST/SHA256-$k.txt >/dev/null || { echo "REFUSING: $D changed since $LAST (Head edit?)"; diff /tmp/pw2-now-$k.txt $LAST/SHA256-$k.txt | head; exit 1; }
done
TS=$(date +%Y%m%d-%H%M%S); SNAP=$ROOT/$TS-before-$TAG; mkdir -p $SNAP
for k in v2 style; do D=$O; [ $k = style ] && D=$S; rsync -a --exclude .DS_Store $D/ $SNAP/$k/; cp /tmp/pw2-now-$k.txt $SNAP/SHA256-$k.txt; done
NEW=$ROOT/$TS-after-$TAG; mkdir -p $NEW
for k in v2 style; do D=$O; [ $k = style ] && D=$S
  rsync -a batmac@batmac:clu/pw2/live/$k/ $NEW/$k/
  (cd $NEW/$k && find . -type f -print0 | sort -z | xargs -0 shasum -a 256) > $NEW/SHA256-$k.txt
  ssh -o BatchMode=yes batmac@batmac "cd ~/clu/pw2/live/$k && find . -type f -print0 | sort -z | xargs -0 shasum -a 256" | diff -q - $NEW/SHA256-$k.txt >/dev/null || { echo "PULL MISMATCH $k"; exit 1; }
  rsync -a $NEW/$k/ $D/
  (cd $D && find . -type f ! -name .DS_Store -print0 | sort -z | xargs -0 shasum -a 256) | diff -q - $NEW/SHA256-$k.txt >/dev/null && echo "INSTALLED $k -> $D ($(wc -l < $NEW/SHA256-$k.txt | tr -d ' ') files, sha256-equal)" || { echo "INSTALL MISMATCH $k"; exit 1; }
done
echo $NEW > /tmp/pw2-last-snap; echo "snapshot-before: $SNAP"
