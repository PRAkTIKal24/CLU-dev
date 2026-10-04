. /c/Users/PratikJawahar/chlu/chlu-bundle-v2/.claude/outputs/hub-deletion/hubenv.sh
# exits when: the cell finishes, a unit FAILs, or the driver disappears.
while true; do
  if [ -f "$HUB/LEDGER.tsv" ] && grep -q "FAIL" "$HUB/LEDGER.tsv"; then
    echo "FAIL detected"; grep FAIL "$HUB/LEDGER.tsv"; exit 2; fi
  if grep -q "^LEDGER:" "$HUB/logs/drive.log" 2>/dev/null; then
    echo "CELL COMPLETE"; exit 0; fi
  if ! ps -W 2>/dev/null | grep -q "[d]rive.sh" && ! pgrep -f "drive.sh" >/dev/null 2>&1; then
    if [ ! -f "$HUB/LEDGER.tsv" ] || ! grep -q "^LEDGER:" "$HUB/logs/drive.log" 2>/dev/null; then
      # driver gone without finishing -> only report if no unit is running either
      if ! tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -qE "python.exe +[0-9]+ +Console +1 +[0-9]{3},"; then
        echo "DRIVER GONE without completing"; tail -5 "$HUB/logs/drive.log" 2>/dev/null; exit 3; fi
    fi
  fi
  sleep 60
done
