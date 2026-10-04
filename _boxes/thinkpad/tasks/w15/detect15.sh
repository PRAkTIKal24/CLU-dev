NOW=$(date +%s)
for t in cmaze-t-harness cmaze-o-harness single-design-seeds crossover-seeds; do
  if ! tmux has-session -t w15-$t 2>/dev/null; then echo "GONE $t"; continue; fi
  PANE=$(tmux capture-pane -p -t w15-$t -S -6 2>/dev/null)
  OD=/c/chlu/outputs/w15-$t
  if [ -f "$OD/LOG.md" ]; then AGE=$(( NOW - $(stat -c %Y "$OD/LOG.md") )); else AGE=99999; fi
  if echo "$PANE" | grep -qE 'Do you want to proceed|Esc to cancel . Tab to amend|No, exit'; then
    echo "BLOCK $t (log ${AGE}s)"; continue; fi
  if echo "$PANE" | grep -qE 'API Error|Overloaded|rate.limit'; then
    echo "APIERR $t (log ${AGE}s)"; continue; fi
  # LOG-RECENCY GUARD: fresh log => ACTIVE regardless of pane
  if [ "$AGE" -lt 600 ]; then echo "ACTIVE $t (log ${AGE}s)"; continue; fi
  if echo "$PANE" | grep -q 'esc to interrupt'; then echo "WORK $t (log ${AGE}s)"; continue; fi
  if echo "$PANE" | grep -qE '[0-9]+ (shells|monitors)|to see subagents'; then echo "BUSYBG $t (log ${AGE}s)"; continue; fi
  echo "IDLE $t (log ${AGE}s)"
done
# wave-16 gate
G=0
for h in cmaze-t-harness cmaze-o-harness; do
  F=/c/chlu/outputs/w15-$h/HARNESS-READY.json
  [ -f "$F" ] && grep -q '"ready"' "$F" 2>/dev/null && G=$((G+1))
done
echo "GATE16 $G/2"
