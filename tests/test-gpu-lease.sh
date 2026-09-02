#!/usr/bin/env bash
# test-gpu-lease.sh — offline test matrix for gpu-lease's exclusive + shared modes.
#
# Runs the real script against stubbed nvidia-smi / systemctl / sudo / curl /
# ollama (tests/stub) and a throwaway XDG_RUNTIME_DIR, so nothing touches the
# live lease, the live GPU or voxtral.service.  Safe to run anywhere.
#
#   ./tests/test-gpu-lease.sh            # run all
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LEASE="$HERE/../gpu-lease"
export GPU_LEASE_BIN_OVERRIDE="$HERE/stub"
export GPU_LEASE_NO_TIMER=1
export GPU_LEASE_HEALTH_URL="http://stub/health"
export GPU_LEASE_METRICS_URL="http://stub/metrics"
export GPU_LEASE_CAPACITY_MB=24576
export GPU_LEASE_RESERVE_MB=5120          # budget = 19456 MiB

PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); printf '  ok   %s\n' "$*"; }
bad()  { FAIL=$((FAIL+1)); printf '  FAIL %s\n' "$*"; }
chk()  { if [ "$2" = "$3" ]; then ok "$1 ($2)"; else bad "$1: expected '$3', got '$2'"; fi; }

setup() {                      # fresh isolated state; voxtral starts active
  export XDG_RUNTIME_DIR; XDG_RUNTIME_DIR="$(mktemp -d /tmp/gpulease-test.XXXXXX)"
  export STUB_STATE="$XDG_RUNTIME_DIR/stub"; mkdir -p "$STUB_STATE"
  echo active > "$STUB_STATE/voxtral"; echo 0 > "$STUB_STATE/requests"; : > "$STUB_STATE/actions"
}
teardown() { rm -rf "$XDG_RUNTIME_DIR"; }
vox()      { cat "$STUB_STATE/voxtral"; }
nsup()     { jq -r '.suppressors|length' "$XDG_RUNTIME_DIR/gpu-lease.shared.json" 2>/dev/null || echo 0; }
sumsup()   { jq -r '[.suppressors[].vram_mb]|add // 0' "$XDG_RUNTIME_DIR/gpu-lease.shared.json" 2>/dev/null || echo 0; }
gl()       { "$LEASE" "$@" >"$XDG_RUNTIME_DIR/out" 2>"$XDG_RUNTIME_DIR/err"; echo $?; }
errtail()  { tail -2 "$XDG_RUNTIME_DIR/err"; }
banner()   { printf '\n== %s ==\n' "$*"; }

# --------------------------------------------------------------- 1. refcount
banner "1. refcount 0->1->2->1->0 and the voxtral transitions"
setup
chk "voxtral starts active" "$(vox)" active
chk "suppress A rc" "$(gl suppress trackA --vram 9200)" 0
chk "  voxtral stopped on 0->1" "$(vox)" inactive
chk "  refcount" "$(nsup)" 1
chk "suppress B rc" "$(gl suppress capB --vram 5400)" 0
chk "  refcount" "$(nsup)" 2
chk "  declared MiB" "$(sumsup)" 14600
chk "  no second voxtral bounce" "$(grep -c 'start voxtral' "$STUB_STATE/actions")" 0
chk "unsuppress B rc" "$(gl unsuppress capB)" 0
chk "  refcount" "$(nsup)" 1
chk "  voxtral still down (A remains)" "$(vox)" inactive
chk "unsuppress A rc" "$(gl unsuppress trackA)" 0
chk "  refcount" "$(nsup)" 0
chk "  voxtral restarted on 1->0" "$(vox)" active
teardown

# ------------------------------------------------------------- 2. VRAM gate
banner "2. suppress refused on the VRAM gate"
setup
chk "suppress 9200 rc" "$(gl suppress trackA --vram 9200)" 0
chk "suppress 5400 rc" "$(gl suppress capB --vram 5400)" 0     # 14600 <= 19456
rc="$(gl suppress bigC --vram 6000)"                            # 20600 > 19456
chk "suppress 6000 refused" "$rc" 1
case "$(errtail)" in *"VRAM budget"*) ok "  refusal names the budget" ;; *) bad "  refusal reason: $(errtail)" ;; esac
chk "  refcount unchanged" "$(nsup)" 2
chk "oversized single request refused" "$(gl suppress hugeD --vram 20000)" 1
case "$(errtail)" in *"exceeds the whole shared budget"*) ok "  oversized names 'claim'" ;; *) bad "  oversized reason: $(errtail)" ;; esac
chk "a fitting request is still granted" "$(gl suppress smallE --vram 4800)" 0
chk "  refcount" "$(nsup)" 3
teardown

# ------------------------------------------------- 3. claim blocked by shared
banner "3. claim blocked by suppressors; suppress blocked by claim"
setup
chk "suppress rc" "$(gl suppress trackA --vram 9200)" 0
chk "claim refused" "$(gl claim qwen)" 1
case "$(errtail)" in *"suppressor"*) ok "  refusal names the suppressor" ;; *) bad "  refusal reason: $(errtail)" ;; esac
chk "  no lock written" "$([ -f "$XDG_RUNTIME_DIR/gpu-lease.json" ] && echo yes || echo no)" no
chk "claim --force overrides" "$(gl claim qwen --force)" 0
chk "  lock written" "$([ -f "$XDG_RUNTIME_DIR/gpu-lease.json" ] && echo yes || echo no)" yes
chk "suppress refused while lease held" "$(gl suppress capB --vram 5400)" 1
case "$(errtail)" in *"exclusive lease"*) ok "  refusal names the lease" ;; *) bad "  refusal reason: $(errtail)" ;; esac
teardown

# ------------------------------ 4. release must not start voxtral under shared
banner "4. release leaves voxtral down while suppressors remain"
setup
chk "claim rc" "$(gl claim qwen)" 0
chk "  voxtral down" "$(vox)" inactive
# a suppressor sneaks in via --force-style direct state write is not allowed;
# instead release the lease first, suppress, then re-claim --force to model the
# real hazard: an exclusive holder releasing while shared tenants are running.
chk "suppress under the lease (--force)" "$(gl suppress trackA --vram 9200 --force)" 1
# correct path: drop lease, take suppressor, then a stale release arrives
chk "release rc" "$(gl release)" 0
chk "  voxtral back" "$(vox)" active
chk "suppress rc" "$(gl suppress trackA --vram 9200)" 0
chk "  voxtral down" "$(vox)" inactive
chk "stray release (no lease held) rc" "$(gl release)" 0
chk "  voxtral STILL down" "$(vox)" inactive
case "$(grep -c 'start voxtral' "$STUB_STATE/actions")" in 1) ok "  exactly one voxtral start so far" ;; *) bad "  voxtral starts: $(grep -c 'start voxtral' "$STUB_STATE/actions")" ;; esac
chk "unsuppress rc" "$(gl unsuppress trackA)" 0
chk "  voxtral back" "$(vox)" active
teardown

# --------------------------------------------------- 5. expiry is authoritative
banner "5. expired lease / expired suppressor treated as free"
setup
chk "claim ttl 1s rc" "$(gl claim qwen --ttl 1s)" 0
sleep 2
chk "suppress past an expired lease" "$(gl suppress capB --vram 5400)" 0
chk "  expired lock reaped" "$([ -f "$XDG_RUNTIME_DIR/gpu-lease.json" ] && echo yes || echo no)" no
teardown
setup
chk "suppress ttl 1s rc" "$(gl suppress trackA --vram 18000 --ttl 1s)" 0
sleep 2
chk "second suppress past an expired suppressor" "$(gl suppress capB --vram 5400)" 0
chk "  refcount is 1 (expired one reaped)" "$(nsup)" 1
chk "  declared MiB" "$(sumsup)" 5400
teardown
setup
chk "claim ttl 1s rc" "$(gl claim qwen --ttl 1s)" 0
sleep 2
chk "claim by another holder past expiry" "$(gl claim other)" 0
chk "  new holder" "$(jq -r .label "$XDG_RUNTIME_DIR/gpu-lease.json")" other
teardown

# ------------------------------------------------------------- 6. legacy lock
banner "6. legacy single-holder lock file read by the new code"
setup
# byte-for-byte the shape the pre-shared-mode script wrote (2026-09-01 lock)
n=$(date +%s); e=$((n - 60))
jq -n --arg holder "claude@pony" --arg label "chunktrack-full-video" --arg ttl "8h" \
      --argjson claimed_at $((n-28860)) --arg claimed_h "$(date -Is -d "@$((n-28860))")" \
      --argjson expires_at "$e" --arg expires_h "$(date -Is -d "@$e")" \
   '{holder:$holder,label:$label,ttl:$ttl,claimed_at:$claimed_at,claimed_at_h:$claimed_h,expires_at:$expires_at,expires_at_h:$expires_h}' \
   > "$XDG_RUNTIME_DIR/gpu-lease.json"
echo inactive > "$STUB_STATE/voxtral"
out="$("$LEASE" status 2>&1)"
case "$out" in *EXPIRED*) ok "status flags the legacy lock as expired" ;; *) bad "status: $(echo "$out"|head -3)" ;; esac
case "$out" in *chunktrack-full-video*) ok "status reads the legacy label" ;; *) bad "label missing from status" ;; esac
chk "release --expired on a legacy lock" "$(gl release --expired)" 0
chk "  lock removed" "$([ -f "$XDG_RUNTIME_DIR/gpu-lease.json" ] && echo yes || echo no)" no
chk "  voxtral restarted" "$(vox)" active
teardown
setup
# legacy lock with NO expires_at at all (hand-edited / older format)
jq -n --arg holder "claude@pony" --arg label "old" --arg ttl "2h" \
      --argjson claimed_at "$(( $(date +%s) - 10 ))" \
   '{holder:$holder,label:$label,ttl:$ttl,claimed_at:$claimed_at}' \
   > "$XDG_RUNTIME_DIR/gpu-lease.json"
chk "no-expires_at lock still blocks a claim" "$(gl claim other)" 1
chk "  ttl-derived expiry not yet reached" "$([ -f "$XDG_RUNTIME_DIR/gpu-lease.json" ] && echo yes || echo no)" yes
# expires_at is derivable as claimed_at+ttl even without an expires_at field,
# so an early timer must re-arm rather than release this one too
chk "release --expired re-arms (ttl-derived expiry still future)" "$(gl release --expired)" 0
chk "  lock retained" "$([ -f "$XDG_RUNTIME_DIR/gpu-lease.json" ] && echo yes || echo no)" yes
case "$(errtail)" in *"re-arming"*) ok "  re-armed on the derived expiry" ;; *) bad "  err: $(errtail)" ;; esac
teardown
setup
# legacy lock whose ttl HAS elapsed but which carries no expires_at
jq -n --arg holder "claude@pony" --arg label "old2" --arg ttl "1s" \
      --argjson claimed_at "$(( $(date +%s) - 600 ))" \
   '{holder:$holder,label:$label,ttl:$ttl,claimed_at:$claimed_at}' \
   > "$XDG_RUNTIME_DIR/gpu-lease.json"
chk "ttl-derived expiry reaped on claim" "$(gl claim other)" 0
chk "  new holder" "$(jq -r .label "$XDG_RUNTIME_DIR/gpu-lease.json")" other
teardown

# --------------------------------------------------- 7. timer backstop re-arm
banner "7. safety timer that fires early re-arms instead of releasing"
setup
chk "claim ttl 1h rc" "$(gl claim qwen --ttl 1h)" 0
chk "premature release --expired rc" "$(gl release --expired)" 0
case "$(errtail)" in *"re-arming"*) ok "  re-armed, did not release" ;; *) bad "  err: $(errtail)" ;; esac
chk "  lock still held" "$(jq -r .label "$XDG_RUNTIME_DIR/gpu-lease.json")" qwen
chk "  voxtral still down" "$(vox)" inactive
chk "release (real) rc" "$(gl release)" 0
chk "  voxtral back" "$(vox)" active
teardown
setup
chk "suppress ttl 1h rc" "$(gl suppress trackA --vram 9200 --ttl 1h)" 0
chk "premature unsuppress --expired rc" "$(gl unsuppress trackA --expired)" 0
case "$(errtail)" in *"re-arming"*) ok "  re-armed, did not drop" ;; *) bad "  err: $(errtail)" ;; esac
chk "  suppressor still present" "$(nsup)" 1
chk "  voxtral still down" "$(vox)" inactive
teardown

# ------------------------------------------------------------ 8. idleness gate
banner "8. idleness gate applies to suppress, not just claim"
setup
echo 2 > "$STUB_STATE/requests"
chk "suppress refused while ASR busy" "$(gl suppress capB --vram 5400)" 1
case "$(errtail)" in *"in flight"*) ok "  refusal names in-flight requests" ;; *) bad "  err: $(errtail)" ;; esac
chk "  entry rolled back" "$(nsup)" 0
chk "  voxtral untouched" "$(vox)" active
chk "suppress --force overrides" "$(gl suppress capB --vram 5400 --force)" 0
chk "  voxtral stopped" "$(vox)" inactive
chk "  refcount" "$(nsup)" 1
teardown
setup
# a SECOND suppressor must not re-run the idleness gate (voxtral already down)
chk "first suppress" "$(gl suppress trackA --vram 9200)" 0
echo 3 > "$STUB_STATE/requests"     # meaningless while voxtral is down
chk "second suppress unaffected by stale request count" "$(gl suppress capB --vram 5400)" 0
chk "  refcount" "$(nsup)" 2
teardown

# ------------------------------------------------------ 9. idempotence / misc
banner "9. idempotence, refresh, and status"
setup
chk "unsuppress of an absent label" "$(gl unsuppress ghost)" 0
chk "  voxtral untouched (already active)" "$(vox)" active
chk "suppress rc" "$(gl suppress capB --vram 5400)" 0
chk "re-suppress same label refreshes" "$(gl suppress capB --vram 6000)" 0
chk "  still one entry" "$(nsup)" 1
chk "  new size recorded" "$(sumsup)" 6000
chk "double unsuppress rc" "$(gl unsuppress capB)" 0
chk "  second unsuppress rc" "$(gl unsuppress capB)" 0
chk "  refcount" "$(nsup)" 0
out="$("$LEASE" status 2>&1)"
case "$out" in *"shared:  none"*) ok "status shows no suppressors" ;; *) bad "status shared line" ;; esac
case "$out" in *"budget:"*) ok "status shows the budget line" ;; *) bad "status budget line" ;; esac
teardown

# ------------------------------------------- 10. concurrent admission is atomic
banner "10. concurrent suppressors never oversubscribe the budget"
setup                                     # budget 19456 MiB
for i in $(seq 1 10); do
  ( "$LEASE" suppress "t$i" --vram 3000 >/dev/null 2>&1 ) &
done
wait
n="$(nsup)"; sum="$(sumsup)"
chk "granted count" "$n" 6                # floor(19456/3000) = 6
chk "declared total within budget" "$([ "$sum" -le 19456 ] && echo yes || echo no)" yes
chk "  declared total" "$sum" 18000
chk "  voxtral stopped exactly once" "$(grep -c 'stop voxtral' "$STUB_STATE/actions")" 1
for l in $(jq -r '.suppressors|keys[]' "$XDG_RUNTIME_DIR/gpu-lease.shared.json"); do
  "$LEASE" unsuppress "$l" >/dev/null 2>&1
done
chk "all released" "$(nsup)" 0
chk "  voxtral back" "$(vox)" active
chk "  voxtral started exactly once" "$(grep -c 'start voxtral' "$STUB_STATE/actions")" 1
teardown

printf '\n===== %d passed, %d failed =====\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
