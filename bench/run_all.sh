#!/bin/bash
# Part 9 full benchmark: engine, tuned HF and vLLM; fixed and mixed lengths; rates past saturation; the burst.
#   bash bench/run_all.sh            full run (~2.5 h) -> results/final/
#   QUICK=1 bash bench/run_all.sh    a short run of everything (~10 min) -> results/scratch/run_all_quick/
# Lock the GPU clock first. Re-running skips results that already exist, so an interrupted run can resume.
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate

QUICK=${QUICK:-0}
VLLM=${VLLM:-$HOME/venvs/vllm/bin/vllm}     # vLLM lives in its own venv: it pins its own torch
LOCK_MHZ=${LOCK_MHZ:-1402}
MODEL=Qwen/Qwen2.5-0.5B-Instruct
FIXED=bench/prompts/synthetic_200.jsonl     # 200-token prompts, 128 tokens each
MIXED=bench/prompts/mixed_200.jsonl         # same prompts, max_tokens 16-505

if [ "$QUICK" = 1 ]; then
  OUT=results/scratch/run_all_quick; RUNS=1; SOAK=50; BURST_N=50
  ENGINE_RATES="3 24"; HF_RATES="3 14"; VLLM_RATES="3 24"
else
  OUT=results/final; RUNS=3; SOAK=500; BURST_N=300
  ENGINE_RATES="1 3 6 10 14 18 24"          # engine saturates ~14-15 req/s on mixed (sat_probe)
  HF_RATES="1 3 6 10 14"                    # HF saturates by 10; past that it only queues
  VLLM_RATES="1 3 6 10 14 18 24"
fi

nreq() {  # ~40 s of arrivals per run, at least 100 requests
  if [ "$QUICK" = 1 ]; then echo 30; else awk -v r=$1 'BEGIN { n = int(r * 40); print (n > 100 ? n : 100) }'; fi
}

clk=$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits | head -1)
if [ "$clk" != "$LOCK_MHZ" ]; then
  echo "GPU clock is $clk MHz, not $LOCK_MHZ: lock it first (or set LOCK_MHZ)"; exit 1
fi

mkdir -p $OUT/logs $OUT/soak
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv -l 1 >> $OUT/logs/gpu_log.csv &
SMI=$!
{ git rev-parse HEAD; git status --short; } > $OUT/logs/commit.txt     # the commit, plus any uncommitted changes
"$VLLM" --version > $OUT/logs/vllm_version.txt 2>&1     # vLLM's /health is empty, so record it here

SRV=""; PORT=""
stop() {
  [ -z "$SRV" ] && return
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; SRV=""
  for i in $(seq 60); do     # vLLM's EngineCore child can outlive the parent for a few seconds
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 500 ] && return
    sleep 1
  done
  echo "  GPU memory still in use after stop"
}
trap 'stop; kill $SMI 2>/dev/null' EXIT

start() {  # $1 = name for logs, $2 = port, rest = command
  local name=$1; PORT=$2; shift 2
  "$@" > $OUT/logs/server_$name.log 2>&1 &
  SRV=$!
  for i in $(seq 300); do
    curl -sf localhost:$PORT/health >/dev/null 2>&1 && { echo "[$name] up after ${i}s, $(date +%T)"; return 0; }
    kill -0 $SRV 2>/dev/null || break
    sleep 1
  done
  echo "[$name] failed to start, skipping it"; tail -5 $OUT/logs/server_$name.log; stop; return 1
}

soak() {  # warm the GPU to steady temperature after each start; always runs, even on resume
  echo "  $(date +%T) soak"
  python -m bench.loadgen --url http://localhost:$PORT/v1/completions --rate 3 --num-requests $SOAK \
    --label soak --output $OUT/soak/$1.json 2>&1 | grep -E "num_errors|Error"
}

lg() {  # $1 = result name, rest = loadgen args
  local name=$1 f=$OUT/$1.json; shift
  [ -f $f ] && { echo "  skip $name (exists)"; return; }
  echo "  $(date +%T) $name"
  python -m bench.loadgen --url http://localhost:$PORT/v1/completions --output $f "$@" 2>&1 \
    | grep -E "num_errors|throughput|Error"
}

prompts_for() { [ $1 = mixed ] && echo $MIXED || echo $FIXED; }

rate_runs() {  # $1 = server, $2 = rates
  for wl in fixed mixed; do for r in $2; do for n in $(seq $RUNS); do
    lg ${1}_${wl}_rate${r}_run$n --rate $r --num-requests $(nreq $r) --prompts $(prompts_for $wl) \
      --label "$1 $wl rate $r"
  done; done; done
}

burst_runs() {  # $1 = server
  for wl in fixed mixed; do for n in $(seq $RUNS); do
    lg ${1}_${wl}_burst_run$n --rate 1000 --num-requests $BURST_N --prompts $(prompts_for $wl) \
      --label "$1 $wl burst"
  done; done
}

echo "start $(date +%T), $OUT"

start engine_256 8000 env ENGINE_MAX_NUM_SEQS=256 uvicorn server.app:app --port 8000 \
  && { soak engine_256; rate_runs engine "$ENGINE_RATES"; stop; }
start engine_192 8000 env ENGINE_MAX_NUM_SEQS=192 uvicorn server.app:app --port 8000 \
  && { soak engine_192; burst_runs engine; stop; }

# tuned HF (b60ae44): batch 300; short window for steady traffic, 0.5 s to catch the whole burst
start hf_w20 8001 env HF_MAX_BATCH=300 HF_WINDOW=0.02 uvicorn bench.baselines.hf_server:app --port 8001 \
  && { soak hf_w20; rate_runs hf "$HF_RATES"; stop; }
start hf_w500 8001 env HF_MAX_BATCH=300 HF_WINDOW=0.5 uvicorn bench.baselines.hf_server:app --port 8001 \
  && { soak hf_w500; burst_runs hf; stop; }

start vllm 8002 "$VLLM" serve $MODEL --port 8002 --gpu-memory-utilization 0.8 --max-model-len 2048 \
  && { soak vllm; rate_runs vllm "$VLLM_RATES"; burst_runs vllm; stop; }

echo "done $(date +%T)"