#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLAMA_MAIN="$SCRIPT_DIR/cmake-out/examples/models/llama/llama_main"
if [ ! -f "$LLAMA_MAIN" ]; then
    echo "Error: llama_main not found. Run build_llama_runner.sh first." >&2
    exit 1
fi

usage() {
    cat <<EOF
Usage: $(basename "$0") --model <model.pte> --tokenizer <tokenizer.model> [options]

Required:
  --model       Path to exported .pte model file
  --tokenizer   Path to tokenizer.model file

Optional:
  --prompt              Input prompt (default: "Hello, world!")
  --seq_len             Max sequence length including prompt (default: 128)
  --temperature         Sampling temperature, 0 for greedy (default: 0)
  --method_name         Decode method name (default: "forward")
  --prefill_method_name Separate prefill method name (default: "")

Example (single method):
  $(basename "$0") \\
    --model llama3_2.pte \\
    --tokenizer ~/.llama/checkpoints/Llama3.2-1B/tokenizer.model \\
    --prompt "What is the capital of France?"

Example (separate prefill/decode):
  $(basename "$0") \\
    --model llama3_prefill_decode.pte \\
    --tokenizer ~/.llama/checkpoints/Llama3.2-1B/tokenizer.model \\
    --prompt "What is the capital of France?" \\
    --method_name decode --prefill_method_name kv_forward
EOF
    exit 1
}

MODEL=""
TOKENIZER=""
PROMPT="Hello, world!"
SEQ_LEN=128
TEMPERATURE=0
METHOD_NAME="forward"
PREFILL_METHOD_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)               MODEL="$2"; shift 2 ;;
        --tokenizer)           TOKENIZER="$2"; shift 2 ;;
        --prompt)              PROMPT="$2"; shift 2 ;;
        --seq_len)             SEQ_LEN="$2"; shift 2 ;;
        --temperature)         TEMPERATURE="$2"; shift 2 ;;
        --method_name)         METHOD_NAME="$2"; shift 2 ;;
        --prefill_method_name) PREFILL_METHOD_NAME="$2"; shift 2 ;;
        -h|--help)             usage ;;
        *)                     echo "Unknown option: $1" >&2; usage ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$TOKENIZER" ]; then
    echo "Error: --model and --tokenizer are required." >&2
    usage
fi

ARGS=(
    --model_path="$MODEL"
    --tokenizer_path="$TOKENIZER"
    --prompt="$PROMPT"
    --seq_len="$SEQ_LEN"
    --temperature="$TEMPERATURE"
    --method_name="$METHOD_NAME"
)
if [ -n "$PREFILL_METHOD_NAME" ]; then
    ARGS+=(--prefill_method_name="$PREFILL_METHOD_NAME")
fi

"$LLAMA_MAIN" "${ARGS[@]}"
