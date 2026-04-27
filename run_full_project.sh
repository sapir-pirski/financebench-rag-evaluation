#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TARGET_PYTHON_VERSION="${TARGET_PYTHON_VERSION:-3.9.6}"
VENV_DIR="${VENV_DIR:-.venv}"
KERNEL_NAME="${KERNEL_NAME:-assignment_2-rag-venv}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-Python (.venv assignment_2-rag)}"
NOTEBOOK_TIMEOUT="${NOTEBOOK_TIMEOUT:-21600}"
LOG_DIR="${LOG_DIR:-run_logs}"
RUN_FAISS="${RUN_FAISS:-1}"
RUN_PGVECTOR="${RUN_PGVECTOR:-0}"
SKIP_ENV_CHECK="${SKIP_ENV_CHECK:-0}"
DEFAULT_JUDGE_MODEL="deepseek-ai/DeepSeek-V3.2"

FAISS_NOTEBOOK="rag_faiss/assignment_2_rag_faiss.ipynb"
PGVECTOR_NOTEBOOK="rag_pgvector/assignment_2_rag_pgvector.ipynb"

log() {
  printf '[run_full_project] %s\n' "$*"
}

die() {
  printf '[run_full_project] ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [ -f "$path" ] || die "Missing required file: $path"
}

python_version() {
  local executable="$1"
  "$executable" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || true
}

resolve_executable() {
  local executable="$1"
  if [[ "$executable" = */* ]]; then
    if [ -x "$executable" ]; then
      printf '%s\n' "$executable"
    fi
  else
    command -v "$executable" 2>/dev/null || true
  fi
}

find_target_python() {
  local candidate resolved version
  for candidate in "$PYTHON_BIN" python3.9 python3; do
    resolved="$(resolve_executable "$candidate")"
    if [ -n "$resolved" ]; then
      version="$(python_version "$resolved")"
      if [ "$version" = "$TARGET_PYTHON_VERSION" ]; then
        printf '%s\n' "$resolved"
        return 0
      fi
    fi
  done

  if command -v uv >/dev/null 2>&1; then
    resolved="$(uv python find "$TARGET_PYTHON_VERSION" 2>/dev/null || true)"
    if [ -n "$resolved" ] && [ "$(python_version "$resolved")" = "$TARGET_PYTHON_VERSION" ]; then
      printf '%s\n' "$resolved"
      return 0
    fi
  fi

  return 1
}

select_python_bin() {
  local target_python
  target_python="$(find_target_python || true)"
  if [ -z "$target_python" ] && command -v uv >/dev/null 2>&1; then
    log "Python $TARGET_PYTHON_VERSION not found; installing with uv"
    uv python install "$TARGET_PYTHON_VERSION"
    target_python="$(find_target_python || true)"
  fi

  if [ -z "$target_python" ]; then
    die "Python $TARGET_PYTHON_VERSION is required. Install it manually, install uv, or set PYTHON_BIN to a Python $TARGET_PYTHON_VERSION executable."
  fi

  PYTHON_BIN="$target_python"
  log "Using Python $(python_version "$PYTHON_BIN") at $PYTHON_BIN"
}

venv_python() {
  if [[ "$VENV_DIR" = /* ]]; then
    printf '%s/bin/python' "$VENV_DIR"
  else
    printf '%s/%s/bin/python' "$PROJECT_ROOT" "$VENV_DIR"
  fi
}

venv_jupyter() {
  if [[ "$VENV_DIR" = /* ]]; then
    printf '%s/bin/jupyter' "$VENV_DIR"
  else
    printf '%s/%s/bin/jupyter' "$PROJECT_ROOT" "$VENV_DIR"
  fi
}

env_value() {
  local key="$1"
  if [ ! -f ".env" ]; then
    return 1
  fi
  awk -F= -v key="$key" '
    $0 !~ /^[[:space:]]*#/ && $1 == key {
      sub(/^[^=]*=/, "")
      gsub(/^[[:space:]]+|[[:space:]]+$/, "")
      print
      exit
    }
  ' .env
}

check_env() {
  if [ "$SKIP_ENV_CHECK" = "1" ]; then
    log "Skipping .env validation because SKIP_ENV_CHECK=1"
    return
  fi

  require_file ".env"

  local nebius_api_key
  nebius_api_key="$(env_value NEBIUS_API_KEY || true)"
  case "$nebius_api_key" in
    ""|"your_api_key_here"|"your_nebius_api_key_here")
      die "Set NEBIUS_API_KEY in .env before running the notebooks."
      ;;
  esac

  if [ "$RUN_PGVECTOR" = "1" ]; then
    local pgpassword
    pgpassword="$(env_value PGPASSWORD || true)"
    case "$pgpassword" in
      ""|"your_database_password"|"your_database_password_here")
        die "Set PGPASSWORD in .env before running the pgvector notebook."
        ;;
    esac
    require_file "data/nebius_msp_ca.pem"
  fi
}

configure_assignment_defaults() {
  local configured_judge_model
  configured_judge_model="${NEBIUS_JUDGE_MODEL:-}"
  if [ -z "$configured_judge_model" ]; then
    configured_judge_model="$(env_value NEBIUS_JUDGE_MODEL || true)"
  fi
  export NEBIUS_JUDGE_MODEL="${configured_judge_model:-$DEFAULT_JUDGE_MODEL}"
  log "Using judge model: $NEBIUS_JUDGE_MODEL"
}

create_or_update_venv() {
  if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  else
    log "Using existing virtual environment at $VENV_DIR"
    local existing_version
    existing_version="$("$(venv_python)" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || true)"
    if [ "$existing_version" != "$TARGET_PYTHON_VERSION" ]; then
      die "Existing virtual environment at $VENV_DIR uses Python ${existing_version:-unknown}; expected $TARGET_PYTHON_VERSION. Remove it or set VENV_DIR to a new path."
    fi
  fi

  log "Installing notebook runner dependencies"
  "$(venv_python)" -m pip install --upgrade pip
  "$(venv_python)" -m pip install jupyter nbconvert ipykernel
}

create_local_kernel() {
  local kernel_dir="$PROJECT_ROOT/.jupyter/kernels/$KERNEL_NAME"
  local python_path
  python_path="$(venv_python)"

  mkdir -p "$kernel_dir"
  cat > "$kernel_dir/kernel.json" <<JSON
{
  "argv": [
    "$python_path",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "$KERNEL_DISPLAY_NAME",
  "language": "python"
}
JSON

  export JUPYTER_PATH="$PROJECT_ROOT/.jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
  log "Registered local notebook kernel: $KERNEL_NAME"
}

execute_notebook() {
  local notebook="$1"
  local notebook_name
  notebook_name="$(basename "$notebook")"
  local stem="${notebook_name%.ipynb}"
  local log_file="$LOG_DIR/${stem}.log"

  require_file "$notebook"
  mkdir -p "$LOG_DIR"

  log "Executing $notebook; log: $log_file"
  "$(venv_jupyter)" nbconvert \
    --to notebook \
    --execute "$notebook" \
    --inplace \
    --ExecutePreprocessor.kernel_name="$KERNEL_NAME" \
    --ExecutePreprocessor.timeout="$NOTEBOOK_TIMEOUT" \
    2>&1 | tee "$log_file"
}

main() {
  if [ "$RUN_FAISS" != "1" ] && [ "$RUN_PGVECTOR" != "1" ]; then
    die "Nothing to run. Set RUN_FAISS=1 and/or RUN_PGVECTOR=1."
  fi

  if [ "$RUN_FAISS" = "1" ]; then
    require_file "$FAISS_NOTEBOOK"
  fi
  if [ "$RUN_PGVECTOR" = "1" ]; then
    require_file "$PGVECTOR_NOTEBOOK"
  fi

  check_env
  configure_assignment_defaults
  select_python_bin
  create_or_update_venv
  create_local_kernel

  export PYTHONUNBUFFERED=1
  unset TRANSFORMERS_CACHE
  export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
  export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$PROJECT_ROOT/.cache/torch/sentence_transformers}"
  mkdir -p "$HF_HOME" "$SENTENCE_TRANSFORMERS_HOME"
  export HF_HUB_DISABLE_IMPLICIT_TOKEN="${HF_HUB_DISABLE_IMPLICIT_TOKEN:-1}"
  export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-20}"
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-180}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export ASSIGNMENT_2_API_FAITHFULNESS_WORKERS="${ASSIGNMENT_2_API_FAITHFULNESS_WORKERS:-4}"
  export ASSIGNMENT_2_RAGAS_SCORE_TIMEOUT_SECONDS="${ASSIGNMENT_2_RAGAS_SCORE_TIMEOUT_SECONDS:-45}"
  export ASSIGNMENT_2_RAGAS_MAX_RETRIES="${ASSIGNMENT_2_RAGAS_MAX_RETRIES:-2}"

  if [ "$RUN_FAISS" = "1" ]; then
    execute_notebook "$FAISS_NOTEBOOK"
  else
    log "Skipping FAISS notebook because RUN_FAISS=$RUN_FAISS"
  fi

  if [ "$RUN_PGVECTOR" = "1" ]; then
    execute_notebook "$PGVECTOR_NOTEBOOK"
  else
    log "Skipping pgvector notebook because RUN_PGVECTOR=$RUN_PGVECTOR"
  fi

  log "Done."
  if [ "$RUN_FAISS" = "1" ]; then
    log "FAISS outputs: rag_faiss/outputs/"
  fi
  if [ "$RUN_PGVECTOR" = "1" ]; then
    log "pgvector outputs: rag_pgvector/outputs/"
  else
    log "pgvector notebook was not run. Set RUN_PGVECTOR=1 to run it when PostgreSQL is reachable."
  fi
}

main "$@"
