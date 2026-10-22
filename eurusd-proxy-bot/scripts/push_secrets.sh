#!/usr/bin/env bash
# Copy the secret values from your local .env into GitHub Actions secrets.
#
#   ./scripts/push_secrets.sh                    # uses the repo this folder came from
#   ./scripts/push_secrets.sh owner/repo         # or name one explicitly
#   ./scripts/push_secrets.sh --dry-run          # list what would be uploaded
#
# Values are read from .env, piped straight into `gh secret set`, and never
# printed, echoed, or written anywhere else. Config that is not a secret
# (PAIR, SIGNAL_HORIZON_MINUTES, ...) is deliberately left out.
#
# This exists so you never have to paste a key into a chat window.

set -euo pipefail

BOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$BOT_DIR/.env"

DRY_RUN=0
REPO=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) REPO="$arg" ;;
  esac
done

# The names that are genuinely secret. Everything else stays local.
SECRET_NAMES=(
  TELEGRAM_BOT_TOKEN
  TELEGRAM_OWNER_CHAT_ID
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY
  TWELVE_DATA_API_KEY
)
for i in $(seq 1 30); do
  SECRET_NAMES+=("TWELVE_DATA_KEY_$i")
done

if [ ! -f "$ENV_FILE" ]; then
  echo "No .env at $ENV_FILE" >&2
  echo "Run:  cp .env.example .env   then fill it in." >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "gh (GitHub CLI) is not installed." >&2
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "gh is not logged in. Run:  gh auth login" >&2
  exit 1
fi

if [ -z "$REPO" ]; then
  REPO="$(git -C "$BOT_DIR" remote get-url origin 2>/dev/null || echo "")"
  REPO="${REPO#https://github.com/}"
  REPO="${REPO#git@github.com:}"
  REPO="${REPO%.git}"
fi
if [ -z "$REPO" ]; then
  echo "Cannot work out which repo to use." >&2
  echo "Run:  ./scripts/push_secrets.sh owner/repo" >&2
  exit 1
fi

if ! gh repo view "$REPO" >/dev/null 2>&1; then
  echo "Cannot see repo $REPO (it may not exist, or you may lack access)." >&2
  exit 1
fi

get_env_value() {
  local key="$1"
  sed -n "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*//p" "$ENV_FILE" | tail -1 | tr -d '"' | tr -d "'" | sed 's/[[:space:]]*$//'
}

uploaded=0
for key in "${SECRET_NAMES[@]}"; do
  value="$(get_env_value "$key")"
  if [ -z "$value" ]; then
    continue
  fi
  # Skip obvious placeholders so a template value never overwrites a real one.
  case "$value" in
    your-*|YOUR-*|changeme|CHANGE_ME|x|XX*) echo "  skip  $key (looks like a placeholder)"; continue ;;
  esac
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  would set  $key  (${#value} chars, not shown)"
  else
    printf '%s' "$value" | gh secret set "$key" --repo "$REPO" >/dev/null
    echo "  set   $key  (${#value} chars, not shown)"
  fi
  uploaded=$((uploaded + 1))
done

echo
if [ "$uploaded" -eq 0 ]; then
  echo "No secret values found in $ENV_FILE."
  echo "Fill in TELEGRAM_BOT_TOKEN, TELEGRAM_OWNER_CHAT_ID, SUPABASE_URL,"
  echo "SUPABASE_SERVICE_ROLE_KEY and TWELVE_DATA_API_KEY, then run this again."
  exit 1
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "--dry-run: $uploaded value(s) would be uploaded to $REPO. Nothing was sent."
else
  echo "$uploaded secret(s) uploaded to $REPO."
  echo "Verify (names only, values are write-only):"
  echo "  gh secret list --repo $REPO"
fi
