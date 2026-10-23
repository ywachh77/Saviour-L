#!/usr/bin/env bash
# Create a PRIVATE GitHub repo for the bot and push this folder as its root.
#
#   ./scripts/publish_repo.sh                 # creates ywachh77/eurusd-proxy-bot
#   ./scripts/publish_repo.sh my-bot-name     # or a name you choose
#   ./scripts/publish_repo.sh --dry-run       # show what would happen
#
# Run this from YOUR machine, where `gh` is logged in with your personal
# account. The GitHub App token used inside an agent sandbox cannot create
# repositories, so this step has to be yours — but it is one command.
#
# Nothing secret is transmitted: the repo is created from local files only, and
# .env is git-ignored so it is never part of the push.

set -euo pipefail

DRY_RUN=0
NAME="eurusd-proxy-bot"
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) NAME="$arg" ;;
  esac
done

BOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v gh >/dev/null 2>&1; then
  echo "gh (GitHub CLI) is not installed." >&2
  echo "  Ubuntu:  sudo apt install gh" >&2
  echo "  macOS:   brew install gh" >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "gh is not logged in. Run:  gh auth login" >&2
  exit 1
fi

OWNER="$(gh api user --jq '.login' 2>/dev/null || echo "")"
if [ -z "$OWNER" ]; then
  echo "Could not read your GitHub username." >&2
  exit 1
fi

if gh repo view "$OWNER/$NAME" >/dev/null 2>&1; then
  echo "Repo $OWNER/$NAME already exists." >&2
  echo "To push this folder into it anyway:" >&2
  echo "  cd '$BOT_DIR' && git init && git add . && git commit -m 'FUSION BOT'" >&2
  echo "  git remote add origin https://github.com/$OWNER/$NAME.git" >&2
  echo "  git branch -M main && git push -u origin main" >&2
  exit 1
fi

if [ ! -f "$BOT_DIR/.env" ]; then
  echo "Note: no .env in $BOT_DIR — that is expected. It is git-ignored on purpose." >&2
fi

echo "About to create a PRIVATE repo: https://github.com/$OWNER/$NAME"
echo "  source : $BOT_DIR"
echo "  private: yes"
echo

if [ "$DRY_RUN" -eq 1 ]; then
  echo "--dry-run: stopping here, nothing was created."
  exit 0
fi

read -r -p "Continue? [y/N] " answer
case "$answer" in
  [yY]*) ;;
  *) echo "Cancelled."; exit 0 ;;
esac

gh repo create "$OWNER/$NAME" --private --source="$BOT_DIR" --remote=origin --push

echo
echo "Done: https://github.com/$OWNER/$NAME"
echo
echo "Next: upload your .env values as Actions secrets (never printed, never stored in git):"
echo "  ./scripts/push_secrets.sh $OWNER/$NAME"
