#!/data/data/com.termux/files/usr/bin/bash
# FUSION BOT — Termux auto-start (optional, 24/7 convenience)
#
# Install:
#   mkdir -p ~/.termux/boot
#   cp scripts/termux_boot.sh ~/.termux/boot/start-fusion-bot.sh
#   chmod +x ~/.termux/boot/start-fusion-bot.sh
#   termux-setup-storage        # optional, lets Termux keep running
#
# It restarts itself if the bot exits, so a crash at 3am does not cost you the
# rest of the night. The bot's own job slots make a restart safe: any minute
# already processed is skipped, never processed twice.

# publish_repo.sh pushes the bot folder as the ROOT of the new repo, so a clone
# lands at ~/eurusd-proxy-bot with scripts/ directly inside it. Older layouts
# nested it one level deeper; both are accepted so this works either way, and
# FUSION_BOT_DIR overrides everything.
BOT_DIR="${FUSION_BOT_DIR:-$HOME/eurusd-proxy-bot}"
if [ ! -f "$BOT_DIR/scripts/run_bot.py" ] && [ -f "$BOT_DIR/eurusd-proxy-bot/scripts/run_bot.py" ]; then
  BOT_DIR="$BOT_DIR/eurusd-proxy-bot"
fi
LOG_DIR="${FUSION_BOT_LOG:-$HOME/eurusd-proxy-bot/var}"
PYTHON="$(command -v python3 || command -v python)"

mkdir -p "$LOG_DIR"

if [ -z "$PYTHON" ] || [ ! -f "$BOT_DIR/scripts/run_bot.py" ]; then
  echo "FUSION BOT: python or $BOT_DIR/scripts/run_bot.py is missing; not starting." >> "$LOG_DIR/boot.log"
  echo "  set FUSION_BOT_DIR if you cloned somewhere else." >> "$LOG_DIR/boot.log"
  exit 1
fi

termux-wake-lock 2>/dev/null

cd "$BOT_DIR" || exit 1

while true; do
  echo "$(date -u +%FT%TZ) starting FUSION BOT" >> "$LOG_DIR/boot.log"
  "$PYTHON" scripts/run_bot.py >> "$LOG_DIR/bot.log" 2>&1
  echo "$(date -u +%FT%TZ) FUSION BOT exited ($?); restarting in 15s" >> "$LOG_DIR/boot.log"
  sleep 15
done
