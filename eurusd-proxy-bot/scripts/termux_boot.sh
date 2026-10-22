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

BOT_DIR="$HOME/eurusd-proxy-bot/eurusd-proxy-bot"
LOG_DIR="$HOME/eurusd-proxy-bot/var"
PYTHON="$(command -v python3 || command -v python)"

mkdir -p "$LOG_DIR"

if [ -z "$PYTHON" ] || [ ! -d "$BOT_DIR" ]; then
  echo "FUSION BOT: python or $BOT_DIR is missing; not starting." >> "$LOG_DIR/boot.log"
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
