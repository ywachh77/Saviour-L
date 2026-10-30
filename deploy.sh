#!/usr/bin/env bash
# Saviour-L Deployment Script for GratisVPS
# Run this on your VPS after cloning the repo.
set -euo pipefail

echo "=== Saviour-L Deployment ==="
echo ""

# --- 1. Install Python 3.11+ and pip ---
echo "[1/5] Installing Python and pip..."
if command -v python3 &>/dev/null && python3 --version | grep -q "3\.1[1-9]"; then
    echo "  Python $(python3 --version) already installed."
else
    sudo apt update -qq
    sudo apt install -y -qq python3.11 python3-pip python3.11-venv curl
    echo "  Python 3.11 installed."
fi

# --- 2. Create virtual environment and install dependencies ---
echo "[2/5] Setting up virtual environment..."
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r eurusd-proxy-bot/requirements.txt -q
echo "  Dependencies installed."

# --- 3. Configure .env ---
echo "[3/5] Configuring environment..."
ENV_FILE="eurusd-proxy-bot/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "  .env file not found. Creating one."
    cat > "$ENV_FILE" << 'ENVEOF'
# --- Twelve Data API ---
TWELVE_DATA_API_KEY=your_twelvedata_key_here

# --- Supabase ---
SUPABASE_URL=https://your_project_ref.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here

# --- Telegram (optional) ---
TELEGRAM_BOT_TOKEN=
TELEGRAM_OWNER_CHAT_ID=

# --- Optional overrides ---
SIGNAL_COOLDOWN_MINUTES=15
MAX_CANDLE_AGE_SECONDS=130
ENVEOF
    echo "  Created $ENV_FILE — EDIT IT NOW with your credentials!"
    echo "  Run: nano $ENV_FILE"
    echo ""
    echo "  Press Enter when done..."
    read -r
else
    echo "  .env file already exists at $ENV_FILE"
    echo "  Make sure your credentials are filled in: nano $ENV_FILE"
fi

chmod 600 "$ENV_FILE"

# --- 4. Start the server ---
echo "[4/5] Starting Saviour Console..."

# Check if already running
if pgrep -f "web_dashboard.py" &>/dev/null; then
    echo "  Server is already running. Stopping old instance..."
    pkill -f "web_dashboard.py" || true
    sleep 2
fi

# Start in background with nohup
nohup python3 eurusd-proxy-bot/scripts/web_dashboard.py \
    > /tmp/saviour-console.log 2>&1 &
SERVER_PID=$!
echo "  Server started (PID: $SERVER_PID)"
echo "  Logs: tail -f /tmp/saviour-console.log"

# Wait for server to be ready
echo "  Waiting for server to start..."
for i in $(seq 1 10); do
    if curl -s http://127.0.0.1:8080/api/status &>/dev/null; then
        echo "  Server is ready at http://127.0.0.1:8080"
        break
    fi
    sleep 1
done

# --- 5. Optional: Install pm2 for process management ---
echo "[5/5] Process management..."
echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Server is running in the background."
echo ""
echo "Commands:"
echo "  tail -f /tmp/saviour-console.log    # View logs"
echo "  pkill -f web_dashboard.py           # Stop server"
echo "  python3 eurusd-proxy-bot/scripts/web_dashboard.py  # Run in foreground"
echo ""
echo "Access the dashboard at: http://YOUR_VPS_IP:8080"
echo ""
echo "To save credentials via the dashboard API:"
echo "  curl -X POST http://127.0.0.1:8080/api/credentials \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"twelvedata_keys\":\"YOUR_KEY\",\"supabase_url\":\"https://...\",\"supabase_service_key\":\"...\"}'"
