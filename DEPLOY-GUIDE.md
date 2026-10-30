# Saviour-L — GratisVPS Deployment Guide

## Step 1: Sign Up for GratisVPS

1. Go to [gratisvps.net](https://gratisvps.net/)
2. Click "Get your free VPS"
3. Register with your email (no credit card needed)
4. Choose:
   - OS: **Ubuntu 24.04**
   - Region: pick the one closest to you (US, Germany, or Singapore)
5. Wait ~1 minute for the server to be created
6. You'll get an email with the root password and SSH details

## Step 2: Connect to Your VPS

```bash
ssh root@YOUR_VPS_IP
```

When prompted for password, enter the one from the email.

## Step 3: Install Git and Clone the Repo

```bash
apt update && apt install -y git
git clone https://github.com/ywachh77/Saviour-L.git
cd Saviour-L
```

## Step 4: Run the Deploy Script

```bash
chmod +x deploy.sh
./deploy.sh
```

The script will:
1. Install Python 3.11 and pip
2. Create a virtual environment
3. Install dependencies (requests, python-dotenv)
4. Create a `.env` file for credentials
5. Start the server on port 8080

## Step 5: Configure Credentials

The script will pause and ask you to edit `.env`. Fill in:

```
TWELVE_DATA_API_KEY=f6dfb506b93c425c83676a0a3942affb
SUPABASE_URL=https://idudlcxkihqnvecwqqoj.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlkdWRsY3hraWhxbnZlY3dxcW9qIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc5MjY3MjI1MywiZXhwIjoyMTA4MjQ4MjUzfQ.uMafh4b2NCSqcFe5796i18h9pLthNCVNKnpAtMSojYs
```

Or save them via the API:
```bash
curl -X POST http://127.0.0.1:8080/api/credentials \
  -H 'Content-Type: application/json' \
  -d '{"twelvedata_keys":"f6dfb506b93c425c83676a0a3942affb","supabase_url":"https://idudlcxkihqnvecwqqoj.supabase.co","supabase_service_key":"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlkdWRsY3hraWhxbnZlY3dxcW9qIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc5MjY3MjI1MywiZXhwIjoyMTA4MjQ4MjUzfQ.uMafh4b2NCSqcFe5796i18h9pLthNCVNKnpAtMSojYs"}'
```

## Step 6: Access the Dashboard

Open your browser and go to:
```
http://YOUR_VPS_IP:8080
```

Click **Generate Signal** to see live signals.

## Step 7: Keep It Running

The server runs in the background via `nohup`. To:
- View logs: `tail -f /tmp/saviour-console.log`
- Stop: `pkill -f web_dashboard.py`
- Restart: `python3 eurusd-proxy-bot/scripts/web_dashboard.py &`

## What Changes on the VPS vs Sandbox

| Feature | Sandbox | GratisVPS |
|---------|---------|-----------|
| Twelve Data API | Blocked (TLS drops) | ✅ Reachable |
| Feed Mode | PAPER_SIMULATION | ✅ LIVE |
| Candle Data | Synthetic (paper) | ✅ Real M1 candles |
| Signal Prices | Generated | ✅ Real market prices |
| 24/7 Operation | Resets between turns | ✅ Persistent |
| Settlement | Can't settle (no exit candles) | ✅ Auto-settles every 15s |
| Telegram | Cloud-synced only | ✅ Direct Telegram API |

## Port 8080 Requirement

GratisVPS gives you a shared IPv4. Port 8080 should be open by default. If you need to open it:

```bash
iptables -A INPUT -p tcp --dport 8080 -j ACCEPT
```

## Troubleshooting

**Server won't start:**
```bash
tail -f /tmp/saviour-console.log
```

**Port 8080 in use:**
```bash
pkill -f web_dashboard.py
sleep 2
python3 eurusd-proxy-bot/scripts/web_dashboard.py &
```

**Can't reach Twelve Data:**
```bash
curl -v https://api.twelvedata.com/time_series?symbol=EUR/USD&interval=1min&apikey=YOUR_KEY
```

**Reset everything:**
```bash
pkill -f web_dashboard.py
rm -f eurusd-proxy-bot/.env
rm -rf .venv
./deploy.sh
```
