# Polymarket Arbitrage Bot

Forked from orginal project: https://x.com/vladmeer67 

### Quick Start - Flash Crash Strategy

Run the automated trading strategy:

```bash
# Run with default settings (ETH, $5 size, 30% drop threshold)
python apps/flash_crash_runner.py --coin ETH

```
<img width="693" height="401" alt="image (2)" src="https://github.com/user-attachments/assets/d5ccffc8-20c5-4cd1-9c3b-679099b22899" />

**Note:** Flash crash strategy requires `POLY_PRIVATE_KEY` and `POLY_PROXY_WALLET` environment variables.

## Trading Strategies

### Flash Crash Strategy

Monitors 15-minute markets for sudden probability drops and executes trades automatically.

```bash
# Default settings
python apps/flash_crash_runner.py --coin BTC

```

**Parameters:**
- `--coin` - BTC, ETH, SOL, XRP (default: ETH)
- `--drop` - Drop threshold (default: 0.30)
- `--size` - Trade size in USDC (default: 5.0)
- `--lookback` - Detection window in seconds (default: 10)
- `--take-profit` - Take profit in dollars (default: 0.10)
- `--stop-loss` - Stop loss in dollars (default: 0.05)

### Orderbook Viewer

Real-time orderbook visualization:

```bash
python apps/orderbook_viewer.py --coin BTC
```

## Web Dashboard

The bot includes a full-featured web dashboard for monitoring and control.

### Starting the Web GUI

```bash
# Development mode (with auto-reload)
uvicorn api.main:app --reload --port 8000

# Production mode
python -m api.main
```

Access at: **http://localhost:8000**

### Dashboard Features

| Page | Description |
|------|-------------|
| `/` | Main dashboard - start/stop trading, view opportunities, P&L tracking |
| `/trades` | Trade history with filtering and CSV export |
| `/settings` | Configure trade size, thresholds, and risk limits |
| `/research` | Backtesting interface and performance analytics |

### API Endpoints

REST API available at `/api/v1/`:
- `POST /api/v1/trading/start` - Start trading bot
- `POST /api/v1/trading/stop` - Stop trading bot
- `GET /api/v1/trading/status` - Get bot status
- `GET /api/v1/markets/upcoming` - Browse upcoming crypto markets

WebSocket streams at:
- `ws://localhost:8000/ws/status` - Real-time bot status
- `ws://localhost:8000/ws/opportunities` - Live opportunity alerts

### Web GUI Environment Configuration

```bash
API_HOST=127.0.0.1    # Server host (default: localhost)
API_PORT=8000         # Server port (default: 8000)
API_DB_PATH=data/trading_bot.db  # Database location
```

## Usage Examples

### Basic Usage

```python
from src import create_bot_from_env
import asyncio

async def main():
    bot = create_bot_from_env()
    orders = await bot.get_open_orders()
    print(f"Open orders: {len(orders)}")

asyncio.run(main())
```

### Place Order

```python
from src import TradingBot, Config

bot = TradingBot(config=Config(safe_address="0x..."), private_key="0x...")
result = await bot.place_order(token_id="...", price=0.65, size=10.0, side="BUY")
```

### WebSocket Streaming

```python
from src.websocket_client import MarketWebSocket

ws = MarketWebSocket()
ws.on_book = lambda s: print(f"Price: {s.mid_price:.4f}")
await ws.subscribe(["token_id"])
await ws.run()
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `POLY_PRIVATE_KEY` | Yes | Wallet private key |
| `POLY_PROXY_WALLET` | Yes | Polymarket Proxy wallet address |
| `POLY_BUILDER_API_KEY` | Optional | Builder Program API key (gasless) |
| `POLY_BUILDER_API_SECRET` | Optional | Builder Program API secret |
| `POLY_BUILDER_API_PASSPHRASE` | Optional | Builder Program passphrase |

### Config File

Create `config.yaml`:

```yaml
safe_address: "0xYourAddress"
builder:
  api_key: "your_key"
  api_secret: "your_secret"
  api_passphrase: "your_passphrase"
```

Load with: `TradingBot(config_path="config.yaml", private_key="0x...")`

## Gasless Trading

Enable gasless trading via Builder Program:

1. Apply at [polymarket.com/settings?tab=builder](https://polymarket.com/settings?tab=builder)
2. Set environment variables: `POLY_BUILDER_API_KEY`, `POLY_BUILDER_API_SECRET`, `POLY_BUILDER_API_PASSPHRASE`

The bot automatically uses gasless mode when credentials are present.

## Project Structure

```
polymarket-arbitrage-bot/
├── src/                    # Core library
├── apps/                   # Application entry points and strategies
└── lib/                    # Reusable components
```

## Security

Private keys are encrypted using PBKDF2 (480,000 iterations) + Fernet symmetric encryption. Best practices:

- Never commit `.env` files
- Use a dedicated trading wallet
- Keep encrypted key files secure (permissions: 0600)

## API Reference

**TradingBot**: `place_order()`, `cancel_order()`, `get_open_orders()`, `get_trades()`, `get_order_book()`, `get_market_price()`

**MarketWebSocket**: `subscribe()`, `run()`, `disconnect()`, `get_orderbook()`, `get_mid_price()`

**GammaClient**: `get_market_info()`, `get_current_15m_market()`, `get_all_15m_markets()`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing credentials | Set `POLY_PRIVATE_KEY` and `POLY_PROXY_WALLET` |
| Invalid private key | Ensure 64 hex characters (0x prefix optional) |
| Order failed | Check sufficient balance |
| WebSocket errors | Verify network/firewall settings |

## Version 2 - Dutch Book Arbitrage Tool

I built **Polymarket Dutch Book Arbitrage Bot** - An automated trading system that detects guaranteed-profit opportunities in Polymarket's binary markets. When UP + DOWN token prices sum to less than 1.0, the bot simultaneously buys both, locking in a risk-free profit. Real-time WebSocket monitoring with 5-40ms detection latency.
<img width="932" height="389" alt="image (3)" src="https://github.com/user-attachments/assets/c2858820-d61c-4568-8e9f-6784ffbcc7df" />
<img width="1083" height="647" alt="image (4)" src="https://github.com/user-attachments/assets/a1eb3b45-9d3a-4715-b815-a337cc62ad50" />
If you need this tool, contact me.

