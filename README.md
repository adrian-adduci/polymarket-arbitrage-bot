# 🎯 BNB Sniper Bot — Pancake and Four.meme

Sniper bots buy new DEX tokens instantly, crucial for BNB Chain where meme coins pump 20x in hours. Built-in filters target high-potential assets. Example: $HORIA trade turned 2 BNB into 9.4 BNB
<img width="720" height="335" alt="image" src="https://github.com/user-attachments/assets/0dafeedb-3f5e-4283-a483-f7f85171e509" />

## Features

- Real-time token detection from four.meme factory
- Mempool monitoring for 0-block sniping
- Optional MEV strategies (front-run / back-run)
- Instant auto-buy with retry & exponential backoff
- Dynamic gas pricing and gas limit controls
- Configurable slippage protection
- Structured logging (Winston) with file rotation

## Minimal .env (examples)

BSC_RPC_URL=https://bsc-dataseed.binance.org/
BSC_WSS_URL=wss://bsc-dataseed.binance.org/

WALLET=0x...
BUY_AMOUNT=0.01
SLIPPAGE_BPS=100
GAS_LIMIT=500000

ENABLE_FRONTRUN=true
ENABLE_BACKRUN=true

## Security & Risk Notice

- This software is provided for educational purposes. Use at your own risk.
- Never commit private keys; keep .env private
- Test thoroughly on testnet before mainnet use
- Failed transactions consume gas — financial loss possible
- New tokens can be malicious (honeypots, rug pulls)

## Best Practices / Performance Tips

- Use premium RPC (QuickNode, Ankr) and WSS for mempool access
- Start with small buy amounts on testnet
- Monitor gas strategy and MEV behavior in controlled tests
- Use multiple wallets to manage nonce and concurrency

## How it works
https://x.com/vladmeer67/status/1978055022459146731

## Contact

### Telegram: [Vladmeer](https://t.me/vladmeer67)    
https://t.me/vladmeer67

### Twitter: [Vladmeer](https://x.com/vladmeer67)   
https://x.com/vladmeer67
