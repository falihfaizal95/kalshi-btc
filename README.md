# Kalshi BTC Scanner Bot

Scans Kalshi's Bitcoin price markets, estimates the probability each market
resolves YES using a log-normal model (Deribit DVOL implied volatility) blended
with an XGBoost classifier, and flags markets where the model disagrees with
the market price by more than a configurable edge threshold. Bets are sized
with fractional Kelly. Optionally places limit orders automatically.

## How it works

1. **Markets** — pulls open markets from Kalshi's BTC series (`KXBTCD`, `KXBTC`)
   using the public API, including above/below and range ("between") markets.
2. **Data** — BTC OHLCV from Binance, 30-day implied volatility from Deribit
   (DVOL, with ATM-option fallback), and the Crypto Fear & Greed index.
3. **Model** — log-normal probability of finishing above/below/between strikes,
   blended 40/60 with an XGBoost model trained on technical features
   (RSI, MACD, ATR, Bollinger, ADX, EMAs, volume, momentum, time features).
4. **Edge & sizing** — edge = model probability − Kalshi mid price. Bets sized
   by fractional Kelly against the ask price, capped per-bet.
5. **Output** — ranked table in the terminal, alerts appended to
   `logs/alerts.csv`, trades to `logs/trades.csv`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### API credentials (optional)

Scanning is **read-only and needs no account**. To check balances or place
orders, generate an API key at kalshi.com → Account & Security → API Keys,
save the private key file, and set in `.env`:

```
KALSHI_API_KEY_ID=<your key id>
KALSHI_PRIVATE_KEY_PATH=~/.kalshi/kalshi-key.pem
```

Requests are signed with RSA-PSS (SHA-256) per Kalshi's API-key auth scheme.

## Usage

```bash
python main.py                  # train model if needed, then scan hourly
python main.py --backtest-only  # run backtest + train model, then exit
python main.py --no-train       # skip training, scan immediately
```

### Trading modes

- **Alerts only** (`PAPER_TRADE=false`, `AUTO_TRADE=false`) — just prints/logs.
- **Paper trading** (`PAPER_TRADE=true`, the default) — simulates fills at the
  ask against a virtual bankroll, settles each position against the actual BTC
  outcome, and tracks running P&L in `tracking/paper_trades.csv`. No money or
  account required. This builds the real-outcome track record the strategy
  learns from.
- **Live trading** (`AUTO_TRADE=true`) — places real Kalshi orders. Requires
  API credentials and takes precedence over paper trading.

The paper account compounds across runs (the CSV is committed to git). Check
its status anytime:

```bash
python -c "import config as c; from paper.account import PaperAccount; \
print(PaperAccount(c.PAPER_TRADES_CSV, c.PAPER_STARTING_BANKROLL).summary())"
```

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `BANKROLL` | 500 | Bankroll in USD used for Kelly sizing |
| `KELLY_FRACTION` | 0.5 | Fraction of full Kelly to bet |
| `MAX_BET_PCT` | 0.05 | Max single bet as a fraction of bankroll |
| `EDGE_THRESHOLD` | 0.05 | Minimum |edge| to alert/trade |
| `AUTO_TRADE` | false | Place orders automatically |
| `KALSHI_DEMO` | false | Use Kalshi's demo environment |

## Backtest caveat

Historical Kalshi order books aren't available, so the backtest simulates
market prices as (lognormal probability + noise). Backtest P&L therefore
validates the pipeline and bet-sizing math — it is **not** evidence of real
trading edge. The XGBoost training itself uses real price outcomes and is
unaffected by this.

## Disclaimer

For research/education. Prediction-market trading involves real financial
risk; nothing here is financial advice. Keep `AUTO_TRADE=false` until you have
verified behavior yourself.
