from dotenv import load_dotenv
load_dotenv()                                    # reads .env

import os, math, time, pandas as pd, numpy as np
from binance import Client
from datetime import datetime, timezone
from datetime import datetime, timedelta, timezone

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import display
import os, matplotlib
matplotlib.use("Agg")           # ← evita errores de display al generar PDF/plots
os.chdir(os.path.dirname(__file__)) 

API_KEY = os.getenv("BINANCE_SUB_KEY")
API_SEC = os.getenv("BINANCE_SUB_SEC")

if not API_KEY or not API_SEC:
    raise RuntimeError("API credentials not found!")

client = Client(API_KEY, API_SEC, tld="com", testnet=False)   # sub-account key

# ▒▒▒ 2. Strategy logic (unchanged from earlier posts) ▒▒▒
def filt9x(a, s, i, fh):
    m2=[0,1,3,6,10,15,21,28,36][i-1]; m3=[0,0,1,4,10,20,35,56,84][i-1]
    m4=[0,0,0,1,5,15,35,70,126][i-1]; m5=[0,0,0,0,1,6,21,56,126][i-1]
    m6=[0,0,0,0,0,1,7,28,84][i-1];  m7=[0,0,0,0,0,0,1,8,36][i-1]
    m8=[0,0,0,0,0,0,0,1,9][i-1];   m9=[0,0,0,0,0,0,0,0,1][i-1]
    x=1-a; f=(a**i)*s
    if len(fh)>=1: f+=i*x*fh[-1]
    if len(fh)>=2: f-=m2*(x**2)*fh[-2]
    if len(fh)>=3: f+=m3*(x**3)*fh[-3]
    if len(fh)>=4: f-=m4*(x**4)*fh[-4]
    if len(fh)>=5: f+=m5*(x**5)*fh[-5]
    if len(fh)>=6: f-=m6*(x**6)*fh[-6]
    if len(fh)>=7: f+=m7*(x**7)*fh[-7]
    if len(fh)>=8: f-=m8*(x**8)*fh[-8]
    if len(fh)>=9: f+=m9*(x**9)*fh[-9]
    return f

def n_pole(series, per, N=4):
    beta=(1-math.cos(4*math.asin(1)/per))/(math.pow(1.414,2/N)-1)
    a=-beta+math.sqrt(beta*beta+2*beta)
    fh, out=[], np.zeros_like(series)
    for idx,s in enumerate(series):
        fn=filt9x(a,s,N,fh)
        out[idx]=fn; fh.append(fn)
        if len(fh)>9: fh.pop(0)
    return out

def stoch_rsi(close, rsi_len=14, st_len=14, k_len=3, d_len=3):
    rs = pd.Series(close).diff().rename("delta")
    up, dn = rs.clip(lower=0), -rs.clip(upper=0)
    roll_up  = up.rolling(rsi_len).mean()
    roll_dn  = dn.rolling(rsi_len).mean()
    rsi      = 100 - 100/(1+roll_up/roll_dn)
    stoch    = (rsi - rsi.rolling(st_len).min()) / \
               (rsi.rolling(st_len).max() - rsi.rolling(st_len).min())
    k = stoch.rolling(k_len).mean()*100
    d = k.rolling(d_len).mean()
    return k.values, d.values

def gaussian_channel_strategy(df,
                               poles=4,
                               per=144,
                               mult=1.414,
                               high_thr=80,
                               low_thr=20):
    src   = df[['high', 'low', 'close']].mean(axis=1).values
    close = df['close'].values

    mid   = n_pole(src, per, N=poles)
    tr    = np.maximum(df['high']-df['low'],
              np.maximum(abs(df['high']-df['close'].shift()),
                         abs(df['low'] -df['close'].shift()))).fillna(0).values
    tr_f  = n_pole(tr, per, N=poles)
    hband = mid + tr_f * mult
    lband = mid - tr_f * mult

    k, d  = stoch_rsi(close)
    gauss_green = mid > np.roll(mid, 1)
    price_above = close > hband
    both_high   = (k > high_thr) & (d > high_thr)
    both_low    = (k < low_thr)  & (d < low_thr)
    k_gt_d      = k > d
    enter_long  = gauss_green & price_above & (both_high | both_low) & k_gt_d
    exit_long   = (close < hband) & np.roll(close >= hband, 1)

    return pd.DataFrame({
        "mid": mid,
        "hband": hband,
        "lband": lband,
        "enter_long": enter_long,
        "exit_long": exit_long
    }, index=df.index)



#Extract data from binance
def fetch_klines(symbol, interval, start_dt, end_dt):
    """
    Stream klines in 1 000-row chunks and return a tidy OHLCV dataframe.
    """
    df_all = []
    while start_dt < end_dt:
        kl = client.get_klines(symbol=symbol,
                               interval=interval,
                               startTime=int(start_dt.timestamp() * 1000),
                               endTime  =int(end_dt.timestamp() * 1000),
                               limit=1000)
        if not kl:
            break

        # 12 columns exactly as Binance returns them
        cols = ("open_time","open","high","low","close","volume",
                "close_time","quote_asset_volume","trades",
                "taker_buy_base","taker_buy_quote","ignore")

        df = pd.DataFrame(kl, columns=cols).astype(float)
        df_all.append(df)

        # advance cursor by the last bar’s close time + 1 ms
        last_close_ms = df["close_time"].iloc[-1]
        start_dt = datetime.fromtimestamp(last_close_ms / 1000 + 0.001,
                                          tz=timezone.utc)

    df = pd.concat(df_all, ignore_index=True)

    # index & trim to OHLCV we actually need
    df["open_time"] = pd.to_datetime(df.open_time, unit='ms', utc=True)
    df.set_index("open_time", inplace=True)
    return df[["open","high","low","close","volume"]]

# ────────────────────────────────────────────────────────────────
def get_realtime_signal(strategy_function,
                        strategy_params,
                        symbol="BTCUSDT",
                        interval="1d",
                        start_date=None):
    end_dt = datetime.now(tz=timezone.utc)
    if start_date is None:
        start_dt = end_dt - timedelta(days=400)
    else:
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)

    df = fetch_klines(symbol, interval, start_dt, end_dt)

    # Run the selected strategy with its own parameters
    sig = strategy_function(df, **strategy_params)
    sig["price"] = df["close"]

    trades, in_pos = [], None
    for ts, row in sig.iterrows():
        if row.enter_long and in_pos is None:
            in_pos = {"enter_position_date": ts, "enter_position_price": row.price}
        elif row.exit_long and in_pos is not None:
            exit_px = row.price
            entry_px = in_pos["enter_position_price"]
            days = (ts - in_pos["enter_position_date"]).days
            trades.append(dict(
                enter_position_date = in_pos["enter_position_date"],
                enter_position_price= entry_px,
                exit_position_date  = ts,
                exit_position_price = exit_px,
                q_days_of_position  = days,
                price_difference_of_position = round(exit_px - entry_px, 2),
                percentage_of_position = round(exit_px / entry_px - 1, 4)
            ))
            in_pos = None

    trades_df = pd.DataFrame(trades).sort_values("enter_position_date")

    last = sig.iloc[-1]
    action = "OPEN_LONG"  if last.enter_long else \
             "CLOSE_LONG" if last.exit_long  else "WAIT"

    return {
        "strategy": strategy_function.__name__.replace("_strategy", "").replace("_", " ").title(),
        "symbol": symbol,
        "interval": interval,
        "signal": action,
        "df": df,
        "signals_df": sig,
        "trades_df": trades_df
    }


def send_strategy_signals_email(signals_list, subject="📈 Strategy Signal Alert"):
    sender_email = os.getenv("EMAIL_SENDER")
    sender_pass  = os.getenv("EMAIL_PASSWORD")
    recipient    = os.getenv("EMAIL_RECIPIENT")
    smtp_server  = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port    = int(os.getenv("SMTP_PORT", 587))

    html = "<table border='1' style='border-collapse:collapse;'>"
    html += "<tr><th>Strategy</th><th>Symbol</th><th>Interval</th><th>Signal</th></tr>"
    for s in signals_list:
        html += f"<tr><td>{s['strategy']}</td><td>{s['symbol']}</td><td>{s['interval']}</td><td>{s['signal']}</td></tr>"
    html += "</table>"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_pass)
            server.sendmail(sender_email, recipient, msg.as_string())
        print("✅ Email sent.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# ▒▒▒ 4. Helper: act on the signal with 100 % of capital ▒▒▒
def act_on_signal(signal, symbol="BTCUSDT", base="BTC", quote="USDT"):
    bal_q  = float(client.get_asset_balance(asset=quote)['free'])
    bal_b  = float(client.get_asset_balance(asset=base )['free'])
    lot    = float(client.get_symbol_info(symbol)['filters'][2]['minQty'])
    price  = float(client.get_symbol_ticker(symbol=symbol)['price'])

    if signal == "OPEN_LONG" and bal_q * price > lot:
        print(f"🏃‍♂️ Buying with {bal_q:.4f} {quote}")
        client.create_order(symbol=symbol, side='BUY', type='MARKET',
                            quoteOrderQty=bal_q)               # 100 % quote
    elif signal == "CLOSE_LONG" and bal_b > lot:
        print(f"🏃‍♂️ Selling {bal_b:.6f} {base}")
        client.create_order(symbol=symbol, side='SELL', type='MARKET',
                            quantity=round(bal_b, 6))
    else:
        print("⭕ No action – either WAIT or insufficient balance.")


def main() -> None:
    """
    Punto de entrada: calcula señales y envía email.
    """
    strategy_map = {
        "Gaussian Channel": {
            "function": gaussian_channel_strategy,
            "params": {
                "poles": 4,
                "per": 144,
                "mult": 1.414,
                "high_thr": 80,
                "low_thr": 20
            }
        }
    }
    symbols   = ["BTCUSDT"]
    intervals = ["1d"]
    all_signals = []

    for strategy_name, strat in strategy_map.items():
        for symbol in symbols:
            for interval in intervals:
                try:
                    result = get_realtime_signal(
                        strategy_function = strat["function"],
                        strategy_params   = strat["params"],
                        symbol  = symbol,
                        interval= interval,
                        start_date = "2018-01-01"
                    )
                    all_signals.append({
                        "strategy": strategy_name,
                        "symbol": symbol,
                        "interval": interval,
                        "signal": result["signal"]
                    })
                except Exception as e:
                    print(f"⚠️ Error en {strategy_name} • {symbol}-{interval}: {e}")

    if all_signals:
        send_strategy_signals_email(all_signals)

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()