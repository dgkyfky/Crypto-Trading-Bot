# Crypto-Trading-Bot
Crypto trading bot automatize using Gaussian Channel and Stochastic RSI strategy with BTC. Scalable to add multiple strategies and multiple crypto assets. Backtesting to analyze how the model works and in wich crypto asset is working better.

Gaussian Channel Strategy (Stoch RSI).ipynb --> All the code is in here, with the backtesting and to test the strategy in different crypto assets.
Automatizacion folder: using the .py file and the .sh file I've automatize it using BASH and Cron (localy) to run every day at 19hs

Next steps: 
- add information to the email to know how the model is performing, the backtesting and the previously alerts (now it's just the alert)
- Test new strategies and with smaller window time: the Gaussian Channel and Stoch RSI strategy performs well for BTC in a 1DAY chart, almost 1700% of profit for the model since 2018 -backtesting- compared to 700% of profit if holding BTC since 2018. Looking for models using 1H or 4H chart (more volatility, more risk).
