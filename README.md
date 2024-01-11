# ETFstreategy_with_xgboost, price deviation, and dual momentum
- Time Period : 2019-2021
- Universe : listed ETF in Korean stock market
- Data : daily OHLCV, monthly ETF Portfolio Deposit File, sector data, daily macro data, daily ETF price deviation data
- Strategy concept
1) Using XGBoost and labeled macro data, predict the weekly direction of the market
2) When bull market is predicted, using dual-momentum as a score of the stocks
3) When bear market is predicted, using Canary ETFs, which are not sensitive to the movement of the market (defensive etfs), underlying on DAA concept
4) Otherwise, using price_deviation of ETFs underlying on the concept that LP would buy or sell to control its price suitable to its underlying asset
5) To lower turnover, set a threshold to weight changes not to rebalance too finely
