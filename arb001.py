from algo_deploy.strategy_module import LiveStrategy, Indicator, extract_ochlv, Indicators_2
from typing import Optional, Dict
import pandas as pd
import numpy as np
import signal
import time

class ARB001(LiveStrategy):
    """
    ARB001: Arbitrage Strategy - v001 - basket arbitrage strategy.
    """

    def __init__(self, strategy_name: str, params: Dict, account_id: str, exchange: str, base: str, quote: str,
                 category: str = 'futures', demo: bool = True):
        super().__init__(strategy_name, account_id, exchange, base, quote, category, demo)
        self.add_params(params)
        self.stop_flag = False

        # Setup signal handler once on init
        signal.signal(signal.SIGINT, self._signal_handler)

        self.init_balance: float = 0
        self.placed_orders: Dict[int, str] = {}  # ts -> order_ids

    def add_params(self, params: dict):
        """Update strategy parameters."""
        # --- Core strategy logic ---
        self.symbols = params.get('symbols', ['SAND_USDT', 'MANA_USDT', 'DOGE_USDT', 'SHIB1000_USDT', 'BTC_USDT', 'ETH_USDT'])
        self.window = params.get('window', 30)
        self.z_threshold = params.get('z_threshold', 1)
        self.value_entry = params.get('value_entry', 10)
        self.stop_loss_threshold = params.get('stop_loss_threshold', 0.2)

        # --- Execution / Order parameters ---
        self.max_position = params.get('max_position', 10)
        self.refresh_order = params.get('refresh_order', 60)

        # --- Time settings ---
        self.interval = params.get('interval', '1m')
        self.sleep_time = params.get('sleep_time', 60)

    def _extract_symbol(self, symbol: str):
        """Extract symbol from base and quote."""
        if symbol not in self.symbols:
            raise ValueError(f"{symbol} is not supported.")
        return symbol.split('_')

    def _extract_all_candles(self):
        """Extract all candles for all symbols."""
        result = {}
        for symbol in self.symbols:
            base, quote = self._extract_symbol(symbol)
            candles = self.get_candles(base, quote, interval=self.interval, use_redis=False)
            o, c, h, l, v = extract_ochlv(candles)
            result[symbol] = {'o': o, 'c': c, 'h': h, 'l': l, 'v': v}
        return result

    def _price_rets_df(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute aligned price and returns DataFrame for all symbols."""
        price = pd.DataFrame()
        candles_dict = self._extract_all_candles()

        for symbol, candle_data in candles_dict.items():
            c = candle_data['c']
            if isinstance(c, (list, np.ndarray)):
                c = pd.Series(c)
            price[symbol] = c

        # Ensure no all-NaN columns
        price = price.dropna(how='all')

        # Compute log returns
        rets = np.log(price).diff().fillna(0)
        return price, rets

    def overall_signal(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate entry signal based on Z-score and Kalman beta estimation.
        """
        window = self.window
        curr_signal = np.zeros(len(self.symbols))
        curr_vola = np.zeros(len(self.symbols))
        
        try:
            _, rets = self._price_rets_df()
            if rets.shape[0] < self.window:
                self.logger.info("Not enough data for signals.")
                return curr_signal, curr_vola

            stds = rets.std(axis=0).replace(0, 1e-6)
            rets_norm = rets / stds
            weights = Indicators_2.pca_basket_weights(rets_norm.values)
            basket_ret = Indicators_2.basket_returns(rets_norm.values, weights)

            if len(basket_ret) < window:
                window = len(basket_ret)

            rets_norm = rets_norm[-window:].values
            basket_ret = basket_ret[-window:]

            _, _, signals = Indicators_2.compute_all_signals(
                rets_norm, basket_ret, window=window,
                threshold=self.z_threshold, adj_factors=0.001
            )

            curr_signal = signals[-1]
            for i in range(len(rets.columns)):
                curr_vola[i] = Indicator.rolling_std(rets.iloc[:, i].values, 30)[-1]

            self.logger.info(f"Signals: {curr_signal} | Vol: {np.round(curr_vola, 5).tolist()}")

        except Exception as e:
            self.logger.error(f"Error in overall_signal: {e} | Line: {e.__traceback__.tb_lineno}")

        return curr_signal, curr_vola


    def get_entry(self) -> Dict[str, tuple[int, float]]:
        """
        Generate entry signals for available symbols.
        Returns:
            Dict[symbol -> (signal, price_limit)]
        """
        signal, vola = self.overall_signal()
        if not np.any(signal):
            self.logger.info("No signals generated in this cycle.")
            return {}

        entry_dict = {}
        self.logger.info("=== Generating Entry Orders ===")
        for i, symbol in enumerate(self.symbols):
            if signal[i] == 0:
                continue

            base, quote = self._extract_symbol(symbol)
            price_i = self.get_imbalance_price(base, quote, 0.01)

            if price_i is None:
                self.logger.warning(f"Skipped {symbol}: No imbalance price available.")
                continue
            
            vol = np.clip(vola[i], 0.0001, 0.01) * signal[i] * -1
            lim_price = price_i * (1 + vol)
            
            entry_dict[symbol] = (int(signal[i]), float(lim_price))

            self.logger.info(
                f"[ENTRY] {symbol} | Signal: {signal[i]:+d} | "
                f"Vola: {vola[i]:.5f} | "
                f"Price: {price_i:.5f} → Limit: {lim_price:.5f}"
            )

        if not entry_dict:
            self.logger.info("No valid entries after filtering.")
        else:
            self.logger.info(f"Total entries generated: {len(entry_dict)}")

        return entry_dict

    def get_opened_positions(self) -> int:
        """Get normalized signed open position count."""
        open_pos = self.get_open_positions().get('data', [])
        pos_val, sign = 0.0, 0

        for pos in open_pos:
            if pos['symbol'] != self.base + self.quote:
                continue
            pos_val += abs(float(pos['positionValue']))
            sign = 1 if pos['side'].upper() == 'BUY' else -1

        if sign == 0:
            return 0
        return int(round(pos_val / self.value_entry) * sign)

    def calculate_position_size(self, value_entry: float = 10, price: Optional[float] = None) -> float:
        """Calculate position size based on available quote balance and risk pct."""
        balance = self.get_balance().get('data', {}).get(self.quote, {}).get('total', 0) * 10

        if self.value_entry > balance:
            self.logger.warning("Insufficient balance to open a new position.")
            return 0.0

        price = price or self.get_price(self.base, self.quote)
        if not price or price <= 0:
            self.logger.warning("Invalid price for sizing.")
            return 0.0
        
        quantity = round(value_entry / price, 6)
        return quantity
    
    def handle_orders(self):
        """Handle existing orders."""
        now = int(time.time() * 1000)
        expired_keys = []

        for ts, data in list(self.placed_orders.items()):
            if now - ts >= self.refresh_order * 1000:
                order_id = data.get('order_id')
                base = data.get('base')
                quote = data.get('quote')
                self.cancel_order(base=base, quote=quote, order_id=order_id)
                self.logger.info(f"Order {order_id} cancelled.")
                expired_keys.append(ts)

        for ts in expired_keys:
            del self.placed_orders[ts]
            
    def execute_trade(self, signal: int, price: float, quantity: float,
                      order_type: str = 'limit', force: str = 'GTC') -> bool:
        """
        Ensures delay between repeated executions using latest_entry_ts.
        """
        
        order_id = None
        if signal == 1:
            ask = self.get_price(self.base, self.quote, 'bid')
            price = min(ask, price)
            order_id = self.buy(quantity, price, order_type=order_type, force=force)

        elif signal == -1:
            bid = self.get_price(self.base, self.quote, 'ask')
            price = max(bid, price)
            order_id = self.sell(quantity, price, order_type=order_type, force=force)

        now = int(time.time() * 1000)
        if order_id:
            self.placed_orders[now] = {'order_id': order_id, 'base': self.base, 'quote': self.quote}

    def close_all_remaining_positions(self):
        """Close all remaining positions forcefully."""
        self.logger.info("Closing all remaining orders and positions forcefully...")
        self.position.cancel_all_orders(self, force_close=True)
        self.position.close_all_positions(self, force_close=True)

    def handle_stop_loss(self):
        """Check stop loss condition."""
        curr_balance = self.get_balance().get('data', {}).get(self.quote, {}).get('total', 0)
        if curr_balance < self.init_balance * (1 - self.stop_loss_threshold):
            self.logger.warning("Stop Loss triggered!")
            self.stop_flag = True
            self.handle_exit()

    def run(self):
        """Run main strategy loop."""
        self.close_all_remaining_positions()
        self.logger.info("Starting Strategy...")
        self.init_balance = self.get_balance().get('data', {}).get(self.quote, {}).get('total', 0)
        try:
            while not self.stop_flag:
                self.handle_orders()
                self.handle_stop_loss()
                signal = self.get_entry()
                if signal:
                    for symbol, (sig, lim_price) in signal.items():
                        base, quote = self._extract_symbol(symbol)
                        self.base, self.quote = base, quote
                        
                        price = self.get_price(self.base, self.quote)
                        size = self.calculate_position_size(self.value_entry, price)
                        if size > 0:
                            self.execute_trade(sig, lim_price, size, order_type='limit')
                        else:
                            self.logger.warning("Zero position size, skipping trade.")
                else:
                    pass

                time.sleep(self.sleep_time)

        except Exception as e:
            self.logger.exception(f"Run error: {e}")

        finally:
            self.handle_exit()
            self.logger.info("Strategy stopped cleanly.")
