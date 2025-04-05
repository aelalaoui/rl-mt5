# src/backtest_environment.py (enhanced version)
import numpy as np
import pandas as pd
from gym import spaces
import time
import random
from datetime import datetime, time as dtime

class BacktestTradingEnvironment:
    def __init__(self, data_path, window_size=20,
                 commission_fee=0.0001,  # 0.01% commission
                 slippage_base=0.0001,   # 1 pip base slippage
                 slippage_vol_impact=0.00005,  # additional slippage based on volume
                 bid_ask_spread=0.0002,  # 2 pips spread
                 market_impact_factor=0.0001,  # price impact of large orders
                 liquidity_limit=0.05,   # max 5% of average volume
                 latency_ms=(10, 50)):   # latency between 10-50ms

        # Load historical data
        self.data = pd.read_csv(data_path)

        # Check if 'time' is a column or an index
        if 'time' in self.data.columns:
            self.data.set_index('time', inplace=True)
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except:
                print("Warning: Could not convert time to datetime")

        print(f"Data loaded with shape: {self.data.shape}")

        # Calculate average trading volume
        if 'volume' in self.data.columns:
            self.avg_volume = self.data['volume'].mean()
        else:
            self.avg_volume = 10000  # default if no volume data
            print("Warning: No volume data found, using default")

        # Trading parameters
        self.window_size = window_size
        self.commission_fee = commission_fee
        self.slippage_base = slippage_base
        self.slippage_vol_impact = slippage_vol_impact
        self.bid_ask_spread = bid_ask_spread
        self.market_impact_factor = market_impact_factor
        self.liquidity_limit = liquidity_limit
        self.latency_ms = latency_ms

        # Current position in the data
        self.current_idx = window_size
        self.max_idx = len(self.data) - 1
        print(f"Starting at index {self.current_idx}, max index: {self.max_idx}")

        # Trading state
        self.position = 0  # 0: no position, 1: long, -1: short
        self.position_price = 0.0
        self.balance = 10000.0
        self.equity_curve = [self.balance]
        self.trades = []

        # Define observation and action spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size*4+3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: do nothing, 1: buy, 2: sell

        # Prepare initial observation
        self._update_observation()

    def _is_market_open(self, timestamp):
        """Check if the market is open based on timestamp"""
        # For EUR/USD, the market is closed on weekends
        # and has reduced activity during certain hours
        if timestamp.weekday() >= 5:  # Saturday and Sunday
            return False

        # Check trading sessions (simplification)
        # Note: EUR/USD trades 24/5, but activity/liquidity varies
        t = timestamp.time()

        # Low liquidity periods (simplified)
        low_liquidity = (
            (dtime(0, 0) <= t <= dtime(3, 0)) or  # Asian session wind-down
            (dtime(21, 0) <= t <= dtime(23, 59))   # After US session
        )

        # Adjust spread and slippage during low liquidity
        if low_liquidity:
            self.current_spread = self.bid_ask_spread * 2
            self.current_slippage_factor = 2.0
        else:
            self.current_spread = self.bid_ask_spread
            self.current_slippage_factor = 1.0

        return True

    def _calculate_slippage(self, action, price, size=1.0):
        """Calculate realistic slippage based on action, price, and market conditions"""
        # Base slippage
        slippage = self.slippage_base * self.current_slippage_factor

        # Add volatility component
        if hasattr(self, 'volatility'):
            slippage += self.volatility * self.slippage_vol_impact

        # Larger slippage for larger position sizes (market impact)
        position_impact = size * self.market_impact_factor
        slippage += position_impact

        # Direction-based slippage (buy orders slip up, sell orders slip down)
        if action == 1:  # buy
            return slippage
        elif action == 2:  # sell
            return -slippage
        return 0

    def _get_bid_ask_prices(self):
        """Get current bid and ask prices based on mid price and spread"""
        mid_price = self.data.iloc[self.current_idx]['close']
        half_spread = self.current_spread / 2
        bid_price = mid_price - half_spread
        ask_price = mid_price + half_spread
        return bid_price, ask_price

    def _update_observation(self):
        """Update the observation based on the current data window"""
        print(f"Updating observation at index {self.current_idx}", flush=True)

        # Extract the data window
        start_idx = max(0, self.current_idx-self.window_size)
        end_idx = self.current_idx

        if end_idx >= len(self.data):
            print(f"Warning: end_idx {end_idx} >= data length {len(self.data)}", flush=True)
            end_idx = len(self.data) - 1

        window_data = self.data.iloc[start_idx:end_idx+1]

        if len(window_data) < self.window_size:
            print(f"Warning: window_data size {len(window_data)} < window_size {self.window_size}", flush=True)
            # Padding if needed
            padding = self.window_size - len(window_data)
            padding_data = window_data.iloc[0:1].copy()
            for _ in range(padding):
                window_data = pd.concat([padding_data, window_data])

        # Calculate volatility for slippage estimation
        self.volatility = window_data['close'].pct_change().std()

        # Update current bid/ask spread based on volatility
        self.current_spread = self.bid_ask_spread * (1 + 5 * self.volatility)

        # Normalize prices relative to last close price
        last_close = window_data.iloc[-1]['close']

        # Create features
        opens = (window_data['open'].values / last_close) - 1.0
        highs = (window_data['high'].values / last_close) - 1.0
        lows = (window_data['low'].values / last_close) - 1.0
        closes = (window_data['close'].values / last_close) - 1.0

        # Add simple technical indicators
        # RSI
        rsi = 0.5  # Default value
        try:
            delta = window_data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            if np.isnan(rsi):
                rsi = 50
        except Exception as e:
            print(f"RSI calculation error: {e}", flush=True)
            rsi = 50

        # Moving averages
        ma_5 = window_data['close'].rolling(5, min_periods=1).mean().iloc[-1]
        ma_20 = window_data['close'].rolling(20, min_periods=1).mean().iloc[-1]

        # Create observation
        observation = np.concatenate([
            opens[-self.window_size:],
            highs[-self.window_size:],
            lows[-self.window_size:],
            closes[-self.window_size:],
            [self.position, rsi/100, (ma_5/last_close)-1]
        ])

        # Replace NaN with 0
        observation = np.nan_to_num(observation)

        self.last_observation = observation
        return observation

    def reset(self):
        """Reset the environment to the beginning of the data"""
        print("Resetting environment", flush=True)
        self.current_idx = self.window_size
        self.position = 0
        self.position_price = 0.0
        self.balance = 10000.0
        self.equity_curve = [self.balance]
        self.trades = []

        return self._update_observation()

    def step(self, action):
        """Execute an action and advance in the data"""
        # Save current state
        timestamp = self.data.index[self.current_idx] if hasattr(self.data.index, 'to_pydatetime') else None
        bid_price, ask_price = self._get_bid_ask_prices()
        mid_price = self.data.iloc[self.current_idx]['close']

        print(f"Step at idx {self.current_idx}, mid: {mid_price:.5f}, bid: {bid_price:.5f}, ask: {ask_price:.5f}, action: {action}", flush=True)

        # Check if market is open (if timestamp is available)
        market_open = True
        if timestamp is not None:
            market_open = self._is_market_open(timestamp)

        # Fixed position size - 2% of account or maximum $1000
        position_size = min(self.balance * 0.02, 1000.0)

        # Calculate current position value and P&L before taking new action
        unrealized_pnl = 0
        if self.position == 1:  # Long
            unrealized_pnl = position_size * (bid_price - self.position_price) / self.position_price
        elif self.position == -1:  # Short
            unrealized_pnl = position_size * (self.position_price - ask_price) / self.position_price

        # Execute action if market is open
        trade_executed = False
        slippage = 0
        commission = 0
        reward = 0
        close_pnl = 0

        if market_open:
            # Simulate network latency
            latency = random.randint(self.latency_ms[0], self.latency_ms[1]) / 1000
            time.sleep(latency)

            if action == 1:  # BUY
                if self.position == 1:
                    # Already long - do nothing
                    print(" Already in LONG position", flush=True)
                    trade_executed = False
                elif self.position == -1:
                    # Close short position first
                    close_slippage = self._calculate_slippage(1, ask_price)
                    close_price = ask_price * (1 + close_slippage)
                    close_commission = close_price * position_size * self.commission_fee

                    # Calculate P&L from closing short
                    close_pnl = position_size * (self.position_price - close_price) / self.position_price
                    self.balance += close_pnl - close_commission

                    # Open new long position
                    slippage = self._calculate_slippage(1, ask_price)
                    execution_price = ask_price * (1 + slippage)
                    commission = execution_price * position_size * self.commission_fee

                    # Update position
                    self.position = 1
                    self.position_price = execution_price
                    self.balance -= commission

                    trade_executed = True
                    print(f" CLOSE SHORT & BUY at {execution_price:.5f} (P&L: {close_pnl:.2f})", flush=True)
                else:  # No position
                    # Open new long position
                    slippage = self._calculate_slippage(1, ask_price)
                    execution_price = ask_price * (1 + slippage)
                    commission = execution_price * position_size * self.commission_fee

                    # Update position
                    self.position = 1
                    self.position_price = execution_price
                    self.balance -= commission

                    trade_executed = True
                    print(f" BUY at {execution_price:.5f} (ask: {ask_price:.5f}, slippage: {slippage:.5f}%)", flush=True)

                # Record trade if executed
                if trade_executed:
                    self.trades.append({
                        'timestamp': timestamp,
                        'action': 'buy',
                        'price': execution_price,
                        'size': position_size,
                        'slippage': slippage,
                        'commission': commission,
                        'pnl_from_close': close_pnl
                    })

            elif action == 2:  # SELL
                if self.position == -1:
                    # Already short - do nothing
                    print(" Already in SHORT position", flush=True)
                    trade_executed = False
                elif self.position == 1:
                    # Close long position first
                    close_slippage = self._calculate_slippage(2, bid_price)
                    close_price = bid_price * (1 + close_slippage)
                    close_commission = close_price * position_size * self.commission_fee

                    # Calculate P&L from closing long
                    close_pnl = position_size * (close_price - self.position_price) / self.position_price
                    self.balance += close_pnl - close_commission

                    # Open new short position
                    slippage = self._calculate_slippage(2, bid_price)
                    execution_price = bid_price * (1 + slippage)
                    commission = execution_price * position_size * self.commission_fee

                    # Update position
                    self.position = -1
                    self.position_price = execution_price
                    self.balance -= commission

                    trade_executed = True
                    print(f" CLOSE LONG & SELL at {execution_price:.5f} (P&L: {close_pnl:.2f})", flush=True)
                else:  # No position
                    # Open new short position
                    slippage = self._calculate_slippage(2, bid_price)
                    execution_price = bid_price * (1 + slippage)
                    commission = execution_price * position_size * self.commission_fee

                    # Update position
                    self.position = -1
                    self.position_price = execution_price
                    self.balance -= commission

                    trade_executed = True
                    print(f" SELL at {execution_price:.5f} (bid: {bid_price:.5f}, slippage: {slippage:.5f}%)", flush=True)

                # Record trade if executed
                if trade_executed:
                    self.trades.append({
                        'timestamp': timestamp,
                        'action': 'sell',
                        'price': execution_price,
                        'size': position_size,
                        'slippage': slippage,
                        'commission': commission,
                        'pnl_from_close': close_pnl
                    })

            elif action == 0:  # CLOSE position
                if self.position == 0:
                    # No position to close
                    print(" No position to close", flush=True)
                    trade_executed = False
                elif self.position == 1:  # Close long
                    close_slippage = self._calculate_slippage(2, bid_price)
                    close_price = bid_price * (1 + close_slippage)
                    close_commission = close_price * position_size * self.commission_fee

                    # Calculate P&L from closing long
                    close_pnl = position_size * (close_price - self.position_price) / self.position_price
                    self.balance += close_pnl - close_commission

                    # Reset position
                    self.position = 0
                    self.position_price = 0.0

                    trade_executed = True
                    print(f" CLOSE LONG at {close_price:.5f} (P&L: {close_pnl:.2f})", flush=True)
                elif self.position == -1:  # Close short
                    close_slippage = self._calculate_slippage(1, ask_price)
                    close_price = ask_price * (1 + close_slippage)
                    close_commission = close_price * position_size * self.commission_fee

                    # Calculate P&L from closing short
                    close_pnl = position_size * (self.position_price - close_price) / self.position_price
                    self.balance += close_pnl - close_commission

                    # Reset position
                    self.position = 0
                    self.position_price = 0.0

                    trade_executed = True
                    print(f" CLOSE SHORT at {close_price:.5f} (P&L: {close_pnl:.2f})", flush=True)

                # Record trade if executed
                if trade_executed:
                    self.trades.append({
                        'timestamp': timestamp,
                        'action': 'close',
                        'price': close_price,
                        'size': position_size,
                        'slippage': close_slippage,
                        'commission': close_commission,
                        'pnl': close_pnl
                    })

        # Advance to next bar
        self.current_idx += 1

        # Check if we've reached the end of the data
        done = self.current_idx >= len(self.data) - 1

        if done:
            print(f"End of data reached at index {self.current_idx}", flush=True)
            # Close all positions at the end
            if self.position != 0:
                final_bid, final_ask = self._get_bid_ask_prices()
                if self.position == 1:  # Long
                    final_price = final_bid * (1 + self._calculate_slippage(2, final_bid))
                    final_commission = final_price * position_size * self.commission_fee
                    final_pnl = position_size * (final_price - self.position_price) / self.position_price
                else:  # Short
                    final_price = final_ask * (1 + self._calculate_slippage(1, final_ask))
                    final_commission = final_price * position_size * self.commission_fee
                    final_pnl = position_size * (self.position_price - final_price) / self.position_price

                self.balance += final_pnl - final_commission
                print(f" Closing position at end with PnL: {final_pnl:.4f}", flush=True)

        # Update observation
        if not done:
            observation = self._update_observation()
        else:
            observation = self.last_observation

        # Calculate reward
        if trade_executed:
            if close_pnl != 0:
                # If we closed a position, use realized P&L as reward
                reward = close_pnl / position_size * 100  # Convert to percentage points
            else:
                # If we opened a position, use transaction cost as negative reward
                reward = -(slippage + self.commission_fee) * 100  # Convert to basis points
        else:
            if self.position == 0:
                reward = -0.01  # Small penalty for inactivity
            else:
                # Use unrealized P&L for reward
                reward = unrealized_pnl / position_size * 100  # Convert to percentage points

        # Update equity curve
        self.equity_curve.append(self.balance)

        print(f" Reward: {reward:.4f}, Balance: {self.balance:.2f}, Done: {done}", flush=True)

        # Additional information
        info = {
            'balance': self.balance,
            'position': self.position,
            'position_size': position_size if self.position != 0 else 0,
            'position_price': self.position_price,
            'unrealized_pnl': unrealized_pnl,
            'bid': bid_price,
            'ask': ask_price,
            'spread': ask_price - bid_price,
            'slippage': slippage,
            'commission': commission,
            'market_open': market_open
        }

        return observation, reward, done, info

    def render(self):
        """Display current state"""
        print(f"Balance: {self.balance:.2f}, Position: {self.position}")