# market_utils.py
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
import pytz

class MarketUtils:
    @staticmethod
    def is_trading_session(timestamp):
        """
        Check if current time is within London or New York trading sessions
        London: 8:00-16:30 GMT
        New York: 13:30-20:00 GMT
        """
        if isinstance(timestamp, str):
            dt = pd.to_datetime(timestamp)
        else:
            dt = timestamp

        # Convert to UTC/GMT for consistency
        if dt.tzinfo is None:
            # Assume UTC if no timezone
            dt = pytz.utc.localize(dt)
        else:
            dt = dt.astimezone(pytz.utc)

        # Check if it's a weekend
        if dt.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return False

        # Define session times (in UTC)
        london_start = time(8, 0)
        london_end = time(16, 30)
        ny_start = time(13, 30)
        ny_end = time(20, 0)

        current_time = dt.time()

        # Check if current time is in either London or NY session
        london_session = london_start <= current_time <= london_end
        ny_session = ny_start <= current_time <= ny_end

        return london_session or ny_session

    @staticmethod
    def is_near_weekend(timestamp, hours_before=12):
        """
        Check if current time is approaching weekend (Friday afternoon)
        """
        if isinstance(timestamp, str):
            dt = pd.to_datetime(timestamp)
        else:
            dt = timestamp

        # Convert to UTC/GMT for consistency
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        else:
            dt = dt.astimezone(pytz.utc)

        # Check if it's Friday
        if dt.weekday() == 4:  # 4 = Friday
            # Check if it's past noon on Friday
            weekend_cutoff = time(20, 0) - timedelta(hours=hours_before)
            return dt.time() >= weekend_cutoff
        return False

    @staticmethod
    def calculate_volatility(price_history, window=20):
        """Calculate price volatility using standard deviation of returns"""
        if len(price_history) < window + 1:
            return 0.01  # Default if not enough history

        recent_prices = price_history[-window:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        return np.std(returns)