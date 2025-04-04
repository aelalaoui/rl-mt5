# src/backtest_environment.py (version modifiée)
import numpy as np
import pandas as pd
from gym import spaces
import time

class BacktestTradingEnvironment:
    def __init__(self, data_path, window_size=20):
        # Charger les données historiques
        self.data = pd.read_csv(data_path)

        # Vérifier si 'time' est une colonne ou un index
        if 'time' in self.data.columns:
            self.data.set_index('time', inplace=True)
            try:
                self.data.index = pd.to_datetime(self.data.index)
            except:
                print("Warning: Could not convert time to datetime")

        print(f"Data loaded with shape: {self.data.shape}")

        self.window_size = window_size

        # Index actuel dans les données
        self.current_idx = window_size
        self.max_idx = len(self.data) - 1
        print(f"Starting at index {self.current_idx}, max index: {self.max_idx}")

        # État du trading
        self.position = 0  # 0: pas de position, 1: long, -1: short
        self.position_price = 0.0
        self.balance = 10000.0
        self.equity_curve = [self.balance]

        # Définir l'espace d'observation et d'action
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size*4+3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: ne rien faire, 1: acheter, 2: vendre

        # Préparer l'observation initiale
        self._update_observation()

    def _update_observation(self):
        """Mettre à jour l'observation basée sur la fenêtre actuelle des données"""
        print(f"Updating observation at index {self.current_idx}", flush=True)

        # Extraire la fenêtre de données
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

        # Normaliser les prix par rapport au dernier prix de clôture
        last_close = window_data.iloc[-1]['close']

        # Créer des caractéristiques
        opens = (window_data['open'].values / last_close) - 1.0
        highs = (window_data['high'].values / last_close) - 1.0
        lows = (window_data['low'].values / last_close) - 1.0
        closes = (window_data['close'].values / last_close) - 1.0

        # Ajouter des indicateurs techniques simples
        # RSI simplifié
        rsi = 0.5  # Valeur par défaut
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

        # Moyennes mobiles simplifiées
        ma_5 = window_data['close'].rolling(5, min_periods=1).mean().iloc[-1]
        ma_20 = window_data['close'].rolling(20, min_periods=1).mean().iloc[-1]

        # Créer l'observation
        observation = np.concatenate([
            opens[-self.window_size:],
            highs[-self.window_size:],
            lows[-self.window_size:],
            closes[-self.window_size:],
            [self.position, rsi/100, (ma_5/last_close)-1]
        ])

        # Remplacer les NaN par 0
        observation = np.nan_to_num(observation)

        self.last_observation = observation
        return observation

    def reset(self):
        """Réinitialiser l'environnement au début des données"""
        print("Resetting environment", flush=True)
        self.current_idx = self.window_size
        self.position = 0
        self.position_price = 0.0
        self.balance = 10000.0
        self.equity_curve = [self.balance]

        return self._update_observation()

    def step(self, action):
        """Exécuter une action et avancer dans les données"""
        # Sauvegarder l'état actuel
        current_price = self.data.iloc[self.current_idx]['close']
        print(f"Step at idx {self.current_idx}, price: {current_price}, action: {action}", flush=True)

        # Calculer le P&L si une position est ouverte
        pnl = 0
        if self.position == 1:  # Long
            pnl = (current_price - self.position_price) / self.position_price
        elif self.position == -1:  # Short
            pnl = (self.position_price - current_price) / self.position_price

        # Exécuter l'action
        if action == 1 and self.position <= 0:  # Acheter
            self.position = 1
            self.position_price = current_price
            print(f"  BUY at {current_price}", flush=True)
        elif action == 2 and self.position >= 0:  # Vendre
            self.position = -1
            self.position_price = current_price
            print(f"  SELL at {current_price}", flush=True)

        # Avancer à la prochaine barre
        self.current_idx += 1

        # Vérifier si nous avons atteint la fin des données
        done = self.current_idx >= len(self.data) - 1

        if done:
            print(f"End of data reached at index {self.current_idx}", flush=True)
            # Fermer toutes les positions à la fin
            if self.position != 0:
                final_price = self.data.iloc[-1]['close']
                if self.position == 1:  # Long
                    pnl = (final_price - self.position_price) / self.position_price
                else:  # Short
                    pnl = (self.position_price - final_price) / self.position_price
                self.balance += self.balance * pnl
                print(f"  Closing position at end with PnL: {pnl:.4f}", flush=True)

        # Mettre à jour l'observation
        if not done:
            observation = self._update_observation()
        else:
            observation = self.last_observation

        # Calculer la récompense
        if action == 0 and self.position == 0:
            reward = -0.01  # Petite pénalité pour l'inactivité
        else:
            reward = pnl * 100  # Convertir en points

        # Mettre à jour le solde
        self.balance += self.balance * pnl
        self.equity_curve.append(self.balance)

        print(f"  Reward: {reward:.4f}, Balance: {self.balance:.2f}, Done: {done}", flush=True)

        # Informations supplémentaires
        info = {
            'balance': self.balance,
            'position': self.position,
            'pnl': pnl
        }

        return observation, reward, done, info

    def render(self):
        """Afficher l'état actuel"""
        print(f"Balance: {self.balance:.2f}, Position: {self.position}")