import numpy as np
import zmq
import time
from gym import spaces

class TradingEnvironment:
    def __init__(self, zmq_pub_port=5556, zmq_sub_port=5555, symbol="EURUSD", timeframe="M15"):
        # Configuration ZeroMQ
        self.context = zmq.Context()
        self.socket_sub = self.context.socket(zmq.SUB)
        self.socket_sub.connect(f"tcp://127.0.0.1:{zmq_sub_port}")
        self.socket_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        
        self.socket_pub = self.context.socket(zmq.PUB)
        self.socket_pub.bind(f"tcp://127.0.0.1:{zmq_pub_port}")
        
        # Paramètres de trading
        self.symbol = symbol
        self.timeframe = timeframe
        self.position = 0  # 0: pas de position, 1: long, -1: short
        self.position_size = 0.01  # Taille de lot
        self.position_price = 0.0
        self.balance = 10000.0
        self.equity = 10000.0
        
        # Définir l'espace d'observation et d'action
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: ne rien faire, 1: acheter, 2: vendre
        
        # Données de marché
        self.market_data = None
        self.last_observation = None
        
        # Attendre la première donnée de marché
        self._wait_for_market_data()
    
    def _wait_for_market_data(self):
        """Attendre de recevoir les premières données de marché"""
        print("Waiting for initial market data...")
        while self.market_data is None:
            if self.socket_sub.poll(100):
                message = self.socket_sub.recv_string()
                self._process_message(message)
            time.sleep(0.1)
        print("Initial market data received")
    
    def _process_message(self, message):
        """Traiter un message reçu de MT5"""
        parts = message.split('|')
        if parts[0] == "MARKET_DATA":
            # Extraire les données de marché
            self.symbol = parts[1]
            bid = float(parts[2])
            ask = float(parts[3])
            
            # Extraire les données OHLCV
            ohlcv_data = []
            for i in range(5):  # 5 bougies
                start_idx = 4 + i * 5
                open_price = float(parts[start_idx])
                high_price = float(parts[start_idx + 1])
                low_price = float(parts[start_idx + 2])
                close_price = float(parts[start_idx + 3])
                volume = float(parts[start_idx + 4])
                ohlcv_data.append([open_price, high_price, low_price, close_price, volume])
            
            # Extraire les informations du compte
            account_idx = 4 + 5 * 5
            self.balance = float(parts[account_idx])
            self.equity = float(parts[account_idx + 1])
            positions_count = int(parts[account_idx + 2])
            
            # Mettre à jour les données de marché
            self.market_data = {
                "bid": bid,
                "ask": ask,
                "ohlcv": ohlcv_data,
                "positions_count": positions_count
            }
            
            # Mettre à jour l'observation
            self._update_observation()
        
        elif parts[0] == "POSITION_UPDATE":
            # Traiter les mises à jour de position
            positions_count = int(parts[1])
            if positions_count > 0:
                # Mettre à jour la position actuelle
                # (code pour traiter les détails de la position)
                pass
    
    def _update_observation(self):
        """Mettre à jour l'observation basée sur les données de marché"""
        if self.market_data is None:
            return
        
        # Extraire les caractéristiques des données OHLCV
        ohlcv = np.array(self.market_data["ohlcv"])
        
        # Normaliser les prix par rapport au dernier prix de clôture
        last_close = ohlcv[0, 3]
        normalized_ohlcv = ohlcv / last_close - 1.0
        
        # Créer l'observation
        observation = np.zeros(20)
        
        # Ajouter les prix normalisés
        observation[0:5] = normalized_ohlcv[:, 0]  # Open
        observation[5:10] = normalized_ohlcv[:, 1]  # High
        observation[10:15] = normalized_ohlcv[:, 2]  # Low
        observation[15:20] = normalized_ohlcv[:, 3]  # Close
        
        self.last_observation = observation
        return observation
    
    def reset(self):
        """Réinitialiser l'environnement"""
        # Fermer toutes les positions
        self._send_command("CLOSE_ALL")
        time.sleep(1)  # Attendre que les positions soient fermées
        
        # Attendre de nouvelles données de marché
        self._wait_for_market_data()
        
        # Réinitialiser l'état
        self.position = 0
        self.position_price = 0.0
        
        return self.last_observation
    
    def step(self, action):
        """Exécuter une action et retourner la nouvelle observation, la récompense, etc."""
        # Exécuter l'action
        reward = 0
        done = False
        info = {}
        
        if action == 1 and self.position <= 0:  # Acheter
            self._send_command(f"OPEN_ORDER|BUY|{self.position_size}|0|0")
            self.position = 1
            self.position_price = self.market_data["ask"]
        elif action == 2 and self.position >= 0:  # Vendre
            self._send_command(f"OPEN_ORDER|SELL|{self.position_size}|0|0")
            self.position = -1
            self.position_price = self.market_data["bid"]
        
        # Attendre la prochaine donnée de marché
        self._wait_for_next_data()
        
        # Calculer la récompense
        if self.position == 1:  # Long
            reward = (self.market_data["bid"] - self.position_price) / self.position_price * 10000
        elif self.position == -1:  # Short
            reward = (self.position_price - self.market_data["ask"]) / self.position_price * 10000
        
        return self.last_observation, reward, done, info
    
    def _wait_for_next_data(self):
        """Attendre la prochaine donnée de marché"""
        old_data = self.market_data
        timeout = time.time() + 30  # Timeout de 30 secondes
        
        while time.time() < timeout:
            if self.socket_sub.poll(100):
                message = self.socket_sub.recv_string()
                self._process_message(message)
                
                # Vérifier si les données ont été mises à jour
                if self.market_data != old_data:
                    return
            
            time.sleep(0.1)
        
        print("Warning: Timeout waiting for new market data")
    
    def _send_command(self, command):
        """Envoyer une commande à MT5"""
        self.socket_pub.send_string(command)
        print(f"Command sent: {command}")
    
    def close(self):
        """Fermer l'environnement"""
        self._send_command("CLOSE_ALL")
        time.sleep(1)
        
        self.socket_sub.close()
        self.socket_pub.close()
        self.context.term()