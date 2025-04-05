# src/connector.py
import zmq
import time
import threading
import json
from typing import Dict, List, Any, Callable

class MT4Connector:
    def __init__(self, host: str, pub_port: int, sub_port: int):
        """
        Initialise la connexion ZeroMQ avec MetaTrader 4
        
        Args:
            host: Adresse IP du serveur ZeroMQ
            pub_port: Port pour recevoir les données de MT4
            sub_port: Port pour envoyer les commandes à MT4
        """
        self.host = host
        self.pub_port = pub_port
        self.sub_port = sub_port
        
        # Initialiser le contexte ZeroMQ
        self.context = zmq.Context()
        
        # Socket pour recevoir les données de MT4
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{host}:{pub_port}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Socket pour envoyer des commandes à MT4
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://{host}:{sub_port}")
        
        # Variables pour stocker les données
        self.market_data = {}
        self.account_info = {}
        self.is_running = False
        self.callback = None
        
        print(f"MT4Connector initialized. Listening on port {pub_port}, sending on port {sub_port}")
    
    def start(self, callback: Callable = None):
        """
        Démarre la réception des données en arrière-plan
        
        Args:
            callback: Fonction à appeler lorsque de nouvelles données sont reçues
        """
        self.callback = callback
        self.is_running = True
        self.receiver_thread = threading.Thread(target=self._receive_data)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()
        print("MT4Connector started")
    
    def stop(self):
        """Arrête la réception des données"""
        self.is_running = False
        if hasattr(self, 'receiver_thread'):
            self.receiver_thread.join(timeout=1.0)
        self.sub_socket.close()
        self.pub_socket.close()
        self.context.term()
        print("MT4Connector stopped")
    
    def _receive_data(self):
        """Fonction de réception des données en arrière-plan"""
        while self.is_running:
            try:
                # Recevoir les données avec un timeout pour pouvoir arrêter proprement
                if self.sub_socket.poll(100):
                    message = self.sub_socket.recv_string()
                    self._process_message(message)
            except zmq.ZMQError as e:
                print(f"ZMQ Error: {e}")
                time.sleep(0.1)
            except Exception as e:
                print(f"Error in receive_data: {e}")
                time.sleep(0.1)
    
    def _process_message(self, message: str):
        """
        Traite un message reçu de MT4
        
        Args:
            message: Message reçu au format string
        """
        parts = message.split('|')
        
        if parts[0] == "MARKET_DATA" and len(parts) > 10:
            # Extraire les données de marché
            self.market_data = {
                "symbol": parts[1],
                "bid": float(parts[2]),
                "ask": float(parts[3]),
                "candles": []
            }
            
            # Extraire les données des bougies
            idx = 4
            for i in range(5):  # 5 dernières bougies
                if idx + 5 <= len(parts):
                    candle = {
                        "open": float(parts[idx]),
                        "high": float(parts[idx + 1]),
                        "low": float(parts[idx + 2]),
                        "close": float(parts[idx + 3]),
                        "volume": int(parts[idx + 4])
                    }
                    self.market_data["candles"].append(candle)
                    idx += 5
            
            # Extraire les informations du compte
            if idx + 3 <= len(parts):
                self.account_info = {
                    "balance": float(parts[idx]),
                    "equity": float(parts[idx + 1]),
                    "orders_total": int(parts[idx + 2])
                }
            
            # Appeler le callback si défini
            if self.callback:
                self.callback(self.market_data, self.account_info)
    
    def open_order(self, order_type: str, volume: float, sl: float = 0, tp: float = 0):
        """
        Ouvre un ordre sur MT4
        
        Args:
            order_type: Type d'ordre ("BUY" ou "SELL")
            volume: Volume de l'ordre
            sl: Prix du stop loss (0 pour désactiver)
            tp: Prix du take profit (0 pour désactiver)
        """
        command = f"OPEN_ORDER|{order_type}|{volume}|{sl}|{tp}"
        self.pub_socket.send_string(command)
        print(f"Sent command: {command}")
    
    def close_order(self, ticket: int):
        """
        Ferme un ordre spécifique
        
        Args:
            ticket: Numéro de ticket de l'ordre à fermer
        """
        command = f"CLOSE_ORDER|{ticket}"
        self.pub_socket.send_string(command)
        print(f"Sent command: {command}")
    
    def close_all_orders(self):
        """Ferme tous les ordres ouverts"""
        command = "CLOSE_ALL"
        self.pub_socket.send_string(command)
        print("Sent command: CLOSE_ALL")
    
    def get_market_data(self):
        """Retourne les dernières données de marché"""
        return self.market_data
    
    def get_account_info(self):
        """Retourne les dernières informations du compte"""
        return self.account_info