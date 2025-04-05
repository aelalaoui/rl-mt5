# config.py
# Configuration globale pour le robot de trading RL

# Paramètres de trading
SYMBOL = "EURUSD"
TIMEFRAME = "M15"  # 15 minutes
INITIAL_BALANCE = 10000  # Balance initiale pour le backtesting

# Paramètres RL
MAX_STEPS = 2000  # Nombre maximum d'étapes par épisode
LOOKBACK_WINDOW_SIZE = 60  # Nombre de bougies précédentes à considérer

# Paramètres de gestion du risque
MAX_POSITION_SIZE = 0.02  # Taille maximale de position (% du capital)
STOP_LOSS_PIPS = 30  # Stop loss en pips
TAKE_PROFIT_PIPS = 60  # Take profit en pips

# Paramètres de communication ZeroMQ
ZMQ_HOST = "127.0.0.1"
ZMQ_PORT_PUB = 5557  # Port pour publier les données de MT4 vers Python
ZMQ_PORT_SUB = 5558  # Port pour envoyer les commandes de Python vers MT4

# Chemins des fichiers
DATA_DIR = "./data"
MODELS_DIR = "./models"