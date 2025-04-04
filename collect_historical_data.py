# create_synthetic_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Créer des dates (2 ans de données M15)
start_date = datetime.now() - timedelta(days=730)
dates = [start_date + timedelta(minutes=15*i) for i in range(70080)]  # 2 ans de M15

# Simuler des prix avec un modèle plus réaliste
np.random.seed(42)
close = 1.1000  # Prix initial
closes = [close]
volatility = 0.0005  # Volatilité initiale

for i in range(1, len(dates)):
    # Simuler la volatilité stochastique
    volatility = max(0.0001, volatility * (1 + np.random.normal(0, 0.05)))

    # Simuler un mouvement de prix avec tendance et saisonnalité
    trend = 0.00001 * np.sin(i / 5000)  # Tendance cyclique à long terme
    seasonal = 0.0001 * np.sin(i / 96)  # Saisonnalité journalière (96 périodes M15 par jour)
    random_walk = np.random.normal(0, volatility)  # Marche aléatoire

    change = trend + seasonal + random_walk
    close = close * (1 + change)
    closes.append(close)

# Créer les autres colonnes de prix
opens = [closes[i-1] for i in range(len(closes))]
opens[0] = closes[0] * 0.9999
highs = [max(opens[i], closes[i]) * (1 + abs(np.random.normal(0, volatility*0.5))) for i in range(len(closes))]
lows = [min(opens[i], closes[i]) * (1 - abs(np.random.normal(0, volatility*0.5))) for i in range(len(closes))]
volumes = [int(np.random.exponential(500) * (1 + abs(np.random.normal(0, 0.3)))) for _ in range(len(closes))]

# Créer le DataFrame
df = pd.DataFrame({
    'time': dates,
    'open': opens,
    'high': highs,
    'low': lows,
    'close': closes,
    'tick_volume': volumes,
    'spread': 2,
    'real_volume': 0
})

df.set_index('time', inplace=True)

# Sauvegarder les données
output_file = "data/EURUSD_SYNTHETIC_2YEARS.csv"
df.to_csv(output_file)

print(f"Données synthétiques créées dans {output_file}: {len(df)} barres")
print("\nAperçu des données:")
print(df.head())