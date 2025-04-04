import pandas as pd
# Dans votre script d'entraînement
data = pd.read_csv("data/EURUSD_SYNTHETIC_2YEARS.csv", index_col='time', parse_dates=True)
split_point = int(len(data) * 0.8)  # 80% pour l'entraînement, 20% pour le test
train_data = data.iloc[:split_point]
test_data = data.iloc[split_point:]

train_data.to_csv("data/EURUSD_train.csv")
test_data.to_csv("data/EURUSD_test.csv")