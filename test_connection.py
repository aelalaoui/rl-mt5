# test_environment.py
from src.environment import TradingEnvironment
import time

# Créer l'environnement
env = TradingEnvironment()

# Tester le reset
print("Testing reset...")
observation = env.reset()
print(f"Initial observation shape: {observation.shape}")

# Tester quelques actions
print("\nTesting actions...")
for action in [0, 1, 2, 0]:  # Ne rien faire, acheter, vendre, ne rien faire
    print(f"Taking action: {action}")
    next_obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}")
    print(f"Observation shape: {next_obs.shape}")
    time.sleep(5)  # Attendre 5 secondes entre les actions

# Fermer l'environnement
env.close()
print("Environment test completed!")