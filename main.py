import numpy as np
import time
import argparse
import os
import sys
from src.environment import TradingEnvironment
from src.backtest_environment import BacktestTradingEnvironment
from src.agent import DQNAgent

def train_agent(episodes=100, batch_size=32):
    # Créer l'environnement et l'agent
    env = TradingEnvironment()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    # Entraîner l'agent
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for time_step in range(500):  # Limiter à 500 étapes par épisode
            # Choisir une action
            action = agent.act(state)

            # Exécuter l'action
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # Se souvenir de l'expérience
            agent.remember(state, action, reward, next_state, done)

            # Mettre à jour l'état
            state = next_state
            total_reward += reward

            # Entraîner l'agent
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if done:
                break

        # Mettre à jour le modèle cible
        if e % 10 == 0:
            agent.update_target_model()

        print(f"Episode: {e+1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}", flush=True)

        # Sauvegarder le modèle
        if (e+1) % 10 == 0:
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/dqn_agent_ep{e+1}.h5")

    # Fermer l'environnement
    env.close()
    return agent

def train_on_historical_data(data_path, episodes=100, batch_size=32):
    """Entraîner l'agent sur des données historiques"""
    print(f"Starting historical data training on {data_path}", flush=True)

    try:
        # Charger les données
        print("Loading data...", flush=True)
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"CSV columns: {df.columns.tolist()}", flush=True)
        print(f"First few rows:\n{df.head()}", flush=True)
        print(f"Data shape: {df.shape}", flush=True)

        # Créer l'environnement de backtesting
        env = BacktestTradingEnvironment(data_path)
        print(f"Environment created. Data size: {len(env.data)} bars", flush=True)

        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        print(f"State size: {state_size}, Action size: {action_size}", flush=True)

        # Créer l'agent
        print("Creating agent...", flush=True)
        agent = DQNAgent(state_size, action_size)
        print("Agent created", flush=True)

        # Métriques pour le suivi
        all_rewards = []
        all_balances = []

        # Entraîner l'agent
        for e in range(episodes):
            print(f"Starting episode {e+1}/{episodes}", flush=True)

            try:
                state = env.reset()
                state = np.reshape(state, [1, state_size])
                total_reward = 0

                done = False
                time_step = 0
                max_steps = 10000  # Limite de sécurité

                print(f"  Initial state shape: {state.shape}", flush=True)

                while not done and time_step < max_steps:
                    # Choisir une action
                    action = agent.act(state)

                    if time_step % 1000 == 0:  # Log tous les 1000 pas
                        print(f"  Step {time_step}, Action: {action}, Balance: {env.balance:.2f}", flush=True)

                    # Exécuter l'action
                    next_state, reward, done, info = env.step(action)
                    next_state = np.reshape(next_state, [1, state_size])

                    # Se souvenir de l'expérience
                    agent.remember(state, action, reward, next_state, done)

                    # Mettre à jour l'état
                    state = next_state
                    total_reward += reward

                    # Entraîner l'agent
                    if len(agent.memory) > batch_size:
                        agent.replay(batch_size)

                    time_step += 1

                if time_step >= max_steps:
                    print(f"Warning: Episode {e+1} reached max steps limit", flush=True)

                print(f"Episode {e+1} completed after {time_step} steps", flush=True)

                # Mettre à jour le modèle cible
                if e % 10 == 0:
                    agent.update_target_model()

                # Enregistrer les métriques
                all_rewards.append(total_reward)
                all_balances.append(info['balance'])

                print(f"Episode: {e+1}/{episodes}, Reward: {total_reward:.2f}, Final Balance: {info['balance']:.2f}, Epsilon: {agent.epsilon:.2f}", flush=True)

                # Sauvegarder le modèle
                if (e+1) % 10 == 0:
                    os.makedirs("models", exist_ok=True)
                    agent.save(f"models/dqn_agent_backtest_ep{e+1}.h5")
                    print(f"Model saved at episode {e+1}", flush=True)

            except Exception as e:
                print(f"Error during episode {e+1}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue

        # Visualiser les résultats
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 1, 1)
            plt.plot(all_rewards)
            plt.title('Rewards per Episode')

            plt.subplot(2, 1, 2)
            plt.plot(all_balances)
            plt.title('Final Balance per Episode')

            plt.tight_layout()
            os.makedirs("logs", exist_ok=True)
            plt.savefig("logs/backtest_training_results.png")
            print("Results visualization saved to logs/backtest_training_results.png", flush=True)
            plt.show()
        except ImportError:
            print("Matplotlib not available for visualization", flush=True)

        return agent

    except Exception as e:
        print(f"Critical error in training: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None

def evaluate_model(model_path, data_path):
    """Évaluer un modèle entraîné sur des données historiques"""
    print(f"Evaluating model {model_path} on {data_path}", flush=True)

    try:
        # Créer l'environnement de backtesting
        env = BacktestTradingEnvironment(data_path)
        print(f"Evaluation environment created. Data size: {len(env.data)} bars", flush=True)

        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        # Créer et charger l'agent
        agent = DQNAgent(state_size, action_size)
        agent.load(model_path)
        agent.epsilon = 0.0  # Pas d'exploration pendant l'évaluation
        print("Agent loaded with epsilon=0.0", flush=True)

        # Métriques pour le suivi
        actions_taken = []
        equity_curve = []

        # Évaluer l'agent
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        print("Starting evaluation...", flush=True)

        done = False
        time_step = 0
        max_steps = 20000  # Limite de sécurité

        while not done and time_step < max_steps:
            # Choisir une action
            action = agent.act(state)
            actions_taken.append(action)

            if time_step % 1000 == 0:  # Log tous les 1000 pas
                print(f"  Step {time_step}, Action: {action}, Balance: {env.balance:.2f}", flush=True)

            # Exécuter l'action
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # Mettre à jour l'état
            state = next_state
            equity_curve.append(info['balance'])

            time_step += 1

        if time_step >= max_steps:
            print(f"Warning: Evaluation reached max steps limit", flush=True)

        print(f"Evaluation completed after {time_step} steps", flush=True)

        # Calculer les métriques de performance
        initial_balance = 10000.0
        final_balance = equity_curve[-1]
        profit_pct = (final_balance / initial_balance - 1) * 100

        # Calculer le drawdown
        peak = initial_balance
        drawdowns = []
        for balance in equity_curve:
            if balance > peak:
                peak = balance
            drawdown_pct = (peak - balance) / peak * 100
            drawdowns.append(drawdown_pct)

        max_drawdown = max(drawdowns)

        # Afficher les résultats
        print(f"Evaluation Results:", flush=True)
        print(f"Initial Balance: ${initial_balance:.2f}", flush=True)
        print(f"Final Balance: ${final_balance:.2f}", flush=True)
        print(f"Profit: {profit_pct:.2f}%", flush=True)
        print(f"Max Drawdown: {max_drawdown:.2f}%", flush=True)

        # Visualiser les résultats
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 10))

            plt.subplot(2, 1, 1)
            plt.plot(equity_curve)
            plt.title('Equity Curve')
            plt.axhline(y=initial_balance, color='r', linestyle='-')

            plt.subplot(2, 1, 2)
            plt.plot(drawdowns)
            plt.title('Drawdown (%)')

            plt.tight_layout()
            os.makedirs("logs", exist_ok=True)
            plt.savefig("logs/model_evaluation.png")
            print("Evaluation visualization saved to logs/model_evaluation.png", flush=True)
            plt.show()
        except ImportError:
            print("Matplotlib not available for visualization", flush=True)

    except Exception as e:
        print(f"Error during evaluation: {e}", flush=True)
        import traceback
        traceback.print_exc()

def live_trading(model_path, duration_hours=24):
    print(f"Starting live trading with model {model_path} for {duration_hours} hours", flush=True)

    try:
        # Créer l'environnement
        env = TradingEnvironment()
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        print("Live trading environment created", flush=True)

        # Créer et charger l'agent
        agent = DQNAgent(state_size, action_size)
        agent.load(model_path)
        agent.epsilon = 0.01  # Très peu d'exploration en trading réel
        print("Agent loaded with epsilon=0.01", flush=True)

        # Trading en direct
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        print("Starting live trading...", flush=True)

        start_time = time.time()
        end_time = start_time + duration_hours * 3600

        trade_count = 0

        while time.time() < end_time:
            # Choisir une action
            action = agent.act(state)

            # Exécuter l'action
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # Mettre à jour l'état
            state = next_state
            total_reward += reward
            trade_count += 1

            print(f"Trade {trade_count}: Action: {action}, Reward: {reward:.4f}, Total Reward: {total_reward:.4f}", flush=True)

            if 'balance' in info:
                print(f"Current Balance: ${info['balance']:.2f}", flush=True)

            if done:
                print("Trading session completed by environment", flush=True)
                break

            # Attendre avant la prochaine action
            time.sleep(60)  # Attendre 1 minute

            # Afficher le temps restant toutes les 10 minutes
            elapsed = time.time() - start_time
            if trade_count % 10 == 0:
                remaining = duration_hours * 3600 - elapsed
                print(f"Time elapsed: {elapsed/3600:.2f} hours, Remaining: {remaining/3600:.2f} hours", flush=True)

        print(f"Live trading completed. Total trades: {trade_count}, Total reward: {total_reward:.4f}", flush=True)

        # Fermer l'environnement
        env.close()

    except Exception as e:
        print(f"Error during live trading: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL Trading Bot')
    parser.add_argument('--mode', type=int, default=0,
                        help='Mode: 0=menu, 1=live training, 2=historical training, 3=evaluation, 4=live trading')
    parser.add_argument('--data', type=str, default='',
                        help='Path to historical data CSV file')
    parser.add_argument('--model', type=str, default='',
                        help='Path to trained model file')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of episodes for training')
    parser.add_argument('--duration', type=float, default=24,
                        help='Duration for live trading in hours')

    args = parser.parse_args()

    if args.mode == 0:
        # Menu interactif
        print("RL Trading Bot")
        print("1. Train a new agent on live data")
        print("2. Train a new agent on historical data")
        print("3. Evaluate a trained model on historical data")
        print("4. Start live trading with a trained agent")

        choice = input("Enter your choice (1/2/3/4): ")

        if choice == "1":
            episodes = int(input("Enter number of episodes: "))
            agent = train_agent(episodes=episodes)
            print("Training completed!")

        elif choice == "2":
            data_path = input("Enter path to historical data CSV: ")
            episodes = int(input("Enter number of episodes: "))
            agent = train_on_historical_data(data_path, episodes=episodes)
            print("Historical data training completed!")

        elif choice == "3":
            model_path = input("Enter path to the trained model: ")
            data_path = input("Enter path to test data CSV: ")
            evaluate_model(model_path, data_path)
            print("Evaluation completed!")

        elif choice == "4":
            model_path = input("Enter the path to the trained model: ")
            duration = float(input("Enter trading duration in hours: "))
            live_trading(model_path, duration_hours=duration)
            print("Trading completed!")

        else:
            print("Invalid choice!")

    else:
        # Mode ligne de commande
        if args.mode == 1:
            print(f"Starting live training for {args.episodes} episodes", flush=True)
            train_agent(episodes=args.episodes)

        elif args.mode == 2:
            if not args.data:
                print("Error: --data argument is required for historical training", flush=True)
            else:
                print(f"Starting historical training on {args.data} for {args.episodes} episodes", flush=True)
                train_on_historical_data(args.data, episodes=args.episodes)

        elif args.mode == 3:
            if not args.model or not args.data:
                print("Error: --model and --data arguments are required for evaluation", flush=True)
            else:
                print(f"Evaluating model {args.model} on {args.data}", flush=True)
                evaluate_model(args.model, args.data)

        elif args.mode == 4:
            if not args.model:
                print("Error: --model argument is required for live trading", flush=True)
            else:
                print(f"Starting live trading with model {args.model} for {args.duration} hours", flush=True)
                live_trading(args.model, duration_hours=args.duration)