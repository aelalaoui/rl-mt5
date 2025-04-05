# main.py (enhanced version)
import numpy as np
import time
import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from src.backtest_environment import BacktestTradingEnvironment
from src.agent import DQNAgent

# Set up TensorFlow logging
tf.get_logger().setLevel('ERROR')

def train_on_historical_data(data_path, episodes=50, batch_size=64,
                             window_size=20,
                             commission_fee=0.0001,  # 0.01% commission
                             slippage_base=0.0001,   # 1 pip base slippage
                             slippage_vol_impact=0.00005,  # additional slippage based on volume
                             bid_ask_spread=0.0002,  # 2 pips spread
                             market_impact_factor=0.0001,  # price impact of large orders
                             liquidity_limit=0.05,   # max 5% of average volume
                             latency_ms=(10, 50),    # latency between 10-50ms
                             use_market_features=True,
                             early_stopping_patience=10):
    """Train the agent on historical data with realistic trading parameters"""
    print(f"Starting historical data training on {data_path}", flush=True)

    try:
        # Create timestamp for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"training_{timestamp}"

        # Create directories for logs and models
        os.makedirs(f"logs/{run_name}", exist_ok=True)
        os.makedirs(f"models/{run_name}", exist_ok=True)

        # Load data
        print("Loading data...", flush=True)
        df = pd.read_csv(data_path)
        print(f"CSV columns: {df.columns.tolist()}", flush=True)
        print(f"First few rows:\n{df.head()}", flush=True)
        print(f"Data shape: {df.shape}", flush=True)

        # Create backtesting environment with realistic parameters
        env = BacktestTradingEnvironment(
            data_path,
            window_size=window_size,
            commission_fee=commission_fee,
            slippage_base=slippage_base,
            slippage_vol_impact=slippage_vol_impact,
            bid_ask_spread=bid_ask_spread,
            market_impact_factor=market_impact_factor,
            liquidity_limit=liquidity_limit,
            latency_ms=latency_ms
        )
        print(f"Environment created. Data size: {len(env.data)} bars", flush=True)

        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        print(f"State size: {state_size}, Action size: {action_size}", flush=True)

        # Create agent
        print("Creating agent...", flush=True)
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            memory_size=10000,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            learning_rate=0.001,
            batch_size=batch_size,
            update_target_every=10,
            use_market_features=use_market_features
        )
        print("Agent created", flush=True)

        # Metrics for tracking
        all_rewards = []
        all_balances = []
        all_trades = []
        episode_trades = []
        best_balance = 0
        patience_counter = 0

        # Training loop
        for e in range(episodes):
            print(f"Starting episode {e+1}/{episodes}", flush=True)

            try:
                state = env.reset()
                state = np.reshape(state, [1, state_size])
                total_reward = 0
                episode_trades = []

                done = False
                time_step = 0
                max_steps = len(env.data) - env.window_size - 1  # Maximum possible steps

                print(f"  Initial state shape: {state.shape}", flush=True)

                while not done and time_step < max_steps:
                    # Choose action
                    action = agent.act(state)

                    if time_step % 1000 == 0:  # Log every 1000 steps
                        print(f"  Step {time_step}, Action: {action}, Balance: {env.balance:.2f}", flush=True)

                    # Execute action
                    next_state, reward, done, info = env.step(action)
                    next_state = np.reshape(next_state, [1, state_size])

                    # Record trade if position changed
                    if info.get('position', 0) != 0 and action > 0:
                        trade_info = {
                            'step': time_step,
                            'action': 'buy' if action == 1 else 'sell',
                            'price': info.get('position_price', 0),
                            'balance': info.get('balance', 0),
                            'reward': reward,
                            'slippage': info.get('slippage', 0),
                            'spread': info.get('spread', 0)
                        }
                        episode_trades.append(trade_info)
                        all_trades.append(trade_info)

                    # Store experience
                    agent.remember(state, action, reward, next_state, done)

                    # Update state
                    state = next_state
                    total_reward += reward

                    # Train agent
                    if len(agent.memory) > batch_size:
                        agent.replay(batch_size)

                    time_step += 1

                if time_step >= max_steps:
                    print(f"Warning: Episode {e+1} reached max steps limit", flush=True)

                print(f"Episode {e+1} completed after {time_step} steps", flush=True)

                # Record metrics
                all_rewards.append(total_reward)
                all_balances.append(info['balance'])

                # Early stopping check
                if info['balance'] > best_balance:
                    best_balance = info['balance']
                    patience_counter = 0
                    # Save best model
                    agent.save(f"models/{run_name}/dqn_agent_best.h5")
                else:
                    patience_counter += 1

                # Print episode summary
                print(f"Episode: {e+1}/{episodes}", flush=True)
                print(f"  Reward: {total_reward:.2f}", flush=True)
                print(f"  Final Balance: {info['balance']:.2f}", flush=True)
                print(f"  Epsilon: {agent.epsilon:.4f}", flush=True)
                print(f"  Trades: {len(episode_trades)}", flush=True)

                # Get agent metrics
                metrics = agent.get_metrics()
                print(f"  Avg Loss: {metrics['avg_loss']:.6f}", flush=True)
                print(f"  Avg Q-Value: {metrics['avg_q_value']:.4f}", flush=True)
                print(f"  Memory Size: {metrics['memory_size']}", flush=True)

                # Save model periodically
                if (e+1) % 10 == 0 or e == episodes - 1:
                    agent.save(f"models/{run_name}/dqn_agent_ep{e+1}.h5")
                    print(f"Model saved at episode {e+1}", flush=True)

                    # Save training progress
                    progress_data = {
                        'episode': list(range(1, e+2)),
                        'reward': all_rewards,
                        'balance': all_balances
                    }
                    pd.DataFrame(progress_data).to_csv(f"logs/{run_name}/training_progress.csv", index=False)

                # Check for early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {e+1} episodes (no improvement for {early_stopping_patience} episodes)", flush=True)
                    break

            except Exception as exc:
                print(f"Error during episode {e+1}: {exc}", flush=True)
                import traceback
                traceback.print_exc()
                continue

        # Visualize results
        try:
            plt.figure(figsize=(15, 12))

            plt.subplot(3, 1, 1)
            plt.plot(all_rewards)
            plt.title('Rewards per Episode')
            plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.plot(all_balances)
            plt.title('Final Balance per Episode')
            plt.grid(True)

            # Plot agent metrics
            plt.subplot(3, 1, 3)
            plt.plot(agent.train_loss_history, label='Loss')
            plt.plot(agent.q_value_history, label='Q-Value')
            plt.title('Agent Metrics')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f"logs/{run_name}/training_results.png")
            print(f"Results visualization saved to logs/{run_name}/training_results.png", flush=True)

            # Save trade history
            trade_df = pd.DataFrame(all_trades)
            if not trade_df.empty:
                trade_df.to_csv(f"logs/{run_name}/trade_history.csv", index=False)

                # Plot trade distribution
                plt.figure(figsize=(12, 6))
                plt.hist(trade_df['reward'], bins=50)
                plt.title('Trade Reward Distribution')
                plt.grid(True)
                plt.savefig(f"logs/{run_name}/trade_distribution.png")

        except Exception as e:
            print(f"Error during visualization: {e}", flush=True)

        return agent, run_name

    except Exception as e:
        print(f"Critical error in training: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None, None

def evaluate_model(model_path, data_path,
                  window_size=20,
                  commission_fee=0.0001,
                  slippage_base=0.0001,
                  slippage_vol_impact=0.00005,
                  bid_ask_spread=0.0002,
                  market_impact_factor=0.0001,
                  liquidity_limit=0.05,
                  latency_ms=(10, 50),
                  use_market_features=True):
    """Evaluate a trained model on historical data with realistic parameters"""
    print(f"Evaluating model {model_path} on {data_path}", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_name = f"evaluation_{timestamp}"
    os.makedirs(f"logs/{eval_name}", exist_ok=True)

    try:
        # Create backtesting environment with realistic parameters
        env = BacktestTradingEnvironment(
            data_path,
            window_size=window_size,
            commission_fee=commission_fee,
            slippage_base=slippage_base,
            slippage_vol_impact=slippage_vol_impact,
            bid_ask_spread=bid_ask_spread,
            market_impact_factor=market_impact_factor,
            liquidity_limit=liquidity_limit,
            latency_ms=latency_ms
        )
        print(f"Evaluation environment created. Data size: {len(env.data)} bars", flush=True)

        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        # Create and load agent
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            use_market_features=use_market_features
        )
        agent.load(model_path)
        agent.epsilon = 0.01  # Small exploration for robustness
        print("Agent loaded with epsilon=0.01", flush=True)

        # Metrics for tracking
        actions_taken = []
        equity_curve = []
        trade_history = []
        position_durations = []
        current_position_start = None

        # Evaluate agent
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        print("Starting evaluation...", flush=True)

        done = False
        time_step = 0
        max_steps = len(env.data) - env.window_size - 1

        while not done and time_step < max_steps:
            # Choose action
            action = agent.act(state, evaluation=True)
            actions_taken.append(action)

            if time_step % 1000 == 0:  # Log every 1000 steps
                print(f"  Step {time_step}, Action: {action}, Balance: {env.balance:.2f}", flush=True)

            # Execute action
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # Track position duration
            if info['position'] != 0 and current_position_start is None:
                current_position_start = time_step
            elif info['position'] == 0 and current_position_start is not None:
                position_durations.append(time_step - current_position_start)
                current_position_start = None

            # Record trade if position changed
            if action > 0:  # Buy or sell action
                trade_info = {
                    'step': time_step,
                    'action': 'buy' if action == 1 else 'sell',
                    'price': info.get('position_price', 0),
                    'balance': info.get('balance', 0),
                    'reward': reward,
                    'pnl': info.get('pnl', 0),
                    'slippage': info.get('slippage', 0),
                    'spread': info.get('spread', 0),
                    'bid': info.get('bid', 0),
                    'ask': info.get('ask', 0)
                }
                trade_history.append(trade_info)

            # Update state
            state = next_state
            equity_curve.append(info['balance'])

            time_step += 1

        if time_step >= max_steps:
            print(f"Warning: Evaluation reached max steps limit", flush=True)

        print(f"Evaluation completed after {time_step} steps", flush=True)

        # Calculate performance metrics
        initial_balance = 10000.0
        final_balance = equity_curve[-1]
        profit_pct = (final_balance / initial_balance - 1) * 100

        # Calculate drawdown
        peak = initial_balance
        drawdowns = []
        for balance in equity_curve:
            if balance > peak:
                peak = balance
            drawdown_pct = (peak - balance) / peak * 100
            drawdowns.append(drawdown_pct)

        max_drawdown = max(drawdowns) if drawdowns else 0

        # Calculate Sharpe ratio (simplified)
        if len(equity_curve) > 1:
            daily_returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
        else:
            sharpe_ratio = 0

        # Calculate win rate
        if trade_history:
            profitable_trades = sum(1 for trade in trade_history if trade['reward'] > 0)
            win_rate = profitable_trades / len(trade_history) * 100
        else:
            win_rate = 0

        # Display results
        print(f"Evaluation Results:", flush=True)
        print(f"Initial Balance: ${initial_balance:.2f}", flush=True)
        print(f"Final Balance: ${final_balance:.2f}", flush=True)
        print(f"Profit: {profit_pct:.2f}%", flush=True)
        print(f"Max Drawdown: {max_drawdown:.2f}%", flush=True)
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}", flush=True)
        print(f"Win Rate: {win_rate:.2f}%", flush=True)
        print(f"Total Trades: {len(trade_history)}", flush=True)

        if position_durations:
            print(f"Avg Position Duration: {np.mean(position_durations):.2f} bars", flush=True)

        # Save results to CSV
        results = {
            'metric': ['Initial Balance', 'Final Balance', 'Profit (%)', 'Max Drawdown (%)',
                      'Sharpe Ratio', 'Win Rate (%)', 'Total Trades', 'Avg Position Duration'],
            'value': [initial_balance, final_balance, profit_pct, max_drawdown,
                     sharpe_ratio, win_rate, len(trade_history),
                     np.mean(position_durations) if position_durations else 0]
        }
        pd.DataFrame(results).to_csv(f"logs/{eval_name}/evaluation_results.csv", index=False)

        # Save trade history
        if trade_history:
            pd.DataFrame(trade_history).to_csv(f"logs/{eval_name}/trade_history.csv", index=False)

        # Save equity curve
        equity_df = pd.DataFrame({
            'step': range(len(equity_curve)),
            'balance': equity_curve,
            'drawdown': drawdowns
        })
        equity_df.to_csv(f"logs/{eval_name}/equity_curve.csv", index=False)

        # Visualize results
        try:
            plt.figure(figsize=(15, 15))

            plt.subplot(3, 1, 1)
            plt.plot(equity_curve)
            plt.title('Equity Curve')
            plt.axhline(y=initial_balance, color='r', linestyle='-')
            plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.plot(drawdowns)
            plt.title('Drawdown (%)')
            plt.grid(True)

            # Plot action distribution
            plt.subplot(3, 1, 3)
            action_counts = np.bincount(actions_taken, minlength=3)
            plt.bar(['Hold', 'Buy', 'Sell'], action_counts)
            plt.title('Action Distribution')
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f"logs/{eval_name}/evaluation_results.png")
            print(f"Evaluation visualization saved to logs/{eval_name}/evaluation_results.png", flush=True)

            # Additional visualizations
            if trade_history:
                # Plot trade outcomes
                plt.figure(figsize=(12, 10))

                plt.subplot(2, 1, 1)
                trade_rewards = [t['reward'] for t in trade_history]
                plt.hist(trade_rewards, bins=20)
                plt.title('Trade Reward Distribution')
                plt.grid(True)

                plt.subplot(2, 1, 2)
                trade_steps = [t['step'] for t in trade_history]
                trade_types = [1 if t['action'] == 'buy' else -1 for t in trade_history]
                plt.scatter(trade_steps, trade_types, c=[1 if t['reward'] > 0 else 0 for t in trade_history],
                           cmap='coolwarm', alpha=0.7)
                plt.title('Trade Timing (Green=Profit, Red=Loss)')
                plt.yticks([-1, 1], ['Sell', 'Buy'])
                plt.grid(True)

                plt.tight_layout()
                plt.savefig(f"logs/{eval_name}/trade_analysis.png")

        except Exception as e:
            print(f"Error during visualization: {e}", flush=True)
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"Error during evaluation: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL Trading Bot with Realistic Parameters')
    parser.add_argument('--mode', type=int, default=0,
                        help='Mode: 0=menu, 1=historical training, 2=evaluation')
    parser.add_argument('--data', type=str, default='',
                        help='Path to historical data CSV file')
    parser.add_argument('--model', type=str, default='',
                        help='Path to trained model file')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of episodes for training')
    parser.add_argument('--window', type=int, default=20,
                        help='Window size for observations')
    parser.add_argument('--batch', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--commission', type=float, default=0.0001,
                        help='Commission fee (as decimal)')
    parser.add_argument('--spread', type=float, default=0.0002,
                        help='Bid-ask spread (as decimal)')
    parser.add_argument('--slippage', type=float, default=0.0001,
                        help='Base slippage (as decimal)')

    args = parser.parse_args()

    if args.mode == 0:
        # Interactive menu
        print("RL Trading Bot with Realistic Parameters")
        print("1. Train a new agent on historical data")
        print("2. Evaluate a trained model on historical data")

        choice = input("Enter your choice (1/2): ")

        if choice == "1":
            data_path = input("Enter path to historical data CSV: ")
            episodes = int(input("Enter number of episodes: "))
            window_size = int(input("Enter window size (default 20): ") or "20")
            batch_size = int(input("Enter batch size (default 64): ") or "64")
            commission = float(input("Enter commission fee (default 0.0001): ") or "0.0001")
            spread = float(input("Enter bid-ask spread (default 0.0002): ") or "0.0002")
            slippage = float(input("Enter base slippage (default 0.0001): ") or "0.0001")

            agent, run_name = train_on_historical_data(
                data_path,
                episodes=episodes,
                batch_size=batch_size,
                window_size=window_size,
                commission_fee=commission,
                bid_ask_spread=spread,
                slippage_base=slippage
            )
            print("Historical data training completed!")
            if run_name:
                print(f"Results saved in logs/{run_name}/ and models/{run_name}/")

        elif choice == "2":
            model_path = input("Enter path to the trained model: ")
            data_path = input("Enter path to test data CSV: ")
            window_size = int(input("Enter window size (default 20): ") or "20")
            commission = float(input("Enter commission fee (default 0.0001): ") or "0.0001")
            spread = float(input("Enter bid-ask spread (default 0.0002): ") or "0.0002")
            slippage = float(input("Enter base slippage (default 0.0001): ") or "0.0001")

            evaluate_model(
                model_path,
                data_path,
                window_size=window_size,
                commission_fee=commission,
                bid_ask_spread=spread,
                slippage_base=slippage
            )
            print("Evaluation completed!")

        else:
            print("Invalid choice!")

    else:
        # Command line mode
        if args.mode == 1:
            if not args.data:
                print("Error: --data argument is required for historical training", flush=True)
            else:
                print(f"Starting historical training on {args.data} for {args.episodes} episodes", flush=True)
                train_on_historical_data(
                    args.data,
                    episodes=args.episodes,
                    batch_size=args.batch,
                    window_size=args.window,
                    commission_fee=args.commission,
                    bid_ask_spread=args.spread,
                    slippage_base=args.slippage
                )

        elif args.mode == 2:
            if not args.model or not args.data:
                print("Error: --model and --data arguments are required for evaluation", flush=True)
            else:
                print(f"Evaluating model {args.model} on {args.data}", flush=True)
                evaluate_model(
                    args.model,
                    args.data,
                    window_size=args.window,
                    commission_fee=args.commission,
                    bid_ask_spread=args.spread,
                    slippage_base=args.slippage
                )