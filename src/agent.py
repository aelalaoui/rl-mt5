# src/agent.py (enhanced version)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import random
from collections import deque
import os
import time

class DQNAgent:
    def __init__(self, state_size, action_size, 
                 memory_size=10000,
                 gamma=0.95, 
                 epsilon=1.0, 
                 epsilon_min=0.01, 
                 epsilon_decay=0.995,
                 learning_rate=0.001,
                 batch_size=64,
                 update_target_every=10,
                 use_market_features=True):
        
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.use_market_features = use_market_features
        
        # Performance tracking
        self.train_loss_history = []
        self.q_value_history = []
        self.reward_history = []
        self.action_history = []
        self.train_count = 0
        
        # Build models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Log directory for TensorBoard
        self.log_dir = "logs/dqn_agent_" + time.strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

    def _build_model(self):
        """Neural network for approximating the Q-function"""
        if self.use_market_features:
            # More sophisticated model with market features
            # Price data input
            price_input = Input(shape=(self.state_size - 3,), name='price_data')
            price_dense = Dense(32, activation='relu')(price_input)
            price_dense = Dense(16, activation='relu')(price_dense)
            
            # Market features input (position, RSI, MA)
            market_input = Input(shape=(3,), name='market_features')
            market_dense = Dense(8, activation='relu')(market_input)
            
            # Combine both inputs
            combined = Concatenate()([price_dense, market_dense])
            dense = Dense(24, activation='relu')(combined)
            dense = Dropout(0.2)(dense)
            dense = Dense(12, activation='relu')(dense)
            output = Dense(self.action_size, activation='linear')(dense)
            
            model = Model(inputs=[price_input, market_input], outputs=output)
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            return model
        else:
            # Simple model
            model = Sequential()
            model.add(Dense(24, input_dim=self.state_size, activation='relu'))
            model.add(Dense(24, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            return model

    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
        print("Target model updated", flush=True)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
        
        # Log reward for tracking
        self.reward_history.append(reward)
        self.action_history.append(action)

    def act(self, state, evaluation=False):
        """Choose action using epsilon-greedy policy"""
        # Initialize act_values to None
        act_values = None

        if not evaluation and np.random.rand() <= self.epsilon:
            # For random actions, we won't have Q-values
            action = random.randrange(self.action_size)
            # Store a placeholder value for random actions
            self.q_value_history.append(0.0)
            return action

        # Prepare state data for prediction
        if self.use_market_features:
            # Split state into price data and market features
            price_data = state[0][:-3]
            market_features = state[0][-3:]  # Get the last 3 elements

            # Reshape for model input
            price_data = np.reshape(price_data, [1, self.state_size - 3])
            market_features = np.reshape(market_features, [1, 3])

            # Use the model to predict with both inputs
            # Note: This needs to match your model architecture
            act_values = self.model.predict([price_data, market_features], verbose=0)
        else:
            # Standard prediction with single input
            act_values = self.model.predict(state, verbose=0)

        # Track Q-values for analysis
        self.q_value_history.append(np.max(act_values[0]))

        return np.argmax(act_values[0])

    def replay(self, batch_size=None):
        """Train the model with experiences from memory"""
        if batch_size is None:
            batch_size = self.batch_size
            
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        if self.use_market_features:
            # Prepare separate arrays for price data and market features
            price_data_states = np.zeros((batch_size, self.state_size - 3))
            market_feature_states = np.zeros((batch_size, 3))
            price_data_next_states = np.zeros((batch_size, self.state_size - 3))
            market_feature_next_states = np.zeros((batch_size, 3))
            
            # Extract data from minibatch
            for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                price_data_states[i] = state[0, :-3]
                market_feature_states[i] = state[0, -3:]
                price_data_next_states[i] = next_state[0, :-3]
                market_feature_next_states[i] = next_state[0, -3:]
            
            # Predict current Q-values and next Q-values
            current_qs = self.model.predict([price_data_states, market_feature_states], verbose=0)
            next_qs = self.target_model.predict([price_data_next_states, market_feature_next_states], verbose=0)
            
            # Update Q-values for actions taken
            for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                if done:
                    current_qs[i, action] = reward
                else:
                    current_qs[i, action] = reward + self.gamma * np.amax(next_qs[i])
            
            # Train the model
            history = self.model.fit(
                [price_data_states, market_feature_states], 
                current_qs, 
                epochs=1, 
                verbose=0
            )
        else:
            # Simple model training
            states = np.zeros((batch_size, self.state_size))
            next_states = np.zeros((batch_size, self.state_size))
            
            for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                states[i] = state
                next_states[i] = next_state
            
            # Predict current Q-values and next Q-values
            current_qs = self.model.predict(states, verbose=0)
            next_qs = self.target_model.predict(next_states, verbose=0)
            
            # Update Q-values for actions taken
            for i, (state, action, reward, next_state, done) in enumerate(minibatch):
                if done:
                    current_qs[i, action] = reward
                else:
                    current_qs[i, action] = reward + self.gamma * np.amax(next_qs[i])
            
            # Train the model
            history = self.model.fit(states, current_qs, epochs=1, verbose=0)
        
        # Track loss
        self.train_loss_history.append(history.history['loss'][0])
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Update target model periodically
        self.train_count += 1
        if self.train_count % self.update_target_every == 0:
            self.update_target_model()
            
        # Log metrics to TensorBoard
        with self.summary_writer.as_default():
            tf.summary.scalar('loss', history.history['loss'][0], step=self.train_count)
            tf.summary.scalar('epsilon', self.epsilon, step=self.train_count)
            tf.summary.scalar('avg_q_value', np.mean(self.q_value_history[-100:]), step=self.train_count)
            tf.summary.scalar('avg_reward', np.mean(self.reward_history[-100:]), step=self.train_count)

    def load(self, name):
        """Load model weights"""
        self.model.load_weights(name)
        self.update_target_model()
        print(f"Model loaded from {name}", flush=True)

    def save(self, name):
        """Save model weights"""
        os.makedirs(os.path.dirname(name), exist_ok=True)
        self.model.save_weights(name)
        print(f"Model saved to {name}", flush=True)
        
    def get_metrics(self):
        """Return agent performance metrics"""
        return {
            'avg_loss': np.mean(self.train_loss_history[-100:]) if self.train_loss_history else 0,
            'avg_q_value': np.mean(self.q_value_history[-100:]) if self.q_value_history else 0,
            'avg_reward': np.mean(self.reward_history[-100:]) if self.reward_history else 0,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory)
        }