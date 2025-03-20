from claude_rl_infra import get_historical_data, DQNAgent, StockTradingEnv, train_agent, testing_agent

# Set random seeds for reproducibility
#np.random.seed(42)
#tf.random.set_seed(42)
#random.seed(42)

# Parameters
LOOKBACK_WINDOW_SIZE = 20
GAMMA = 0.99
#GAMMA = 0.80
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPSILON_INITIAL = 1.0
EPSILON_FINAL = 0.01
#EPSILON_DECAY_STEPS = 1000
EPSILON_DECAY_STEPS = 10000
#EPSILON_DECAY_STEPS = 100
MEMORY_SIZE = 10000
TOTAL_EPISODES = 10
INITIAL_CAPITAL = 10000


# Get MSFT data
print("Fetching MSFT historical data...")
data = get_historical_data('MSFT', period=1)

# Split data for training and testing (80% training, 20% testing)
#split_idx = int(len(data) * 0.8)
#train_data = data.iloc[:split_idx]
#test_data = data.iloc[split_idx-LOOKBACK_WINDOW_SIZE:]  # Include lookback window

# Create training environment
train_env = StockTradingEnv(data, initial_balance=INITIAL_CAPITAL,
                               lookback_window_size=LOOKBACK_WINDOW_SIZE)

# Define state and action sizes
state_size = train_env.observation_space.shape
action_size = train_env.action_space.n

# Create agent
agent = DQNAgent(state_size, action_size, EPSILON_INITIAL,
                 MEMORY_SIZE, EPSILON_FINAL, EPSILON_DECAY_STEPS, GAMMA, LEARNING_RATE)

# Load the trained model
print("loading the model weights...")
agent.load('msft_trading_model.weights.h5')

# Train agent
print("\nTraining the agent...")
#agent.epsilon = EPSILON_FINAL
agent.epsilon = 0.5
train_scores = train_agent(train_env, agent, episodes=40, batch_size=BATCH_SIZE)

testing_agent('msft', agent, data, LOOKBACK_WINDOW_SIZE, INITIAL_CAPITAL)

# Save the trained model
print("saving the model weights...")
agent.save('msft_trading_model.weights.h5')
