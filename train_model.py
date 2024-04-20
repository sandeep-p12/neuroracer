import argparse
import gym
from collections import deque
from CarRacingDQNAgent import CarRacingDQNAgent
from common_functions import process_state_image, generate_state_frame_stack_from_queue

# Command-line argument setup for configuration
parser = argparse.ArgumentParser(description='Train a DQN agent to play CarRacing.')
parser.add_argument('-m', '--model', type=str, help='Path to the last trained model to continue training.')
parser.add_argument('-s', '--start', type=int, default=1, help='Starting episode number.')
parser.add_argument('-e', '--end', type=int, default=1000, help='Ending episode number.')
parser.add_argument('-p', '--epsilon', type=float, default=1.0, help='Starting exploration rate (epsilon).')
args = parser.parse_args()

# Define the action space for the CarRacing game
action_space = [
    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2),  # Steering left, straight, right with gas and some brake
    (-1, 1, 0), (0, 1, 0), (1, 1, 0),         # Steering left, straight, right with full gas
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2),   # Steering left, straight, right with some brake
    (-1, 0, 0), (0, 0, 0), (1, 0, 0)           # Steering left, straight, right with no gas or brake
]

# Configuration settings
RENDER = True                                 # Flag to turn on rendering
SKIP_FRAMES = 4                               # Number of frames to skip for faster training
TRAINING_BATCH_SIZE = 32                      # Batch size for training
SAVE_TRAINING_FREQUENCY = 100                 # Frequency of saving the model
UPDATE_TARGET_MODEL_FREQUENCY = 10            # Frequency of updating the target model

# Environment setup
env = gym.make('CarRacing-v2', render_mode="rgb_array")
agent = CarRacingDQNAgent(action_space=action_space, epsilon=args.epsilon)

# Load a previously trained model if provided
if args.model:
    agent.load(args.model)

# Main training loop
for episode in range(args.start, args.end + 1):
    state = process_state_image(env.reset())  # Reset the environment for a new episode
    total_reward = 0
    negative_reward_counter = 0
    state_frame_stack_queue = deque([state] * agent.frame_stack_num, maxlen=agent.frame_stack_num)
    time_frame_counter = 0
    done = False

    while not done:
        if RENDER:
            env.render()  # Render the environment on screen

        state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
        action = agent.act(state_frame_stack)
        reward = 0

        # Execute the chosen action and accumulate rewards from skipped frames
        for _ in range(SKIP_FRAMES + 1):
            next_state, r, done, info, _ = env.step(action)
            reward += r
            if done:
                break

        # Check for sustained negative rewards
        if time_frame_counter > 100 and reward < 0:
            negative_reward_counter += 1

        # Apply a reward bonus for aggressive driving (full gas, no brake)
        if action[1] == 1 and action[2] == 0:
            reward *= 1.5

        total_reward += reward
        next_state = process_state_image(next_state)
        state_frame_stack_queue.append(next_state)
        
        # Memorize the current transition for future training
        agent.memorize(state_frame_stack, action, reward, generate_state_frame_stack_from_queue(state_frame_stack_queue), done)

        # Train the model if enough memories are collected
        if len(agent.memory) > TRAINING_BATCH_SIZE:
            agent.replay(TRAINING_BATCH_SIZE)
        
        time_frame_counter += 1

        # Check for end conditions
        if done or negative_reward_counter >= 25:
            print(f'Episode: {episode}/{args.end}, Time Frames: {time_frame_counter}, Total Rewards: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}')
            break

    # Update the model weights to the target model at set intervals
    if episode % UPDATE_TARGET_MODEL_FREQUENCY == 0:
        agent.update_target_model()

    # Save the model at set intervals
    if episode % SAVE_TRAINING_FREQUENCY == 0:
        agent.save(f'./models/trial_{episode}.h5')

# Clean up the environment
env.close()
