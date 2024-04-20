import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Subtract
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class CarRacingDQNAgent:
    def __init__(self, action_space, frame_stack_num=5, memory_size=5000, gamma=0.8,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9999, learning_rate=0.0025):
        # Initialize the agent with the given parameters
        self.action_space = action_space         # List of possible actions
        self.frame_stack_num = frame_stack_num   # Number of frames to stack as input to the network
        self.memory = deque(maxlen=memory_size)  # Replay memory to hold past experiences
        self.gamma = gamma                       # Discount factor for future rewards
        self.epsilon = epsilon                   # Exploration rate: probability of choosing a random action
        self.epsilon_min = epsilon_min           # Minimum exploration rate
        self.epsilon_decay = epsilon_decay       # Decay rate of exploration over time
        self.learning_rate = learning_rate       # Learning rate for the optimizer
        self.model = self.build_dueling_model()  # Primary network model for Q-learning
        self.target_model = self.build_dueling_model()  # Target network, which lags the primary model
        self.update_target_model()               # Initialize target model weights to match primary model

    def build_dueling_model(self):
        # Builds a dueling DQN architecture
        inputs = Input(shape=(96, 96, self.frame_stack_num))
        x = Conv2D(6, (7, 7), strides=3, activation='relu')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(12, (4, 4), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)

        # Branch to compute action advantage matrix
        action_fc = Dense(216, activation='relu')(x)
        action_output = Dense(len(self.action_space))(action_fc)

        # Branch to compute state value
        value_fc = Dense(216, activation='relu')(x)
        value_output = Dense(1)(value_fc)

        # Combine the above two streams into Q-values for each action
        q_values = Lambda(lambda a: a[1] + (a[0] - tf.reduce_mean(a[0], axis=1, keepdims=True)))([action_output, value_output])
        model = Model(inputs=inputs, outputs=q_values)
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-7))
        return model

    def update_target_model(self):
        # Update target model to match the primary model's weights
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        # Store experience in replay memory
        self.memory.append((state, self.action_space.index(action), reward, next_state, done))

    def act(self, state):
        # Choose an action based on the current state
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_space)  # Explore: choose a random action
        else:
            q_values = self.model.predict(np.expand_dims(state, 0))[0]
            return self.action_space[np.argmax(q_values)]  # Exploit: choose the best known action

    def replay(self, batch_size):
        # Train the model using randomly sampled experiences from the replay memory
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        states, targets = [], []
        for state, action_index, reward, next_state, done in minibatch:
            # Calculate target Q-value for the next state
            target = reward if done else reward + self.gamma * np.amax(self.target_model.predict(np.expand_dims(next_state.astype(np.float16), 0))[0])
            target_f = self.model.predict(np.expand_dims(state, 0))[0]
            target_f[action_index] = target
            states.append(state)
            targets.append(target_f)
        # Update the network with these target values
        self.model.fit(np.array(states), np.array(targets), batch_size=batch_size, epochs=1, verbose=0)
        # Decay the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        # Load weights into the model
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        # Save the model weights
        self.model.save_weights(name)
