# OpenAI GYM CarRacing DQN

Training machines to play CarRacing 2d from OpenAI GYM by implementing Deep Q Learning/Deep Q Network(DQN) with TensorFlow and Keras as the backend.

### Training Results
We can see that the scores(time frames elapsed) stop rising after around 500 episodes as well as the rewards. Thus let's terminate the training and evaluate the model using the last three saved weight files `trial_400.h5`, `trial_500.h5`, and `trial_600.h5`.
<br>
<img src="resources/training_results.png" width="600px">

#### Training After 400 Episodes
The model knows it should follow the track to acquire rewards after training 400 episodes, and it also knows how to take short cuts. However, making a sharp right turn still seems difficult to it, which results in getting stuck out of the track.
<br>
<img src="resources/trial_400.gif" width="300px">

#### Training After 500 Episodes
The model can now drive faster and smoother after training 500 episodes with making less mistakes.
<br>
<img src="resources/trial_500.gif" width="300px">

#### Training After 600 Episodes
To acquire more rewards greedily, the model has gone bad that learns how to drive recklessly and thus making it going off the track when reaching sharp turns.
<br>
<img src="resources/trial_600.gif" width="300px">

## Usage


### Train the Deep Q Network(DQN)
```
python train_model.py [-m save/trial_XXX.h5] [-s 1] [-e 1000] [-p 1.0]
```
- `-m` The path to the trained model if you wish to continue training after it.
- `-s` The starting training episode, default to 1.
- `-e` The ending training episode, default to 1000.
- `-p` The starting epsilon of the agent, default to 1.0.

### DQN Agent
After having the DQN model trained, let's see how well did the model learned about playing CarRacing.
```
python play_car_racing_by_the_model.py
```

## File Structure

- `train_model.py` The training program.
- `common_functions.py` Some functions that will be used in multiple programs will be put in here.
- `CarRacingDQNAgent.py` The core DQN class. Anything related to the model is placed in here.
- `play_car_racing_by_the_model.py` The program for playing CarRacing by the model.
- `save/` The default folder to save the trained model.

## Details Explained
Deep Q Learning/Deep Q Network(DQN) is just a variation of Q Learning. It makes the neural network act like the Q table in Q Learning thus avoiding creating an unrealistic huge Q table containing Q values for every state and action.

### Q Value
Q value is the expected rewards given by taking the specific action during the specific state.
In a more mathematical saying, Q value can be written as:

> Q(s,a) = r(s,a) + γ(maxQ(s',A))

- `s` is the current state
- `s'` is the next future state
- `a` is the particular action
- `A` is the action space
- `Q(s,a)` is the Q value given by taking the action `a` during the state `s`
- `r(s,a)` is the rewards given by taking the action `a` during the state `s`
- `maxQ(s',A)` is the maximum Q value given by taking any action in the action space `A` during the state `s'`
- `γ` is the discount rate that will discount the future Q value because the future Q value is less important

The Q value given the state `s` and the action `a` is the sum of the rewards given the state `s` and the action `a` and the maximum Q value given any action in the action space and the next state `s'` multiplied by the discount rate.

Therefore, we should always choose the action with the highest Q value to maximize our rewards.

### The DQN Structure
1. Input Layer:

- Shape: (96, 96, 3)
- **Description**: The input consists of 3 consecutive grayscale top-view images of the game environment. Each image is 96x96 pixels. These images are stacked in the channel dimension, which typically holds RGB data in color images.
2. Convolutional Layers:

- First Convolutional Layer:
    - **Filters**: 6
    - **Kernel Size**: 7x7
    - **Strides**: 3
    - **Activation**: ReLU
    This layer captures the initial set of features from the input image stack with a relatively large field of view due to its kernel size.
- Second Convolutional Layer:
    - **Filters**: 12
    - **Kernel Size**: 4x4
    - **Activation**: ReLU
    This layer further refines the features extracted by the first layer, focusing on smaller and more specific features.
3. Max Pooling Layers:

    - Positioned after each convolutional layer with a pool size of 2x2, these layers reduce the spatial dimensions of the feature maps. This operation helps in reducing the computational load and improving the robustness of the feature extraction by retaining only the most salient features.
4. Flattening Layer:

    Converts the 2D feature maps into a 1D feature vector, preparing them for the fully connected (dense) layers. This is essential for the transition from convolutional processing to classification or, in this case, regression of Q values.
5. Dense Layers (Fully Connected Layers):

    - First Dense Layer:
        - **Units**: 216
        - **Activation**: ReLU
        Connected to the flattened output, this layer is used for high-level reasoning from the features extracted by the convolutional layers.
        Second Dense Layer (Action Output):
            - **Units**: Varies based on the number of actions (len(self.action_space))
        Calculates a potential Q value for each action in the game.
        - Alternate Path Dense Layers (Value Output):
            - Units: 216 and 1 for two layers
        The first layer mirrors the action path with 216 units followed by a single unit output layer. This path estimates the overall value of the state.
6. Lambda Layer:

    - Combines the outputs of the action and value layers using a custom function. This function adjusts the action outputs by subtracting the mean action value (a baseline) and adding the state value. This technique is related to the advantage function used in reinforcement learning to stabilize training by reducing the variance of the action values.
7. Model Compilation:

    - **Loss Function**: Mean Squared Error
    - **Optimizer**: Adam with a specific learning rate and a small epsilon value to improve numerical stability.
<br><br>

8. Model Architecture
<img src="resources\model_Architecture.PNG" width="400px">

### How Self-Driving Works
- During the game, the program will stack the latest 3 states of 96x96 pixels grayscale images and feed them to the trained model.
- The model produces the Q values for the 12 actions.
- Choose the action that has the highest Q value for the agent to perform.
- As long as the model is well trained, the action that has the highest Q value should be the best action(could obtain the most rewards) that the agent should react to the environment.
- Repeat the steps above so that the car is now self-driving:)

