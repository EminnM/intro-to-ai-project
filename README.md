# Emin Merden Introduction to Artificial Int. Final Project

## Evolution Video


---

https://github.com/user-attachments/assets/004f2b14-001c-44bc-8c2d-b447d76a2c15


## Environment

All experiments are conducted in the `highway-fast-v0` environment from the `highway-env` benchmark suite.  
The task requires an autonomous vehicle to drive at high speed on a multi-lane highway while avoiding collisions and unnecessary lane changes.

Each episode terminates either upon collision or after a fixed horizon of 200 steps.

---

## Reward Function

The reward function is explicitly shaped to encourage fast, stable, and safe driving behavior.

The total reward is defined as:

R_total = R_speed + R_lane + R_collision

where:

- **Speed Reward (R_speed):**  
  Encourages the agent to maintain a velocity within a predefined optimal range.  
  The speed reward range is set to [25,35] with a reward coefficient of 10. Speeds above 35 tend to result in frequent collisions, while 25 represents the approximate average traffic speed. This range encourages the agent to drive faster than average without sacrificing safety.

- **Lane Change Penalty (R_lane):**  
  A fixed negative reward (-0.25) applied whenever the agent changes lanes, discouraging unnecessary oscillations.

- **Collision Penalty (R_collision):**  
  A large negative reward (-10) applied when a collision occurs, immediately terminating the episode.

Reward normalization is disabled to preserve interpretability of individual reward components.

---

## Agent Architecture

The agent is trained using **Deep Q-Learning (DQN)** with a multilayer perceptron policy.

**Network Architecture:**
- Fully connected MLP
- Two hidden layers
- 256 units per layer
- ReLU activations

This architecture balances representational capacity and training stability for discrete action control.

---

## Training Details

The agent is trained with the following hyperparameters:

| Parameter | Value |
|---------|------|
| Algorithm | DQN |
| Learning Rate | 1e-3 |
| Discount Factor (γ) | 0.99 |
| Replay Buffer Size | 50,000 |
| Batch Size | 32 |
| Target Network Update | Every 500 steps |
| Exploration ε (initial) | 0.2 |
| Exploration Fraction | 0.05 |

Training progress is monitored using custom callbacks that log:
- Average episode speed
- Reward component breakdown
- Collision frequency

Intermediate models are saved at 1% and 10% of total training steps.

---

## Evaluation

Evaluation is performed using a deterministic policy without exploration.  
Qualitative assessment is performed by visually inspecting recorded agent behavior, while quantitative evaluation is based on TensorBoard logs capturing metrics such as average speed, collision penalties, lane-change penalties, and total episode reward.

As the figure below shows, episode reward exhibits an upward trend over training, indicating that the agent progressively improves its policy.

<img width="1225" height="449" alt="image" src="https://github.com/user-attachments/assets/533e639d-c887-4858-8fec-0568c6db2c8e" />

Likewise, the speed reward shows a clear upward trend, indicating that the agent progressively learns to maintain higher velocities within the predefined safe range. This confirms that the increase in total reward is primarily driven by improved speed control rather than incidental reward accumulation. Importantly, this increase does not coincide with a rise in collision penalties, suggesting that the agent successfully balances speed maximization with safety constraints.

<img width="1225" height="448" alt="image" src="https://github.com/user-attachments/assets/2ac5a298-c209-4291-b597-241fc93b7313" />

The collision reward oscillates around −2 to −5 throughout training. This indicates that while collisions are not entirely eliminated, their frequency does not increase as the agent learns to drive faster. In other words, performance gains in speed are achieved without a corresponding degradation in safety, suggesting a stable trade-off between velocity and collision avoidance.

<img width="1216" height="424" alt="image" src="https://github.com/user-attachments/assets/323b0f84-ebe1-4178-93b0-814ba985d359" />


---

## Usage

### Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Training
```bash
python main.py --mode train --timesteps 100000
```

This command:

Trains the DQN agent using ε-greedy exploration

Logs metrics to TensorBoard

Saves intermediate checkpoints and the final model

To visualize training metrics:

```bash
tensorboard --logdir highway_dqn/
```

### Evaluation
```bash
python main.py --mode eval --model-path models/model_100000 --episodes 5
```


