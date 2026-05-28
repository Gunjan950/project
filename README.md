# 🤖 Autonomous Robot Navigation with Reinforcement Learning

A simulation of robot navigation using Q-Learning and DQN on a custom GridWorld environment.

## Project Structure
- `grid_world.py` — Custom OpenAI Gym environment (8×8 grid)
- `q_learning.py` — Tabular Q-Learning agent
- `random_agent.py` — Random policy baseline
- `dqn_agent.py` — Deep Q-Network (bonus)
- `plotting.py` — Reward curve comparison plots

## How to Run
Open the `.ipynb` notebook in Google Colab and run all cells.

## Reward Function
| Event | Reward |
|-------|--------|
| Reach goal | +100 |
| Hit obstacle | -100 |
| Each step | -1 |
| Hit wall | -0.5 |

## Results
| Agent | Success Rate |
|-------|-------------|
| Q-Learning | ~90% |
| DQN | ~80% |
| Random | ~2% |

## Technologies
- Python, NumPy, Matplotlib
- OpenAI Gymnasium
- TensorFlow / Keras (DQN)
