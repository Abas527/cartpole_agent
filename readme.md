# CartPole DQN Agent

A Deep Q-Network (DQN) implementation for solving the classic CartPole-v1 reinforcement learning environment using PyTorch and Gymnasium.

---

## Project Structure

```
cart_pole/
│
├── agent/
│   ├── cartpole.gif           # Demo animation of the trained agent
│   └── dqn_cartpole.pth       # Saved model weights
│
├── src/
│   ├── app.py                 # Streamlit app for interactive demo/visualization
│   ├── evaluate_agent.py      # Script to evaluate a trained agent
│   ├── initialize.py          # Model architecture and initialization
│   ├── replay_buffer.py       # Experience replay buffer implementation
│   └── train.py               # Training script for DQN agent
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

##  Installation

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd cart_pole
   ```

2. **Create and activate a virtual environment (recommended):**
   ```sh
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

---

##  Training

To train the DQN agent on CartPole-v1:

```sh
python src/train.py
```

- The trained model will be saved as dqn_cartpole.pth.
- Training progress and rewards per episode will be printed to the console.

---

##  Evaluation

To evaluate the trained agent:

```sh
python src/evaluate_agent.py
```

- This will load the saved model and run it in the CartPole environment.
- You can modify `evaluate_agent.py` to render or save videos.

---

##  Visualization & Demo

To run the interactive Streamlit app (for demo, visualization, or analysis):

```sh
streamlit run src/app.py
```

---

##  Files & Modules

- **initialize.py**: Defines the DQN neural network architecture.
- **replay_buffer.py**: Implements the experience replay buffer.
- **train.py**: Main training loop for the DQN agent.
- **evaluate_agent.py**: Script to evaluate and visualize the trained agent.
- **app.py**: Streamlit app for interactive exploration.
- **cartpole.gif**: Example animation of the trained agent.
- **dqn_cartpole.pth**: Saved PyTorch model weights.

---

## 🛠️ Requirements

- Python 3.8+
- See requirements.txt for all dependencies:
  - gymnasium
  - torch
  - numpy
  - matplotlib
  - pandas
  - scikit-learn
  - streamlit
  - imageio

---

##  References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/)
- [DQN Paper (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)

---

##  License

This project is for educational and research purposes.

---

##  Acknowledgements

- OpenAI Gym / Gymnasium for the CartPole environment.
- PyTorch for deep learning framework.

---

**Enjoy experimenting with reinforcement learning!**