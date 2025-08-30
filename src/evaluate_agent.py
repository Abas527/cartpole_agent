import gymnasium as gym
import torch
import numpy as np
from initialize import DQN, create_environment
from replay_buffer import create_replay_buffer, add_experience, sample_experiences
import random


def evaluate_agent(env_name, model_path, num_episodes=10):
    env = create_environment(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = DQN(state_dim, action_dim)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()

    epsilion=0

    total_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.array(state)
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = np.array(next_state)
            done = done or truncated
            total_reward += reward
            state = next_state
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward


def main():
    env_name="CartPole-v1"
    model_path="dqn_cartpole.pth"
    evaluate_agent(env_name,model_path,20)


if __name__=="__main__":
    main()
