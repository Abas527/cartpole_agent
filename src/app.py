import gymnasium as gym
import torch
import numpy as np
from initialize import DQN, create_environment
import random
import streamlit as st
from PIL import Image
import imageio



def get_frames_from_agent(env_name, model_path, num_episodes=10,render=False):
    env = gym.make(env_name,render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = DQN(state_dim, action_dim)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()

    frames=[]

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

            #storing frames
            if render:
                frame=env.render()
                frames.append(frame)



        total_rewards.append(total_reward)
    
    env.close()
    return total_rewards,frames


def main():
    env_name="CartPole-v1"
    model_path="dqn_cartpole.pth"

    st.write("running agents")

    # reward,frames=get_frames_from_agent(env_name,model_path,20,render=True)

    st.write("Agent playing cartpole with 30fps , 3502 frames and 315s duration")
    # imageio.mimsave("cartpole.gif", frames, fps=30)
    st.image("cartpole.gif")


    st.write("complete")

if __name__=="__main__":
    main()


