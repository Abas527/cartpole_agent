import torch
from torch import nn
from torch import optim
import random
import numpy as np
from initialize import DQN,create_environment
from replay_buffer import create_replay_buffer, add_experience, sample_experiences



def train_cart_pole(env_name,batch_size,learning_rate,gamma,buffer_size,num_episodes,target_update_freq,optimizer_type='Adam'):
    env=create_environment(env_name)
    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.n
    policy_net=DQN(state_dim,action_dim)
    target_net=DQN(state_dim,action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    buffer=create_replay_buffer(buffer_size,state_dim)
    optimizer=optim.Adam(policy_net.parameters(),lr=learning_rate)
    epsilon_start=1.0
    epsilon_end=0.01
    epsilon_decay=500

    index=0
    buffer_filled=0

    for episode in range(num_episodes):
        state,_=env.reset()
        state=np.array(state)
        done=False
        total_reward=0
        i=0
        while not done:
            epsilion=epsilon_end+(epsilon_start-epsilon_end)*np.exp(-1. * episode / epsilon_decay)
            if random.random()<epsilion:
                action=env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor=torch.FloatTensor(state).unsqueeze(0)
                    q_values=policy_net(state_tensor)
                    action=q_values.argmax().item()
            next_state,reward,done,truncated,_=env.step(action)
            next_state=np.array(next_state)
            done=done or truncated
            total_reward+=reward

            index=add_experience(buffer,state,action,reward,next_state,done,index)
            buffer_filled=min(buffer_filled+1,buffer_size)
            state=next_state
            i+=1

            if buffer_filled>=batch_size:
                experience=sample_experiences(buffer,batch_size,max_index=buffer_filled)
                states=torch.FloatTensor(experience['states'])
                actions=torch.LongTensor(experience['actions']).unsqueeze(1)
                rewards=torch.FloatTensor(experience['rewards']).unsqueeze(1)
                next_states=torch.FloatTensor(experience['next_states'])
                dones=torch.FloatTensor(experience['dones'].astype(np.float32)).unsqueeze(1)

                q_values=policy_net(states).gather(1,actions)
                with torch.no_grad():
                    next_q_values=target_net(next_states).max(1,keepdim=True)[0]
                    target_q_values=rewards+gamma*next_q_values*(1-dones)
                loss=nn.MSELoss()(q_values,target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if i%target_update_freq==0:
                target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {episode},Total Reward: {total_reward},epsilion:{epsilion}")
    torch.save(policy_net.state_dict(),'dqn_cartpole.pth')





def main():
    #defining parameters
    env_name = "CartPole-v1"
    batch_size = 64
    gamma = 0.99
    learning_rate = 0.01
    buffer_size = 10000
    num_episodes = 1000
    target_update_freq = 10
    optimizer = 'Adam'
    train_cart_pole(env_name,batch_size,learning_rate,gamma,buffer_size,num_episodes,target_update_freq,optimizer)



if __name__ == "__main__":
    main()