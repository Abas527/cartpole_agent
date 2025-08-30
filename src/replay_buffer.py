import numpy as np
from collections import deque

def create_replay_buffer(buffer_size, state_dim):
    return {
        'states': np.zeros((buffer_size, state_dim)),
        'actions': np.zeros(buffer_size),
        'rewards': np.zeros(buffer_size),
        'next_states': np.zeros((buffer_size, state_dim)),
        'dones': np.zeros(buffer_size, dtype=bool),
    }
    #states are x,velocity(x),angle theta,angular velocity(theta)
    #actions are left(0) and right(1)
    #reawrd is +1 for every timestamp the pole is upright
    #done is True when the pole is more than 12 degrees from vertical or cart

def add_experience(buffer, state, action, reward, next_state, done, index):
    index = index % len(buffer['states']) 
    buffer['states'][index] = state
    buffer['actions'][index] = int(action)
    buffer['rewards'][index] = reward
    buffer['next_states'][index] = next_state
    buffer['dones'][index] = done
    return index + 1

def sample_experiences(buffer, batch_size,max_index=None):
    if max_index is None:
        max_index = len(buffer['states'])
    indices = np.random.choice(max_index, batch_size, replace=False)
    return {
        'states': buffer['states'][indices],
        'actions': buffer['actions'][indices],
        'rewards': buffer['rewards'][indices],
        'next_states': buffer['next_states'][indices],
        'dones': buffer['dones'][indices],
    }


def main():
    pass

if __name__ == "__main__":
    main()