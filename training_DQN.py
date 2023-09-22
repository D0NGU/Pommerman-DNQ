import matplotlib.pyplot as plt
import numpy as np
from math import exp 
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
import pommerman
from pommerman import agents
import DQN_agent
import gc

# Tensorflow version: 1.15.8
LOAD_NETWORK = False
FOLDER_PATH = "results_with_shaping"
NETWORK_PATH =  FOLDER_PATH + "/network.h5"
TARGET_NETWORK_PATH = FOLDER_PATH + "/target_network.h5"
REWARD_SUMS_PATH = FOLDER_PATH + "/reward_sums.npy"
STEP_SUMS_PATH = FOLDER_PATH + "/step_sums.npy"
EPISODE_RESULTS_PATH = FOLDER_PATH + "/episode_results.npy"
EPSILON_PATH = FOLDER_PATH + "/epsilon.txt"
REWARD_PLOT_PATH = FOLDER_PATH + "/reward_plot.png"
STEP_PLOT_PATH = FOLDER_PATH + "/step_plot.png"

# Setting up the agents
agent_id = 3
agent = DQN_agent.DQNAgent()
agent_list = [agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent(), agent]

# Setting up the environment
env = pommerman.make('PommeFFACompetition-v0', agent_list)
env.set_training_agent(agent_list[agent_id].agent_id)

# Creating the neural networks
n_actions = env.action_space.n


if (LOAD_NETWORK):
    network = tf.keras.models.load_model(NETWORK_PATH)
    target_network = tf.keras.models.load_model(TARGET_NETWORK_PATH)

    reward_sums = np.load(REWARD_SUMS_PATH).tolist()
    step_sums = np.load(STEP_SUMS_PATH).tolist()
    episode_results = np.load(EPISODE_RESULTS_PATH).tolist()

    total_steps_taken = sum(step_sums)

    f = open(EPSILON_PATH, "r")
    epsilon = float(f.read())
    f.close()
else:
    network = agent.make_network(n_actions, 6)
    target_network = agent.make_network(n_actions, 6)

    reward_sums = [] # Stores the rewards from each episode 
    step_sums = []   # Stores the number of steps taken each episode
    episode_results = []   # Stores the wins and losses each episode

    total_steps_taken = 0

    epsilon = 1


def replay(replay_memory, minibatch_size=32):
    minibatch = np.random.choice(replay_memory, minibatch_size, replace=True)

    state_l = np.array(list(map(lambda x: x['state'], minibatch)))
    action_l = np.array(list(map(lambda x: x['action'], minibatch)))
    reward_l = np.array(list(map(lambda x: x['reward'], minibatch)))
    new_state_l = np.array(list(map(lambda x: x['new_state'], minibatch)))
    done_l = np.array(list(map(lambda x: x['done'], minibatch)))

    qvals_new_state_l = target_network.predict(new_state_l)
    target_f = network.predict(state_l)

    # Q-update
    for i,(state, action, reward, qvals_new_state, done) in enumerate(zip(state_l, action_l, reward_l, qvals_new_state_l, done_l)): 
        if not done:  target = reward + gamma * np.max(qvals_new_state)
        else:         target = reward
        target_f[i][action] = target

    network.fit(state_l, target_f, epochs=1, verbose=0)

    return network


def shape_rewards(action, state, last_state, reward, dist2bombs_prev):
    if last_state is None:
        return reward, 0
        
    if reward == 1:
        return 1, 0
    elif reward == -1:
        return -1, 0
        
    # Punishes standing still
    if action == 0 or (state["position"] == last_state["position"] and action != 5):
        reward -= 0.005

    # Rewards moving to a different position
    if state["position"] != last_state["position"]:
        reward += 0.0005

    # Rewards placing a bomb near an enemy or wood
    if state['ammo'] < last_state['ammo']:
        surroundings = [(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0)]
        bomb_pos = state['position']

        enemies = 0
        wood = 0

        for p in surroundings:
            bomb_pos = state['position']
            cell_pos = (bomb_pos[0] + p[0], bomb_pos[1] + p[1])
            if cell_pos[0] > 10 or cell_pos[1] > 10: 
                  continue
            enemies += state['board'][cell_pos] in [e.value for e in state['enemies']]
            wood += state['board'][cell_pos] == 2
    
        reward += enemies * 0.1 + wood * 0.05 + 0.001

    pose_t = np.array(state['position']) 
    pose_tm1 = np.array(last_state['position'])
    moveDist = np.linalg.norm(pose_t-pose_tm1)

    bombs_pos = np.argwhere(state['bomb_life'] != 0)
    dist2bombs = 0

    # Rewards walking awy from a bomb
    for bp in bombs_pos:
        dist2bombs += np.linalg.norm(state['position']-bp)
    dist_delta = dist2bombs - dist2bombs_prev
    if (dist_delta > 0 and moveDist):
        reward += dist_delta * 0.05

    # Limits reward 
    if reward < -0.9:
        reward = -0.9
    elif reward > 0.9:
        reward = 0.9

    return reward, dist2bombs


def plot_and_save(fig_number, data_name, data_list, save_path):
    # Plot average reward per 100 episodes
    average = []
    episode_num = []
    sum = 0

    for i in range(1, len(data_list)):
        if(i % 100 == 0):
            average.append(sum/100)  
            episode_num.append(i)
            sum = 0
        else:
            sum+=data_list[i]

    np_avg = np.array(average)
    np_ep = np.array(episode_num)

    plt.figure(fig_number)
    plt.plot(np_ep, np_avg)
    plt.xlabel("Episode number")
    plt.ylabel("Average " + data_name)
    plt.savefig(save_path)


# Parameters used for training the neural network
n_episodes = 60_001
gamma = 0.99
min_epsilon = 0.1
max_epsilon = 1
LAMBDA = 0.00003
minibatch_size = 32
replay_memory = [] # Replay memory stores state, action, reward, new_state, done'
mem_max_size = 100_000
steps_for_target = 100 # Updates the target network after steps


for n in range(n_episodes): 
    state = env.reset()
    state_training_agent = agent.restructure_state(state[agent_id])

    done=False
    reward_sum = 0
    episode_result = 0
    steps = 0
    last_state = None
    dist2bombs_prev = 0

    while not done: 
        # Uncomment this to see the agent learning
        #env.render()

        # Feedforward pass for current state to get predicted q-values for all actions 
        qvals_s = network.predict(np.expand_dims(state_training_agent, axis=0))
        agent.set_qvals_s(qvals_s)

        # Action from training agent
        if(np.random.random() > epsilon): 
            training_agent_action = agent.act(state[agent_id], env.action_space)
        else:
            training_agent_action = env.action_space.sample()

        # Action from other agents
        actions = env.act(state)
        # Adding action from training agent to the other actions
        actions.append(training_agent_action)

        # Take step
        new_state, reward, done, info = env.step(actions)

        episode_result += reward[agent_id]
        
        # Reward shaping
        new_reward, dist2bombs = shape_rewards(training_agent_action, new_state[agent_id], last_state, reward[agent_id], dist2bombs_prev)
        reward[agent_id] = new_reward 
        last_state = new_state[agent_id]
        dist2bombs_prev = dist2bombs
        # Restructuring and reshaping the new training agent state
        new_state_training_agent = agent.restructure_state(new_state[agent_id])
        
        # Store reward
        reward_sum += reward[agent_id]

        # Add data to memory, respecting memory buffer limit 
        if len(replay_memory) > mem_max_size:
            replay_memory.pop(0)
        replay_memory.append({"state" : state_training_agent, "action" : training_agent_action, "reward" : reward[agent_id], "new_state" : new_state_training_agent, "done" : done})

        steps += 1

        # Updating weights for the target network every steps_for_target steps
        total_steps_taken += 1
        if total_steps_taken % steps_for_target == 0:
            target_network.set_weights(network.get_weights())

        # Update state
        state_training_agent = new_state_training_agent

        # Train the nnet that approximates q(s,a), using the replay memory
        network = replay(replay_memory, minibatch_size = minibatch_size)
        
        # Decaying epsilon 
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * exp(-LAMBDA * total_steps_taken )

    # Store episode reward sum
    reward_sums.append(reward_sum)

    # Store number of steps taken each episode
    step_sums.append(steps)

    # Store result of episode (1 for win, -1 for loss)
    episode_results.append(episode_result)

    if n % 1 == 0: print(n)

    if (n % 500 == 0):
        print("Save complete")
        tf.keras.models.save_model(network, NETWORK_PATH, overwrite=True)
        tf.keras.models.save_model(target_network, TARGET_NETWORK_PATH, overwrite=True)

        # Saves reward_sums to file
        np.save(REWARD_SUMS_PATH, reward_sums)

        # Saves step_sums to file
        np.save(STEP_SUMS_PATH, step_sums)

        # Saves episode_results to file
        np.save(EPISODE_RESULTS_PATH, episode_results)

        # Save plots
        plot_and_save(1, "reward", reward_sums, REWARD_PLOT_PATH)
        plot_and_save(2, "step", step_sums, STEP_PLOT_PATH)

        f = open(EPSILON_PATH, "w")
        f.write(str(epsilon))
        f.close()

print("Done with" + n_episodes - 1)