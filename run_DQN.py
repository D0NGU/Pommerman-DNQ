import pommerman
from pommerman import agents
import DQN_agent
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)


    agent = DQN_agent.DQNAgent()
    network = load_model("results_with_shaping - brukt til poster/target_network.h5")
    agent_id = 3

    # Create a set of agents (exactly four)
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agent,
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    action_list = np.zeros(6)
    action_names = ["Stop", "Up", "Left",  "Down", "Right", "Bomb"]
    bar_colors = ["blue", "green", "red", "cyan", "purple", "orange"]
    def save_action(action):
        action_list[action] += 1

    wins = 0
    losses = 0
    # Run the episodes just like OpenAI Gym
    for i_episode in range(1000):
        state = env.reset()
        state_training_agent = agent.restructure_state(state[agent_id])
        done = False
        while not done:
            env.render()
            qvals_s = network.predict(np.expand_dims(state_training_agent, axis=0))
            agent.set_qvals_s(qvals_s)
            
            actions = env.act(state)

            save_action(actions[agent_id])
            
            state, reward, done, info = env.step(actions)
            
            if (reward[agent_id] == -1):
                losses += 1
                done = True
            elif (reward[agent_id] == 1):
                wins += 1

            state_training_agent = agent.restructure_state(state[agent_id])
            

        print('Episode {} finished'.format(i_episode))


    fig, ax = plt.subplots()
    ax.bar(action_names, action_list, color=bar_colors)
    ax.set_ylabel("Action Count")
    ax.set_title("Actions taken in 50 episodes with reward shaping")
    plt.show()
    env.close()

    print("Wins: " + str(wins))
    print("Losses: " + str(losses))


if __name__ == '__main__':
    main()
