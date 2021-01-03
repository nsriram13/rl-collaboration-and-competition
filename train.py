# -*- coding: utf-8 -*-
import os
import pickle
from collections import deque

import numpy as np
import torch
from absl import app, flags, logging
from unityagents import UnityEnvironment

from deeprl_cc.agent import DDPGAgent
from deeprl_cc.replay_buffer import ReplayBuffer

logging.set_verbosity(logging.DEBUG)

# Hyper-parameters
FLAGS = flags.FLAGS

flags.DEFINE_float("gamma", 0.99, "Discount factor used for DDPG update")
flags.DEFINE_float(
    "tau", 0.01, "Update factor for polyak averaging of target network weights"
)

flags.DEFINE_float("lrate_actor", 5e-4, "Learning rate for the actor network")
flags.DEFINE_float("lrate_critic", 5e-4, "Learning rate for the critic network")
flags.DEFINE_float("weight_decay", 0.0, "Weight decay")

flags.DEFINE_integer(
    "batch_size", 128, "Number of training cases over which each SGD update is computed"
)
flags.DEFINE_integer("update_freq", 1, "Number of env steps between updates")
flags.DEFINE_integer("update_passes", 5, "Number of passes to run when updating")
flags.DEFINE_list(
    "ac_net", [[512, 256], ["relu", "relu"]], "Actor critic network configuration"
)

# Buffer
flags.DEFINE_integer(
    "buffer_size",
    int(1e6),
    "Data for SGD update is sampled from this number of most recent experiences",
)

# Solution constraints
flags.DEFINE_float(
    "target_score", 0.5, "Score to achieve in order to solve the environment"
)
flags.DEFINE_integer("max_episodes", 5000, "Maximum number of episodes")

# Miscellaneous flags
flags.DEFINE_integer("seed", 0, "Random number generator seed")
flags.DEFINE_string(
    "checkpoint", "./checkpoints/checkpoint.pth", "Save the model weights to this file"
)


def main(_):

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    logging.get_absl_handler().use_absl_log_file()

    # Log all the hyperparameters
    logging.info(f"AC Network HParam: Hidden sizes set to {FLAGS.ac_net[0]}")
    logging.info(f"AC Network HParam: Activations set to {FLAGS.ac_net[1]}")
    logging.info(f"Actor Network HParam: Learning rate set to {FLAGS.lrate_actor}")
    logging.info(f"Critic Network HParam: Learning rate set to {FLAGS.lrate_critic}")
    logging.info(f"Optimizer weight decay set to {FLAGS.weight_decay}")
    logging.info(f"Target network weights soft update factor set to {FLAGS.tau}")

    logging.info(f"Replay buffer size set to {FLAGS.buffer_size}")
    logging.info(f"DDPG Learning HParam: Batch size set to {FLAGS.batch_size}")
    logging.info(f"DDPG Learning HParam: Discounting factor set to {FLAGS.gamma}")

    logging.info(
        f"DDPG Learning HParam: Number of env steps between updates set to "
        f"{FLAGS.update_freq}"
    )
    logging.info(
        f"DDPG Learning HParam: Number of passes to run when updating set to "
        f"{FLAGS.update_passes}"
    )

    env = UnityEnvironment(file_name="./Tennis_Linux_NoVis/Tennis.x86_64")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    logging.info(f'Number of agents: {num_agents}')

    # size of each action
    action_size = brain.vector_action_space_size
    logging.info(f'Size of each action: {action_size}')

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    logging.info(
        f'There are {states.shape[0]} agents. '
        f'Each observes a state with length: {state_size}'
    )
    logging.info(f'The state for the first agent looks like: {states[0]}')

    # Setup some variable for keeping track of the performance
    windowed_score = deque(maxlen=100)
    scores = []

    # see if GPU is available for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agents = [
        DDPGAgent(
            state_size,
            action_size,
            num_agents,
            sizes=FLAGS.ac_net[0],
            activations=FLAGS.ac_net[1],
            seed=FLAGS.seed,
            gamma=FLAGS.gamma,
            lrate_actor=FLAGS.lrate_actor,
            lrate_critic=FLAGS.lrate_critic,
            tau=FLAGS.tau,
            weight_decay=FLAGS.weight_decay,
            batch_size=FLAGS.batch_size,
            update_freq=FLAGS.update_freq,
            update_passes=FLAGS.update_passes,
            device=device,
        )
        for _ in range(num_agents)
    ]

    replay_buffer = ReplayBuffer(
        FLAGS.buffer_size, FLAGS.batch_size, num_agents, FLAGS.seed, device
    )

    # Begin training
    logging.info(f"Beginning training")
    logging.info(
        f"Goal: Achieve a score of {FLAGS.target_score:.2f} "
        f"in under {FLAGS.max_episodes} episodes."
    )

    for i_episode in range(1, FLAGS.max_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(num_agents)

        for agent in agents:
            agent.reset()

        for t in range(1000):
            # get actions for each agent
            actions = np.zeros([num_agents, action_size])
            for index, agent in enumerate(agents):
                actions[index, :] = agent.act(states[index], add_noise=True)

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            replay_buffer.store(states, actions, rewards, next_states, dones)
            for agent in agents:
                agent.step(replay_buffer, t)

            states = next_states
            score += rewards

            if any(dones):
                break

        score_max = np.max(score)
        scores.append(score_max)
        windowed_score.append(score_max)
        average_score = np.mean(windowed_score)

        logging.debug(
            f'Episode {i_episode}'
            f' | Score this episode: {score_max:.3f}'  # noqa: E501
            f' | Average Score: {np.mean(windowed_score):.3f}'  # noqa: E501
        )

        if i_episode % 100 == 0:
            logging.info(
                f'Episode {i_episode}'
                f' | Average over last 100 episodes: {average_score:.3f}'  # noqa: E501
            )

        if average_score >= FLAGS.target_score:

            torch.save(
                {
                    **{
                        f'player{i+1}_actor': agent.actor_local.state_dict()
                        for i, agent in enumerate(agents)
                    },
                    **{
                        f'player{i+1}_critic': agent.critic_local.state_dict()
                        for i, agent in enumerate(agents)
                    },
                },
                FLAGS.checkpoint,
            )

            logging.info(
                f"Environment solved in {i_episode-100} episodes! "
                f"Score: {average_score:.3f}."
            )
            break

        if i_episode >= FLAGS.max_episodes:
            print(
                f"Episode {i_episode} exceeded {FLAGS.max_episodes}. "  # noqa: E501
                f"Failed to solve environment!"
            )
            break

    # pickle and save the agent performance scores
    with open('./checkpoints/scores.pkl', 'wb') as f:
        pickle.dump(scores, f)


if __name__ == "__main__":
    app.run(main)
