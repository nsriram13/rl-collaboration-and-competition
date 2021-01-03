# Collaboration and competition
Multi-Agent RL with Unity ML-Agents Tennis environment.

## Project Details
In this project, we train two agents to control rackets to bounce a ball over a net.
If an agent hits the ball over the net, it receives a reward of +0.1. If an agent
lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.
Thus, the goal of each agent is to keep the ball in play. We use a Unity ML-Agents
based environment provided by Udacity for this exercise. More details about the
environment is provided below.

The observation space consists of 8 variables corresponding to the position and velocity
of the ball and racket. Each agent receives its own, local observation. Two continuous
actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an
average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both
agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a
  score for each agent. This yields 2 (potentially different) scores. We then take the maximum of
  these 2 scores. This yields a single score for each episode.
* The environment is considered solved, when the average (over 100 episodes) of those scores is
  at least +0.5.


## Getting Started

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__:
	```bash
	conda create --name drlnd python=3.6
	conda activate drlnd
	```
	- __Windows__:
	```bash
	conda create --name drlnd python=3.6
	conda activate drlnd
	```

2. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
    ```bash
    git clone https://github.com/nsriram13/rl-collaboration-and-competition.git
    cd rl-collaboration-and-competition/python
    pip install .
    ```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

4. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.

5. This repository uses pre-commit hooks for auto-formatting and linting.
    * Run `pre-commit install` to set up the git hook scripts - this installs flake8 formatting, black
    auto-formatting as pre-commit hooks.
    * Run `gitlint install-hook` to install the gitlint commit-msg hook
    * (optional) If you want to manually run all pre-commit hooks on a repository,
    run `pre-commit run --all-files`. To run individual hooks use `pre-commit run <hook_id>`.

6. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

7. Place the file in the root directory of this repo and unzip (or decompress) the file. The notebook/script to train the
agent will look for the environment at the project root.

## Instructions
Run `python train.py` to train the agent with the default set of hyper-parameters.
Alternatively, you can modify the hyper-parameters by invoking the scripts with
the optional CLI flags; for e.g. say you want to use a different learning rate for
training the actor network, you can do so by running `python train.py --lrate_actor=1e-5`.
The full list of flags is shown below. You can access this list anytime by
running `python train.py --help`.

```bash
       USAGE: train.py [flags]
flags:

train.py:
  --ac_net: Actor critic network configuration
    (default: '"[512, 256]","[\'relu\', \'relu\']"')
    (a comma separated list)
  --batch_size: Number of training cases over which each SGD update is computed
    (default: '128')
    (an integer)
  --buffer_size: Data for SGD update is sampled from this number of most recent experiences
    (default: '1000000')
    (an integer)
  --checkpoint: Save the model weights to this file
    (default: './checkpoints/checkpoint.pth')
  --gamma: Discount factor used for DDPG update
    (default: '0.99')
    (a number)
  --lrate_actor: Learning rate for the actor network
    (default: '0.0005')
    (a number)
  --lrate_critic: Learning rate for the critic network
    (default: '0.0005')
    (a number)
  --max_episodes: Maximum number of episodes
    (default: '5000')
    (an integer)
  --seed: Random number generator seed
    (default: '0')
    (an integer)
  --target_score: Score to achieve in order to solve the environment
    (default: '0.5')
    (a number)
  --tau: Update factor for polyak averaging of target network weights
    (default: '0.01')
    (a number)
  --update_freq: Number of env steps between updates
    (default: '1')
    (an integer)
  --update_passes: Number of passes to run when updating
    (default: '5')
    (an integer)
  --weight_decay: Weight decay
    (default: '0.0')
    (a number)
```
