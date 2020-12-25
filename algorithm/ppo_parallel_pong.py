import torch
import numpy as np
import gym
# Parallel
from multiprocessing import Process, Pipe


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
RIGHT = 4
LEFT = 5
ENV = 'PongDeterministic-v4'
SEED = 0


def collect_trajectories(envs, policy, tmax=200, nrand=5):
    n = len(envs.ps)

    state_list = []
    reward_list = []
    prob_list = []
    action_list = []

    envs.reset()

    # Start parallel agents
    envs.step([1] * n)

    # Random steps at the beginning of each episode
    for _ in range(nrand):
        state1, reward1, _, _ = envs.step(np.random.choice([RIGHT, LEFT], n))
        state2, reward2, _, _ = envs.step([0] * n)

    for t in range(tmax):

        # prepare the input
        # preprocess_batch properly converts two frames into
        # shape (n, 2, 80, 80), the proper input for the policy
        # this is required when building CNN with pytorch
        batch_input = preprocess_batch([fr1, fr2])

        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        probs = policy(batch_input).squeeze().cpu().detach().numpy()

        action = np.where(np.random.rand(n) < probs, RIGHT, LEFT)
        probs = np.where(action == RIGHT, probs, 1.0 - probs)

        # advance the game (0=no action)
        # we take one action and skip game forward
        fr1, re1, is_done, _ = envs.step(action)
        fr2, re2, is_done, _ = envs.step([0] * n)

        reward = re1 + re2

        # store the result
        state_list.append(batch_input)
        reward_list.append(reward)
        prob_list.append(probs)
        action_list.append(action)

        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        if is_done.any():
            break


class ParallelEnv:
    def __init__(self, env_name, n, seed):
        # List of environment instances
        env_fns = [gym.make(env_name) for _ in range(n)]

        # Set seed
        for i, env in enumerate(env_fns):
            env.seed(i + seed)

        self.waiting = False  # ?
        self.closed = False  # ?

        # zip asterisk is unzip
        # remotes is parent_conn, and work_remotes is child_conn
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n)])

        # Set process
        self.ps = [
            Process(target=worker,
                    args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn)
            in zip(self.work_remotes, self.remotes, env_fns)
        ]

        # Initialize process
        for p in self.ps:
            p.deamon = True  # If the main process crashes, we should not cause things to hang?
            p.start()

        # work_remotes is child_conn
        for remote in self.work_remotes:
            remote.close()

        # ?
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()

    def step_async(self, actions):
        """
        env.step
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        # ?
        self.waiting = True

    def step_wait(self):
        """
        Returns concatenated outputs of env.step over the parallel agents.
        remotes is parent_conn
        """
        results = [remote.recv() for remote in self.remotes]
        # ?
        self.waiting = False
        states, rewards, dones, infos = zip(*results)
        return np.stack(states), np.stack(rewards), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        # sending reset can receive reset state
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


def worker(remote, parent_remote, env_fn_wrapper):
    """
    https://docs.python.org/3/library/multiprocessing.html
    """
    # Close the connection
    parent_remote.close()
    # Get an instance of gym environment
    env = env_fn_wrapper.x

    while True:

        # parent_remote sends a tuple of ('argument to env', action or None)
        cmd, action = remote.recv()

        if cmd == 'step':
            state, reward, done, info = env.step(action)
            if done:
                state = env.reset()
            # child send?
            remote.send((state, reward, done, info))

        elif cmd == 'reset':
            state = env.reset()
            remote.send(state)

        # reset_task?
        elif cmd == 'reset_task':
            state = env.reset_task()
            remote.send(state)

        elif cmd == 'close':
            remote.close()
            break

        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))

        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    """
    https://docs.python.org/3/library/pickle.html
    Use cloudpickle to serialize contents otherwise multiprocessing tries to use pickle.
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        # loads from a bytes string, and load from file-like object
        self.x = pickle.loads(ob)


def preprocess_batch(images, bkg_color = np.array([144, 72, 17])):
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 5:
        list_of_images = np.expand_dims(list_of_images, 1)
    # subtract bkg and crop
    list_of_images_prepro = np.mean(list_of_images[:,:,34:-16:2,::2]-bkg_color,
                                    axis=-1)/255.
    batch_input = np.swapaxes(list_of_images_prepro,0,1)
    return torch.from_numpy(batch_input).float().to(device)


def main():

    # Parallel environment
    envs = ParallelEnv(ENV, n=2, seed=SEED)


if __name__ == '__main__':
    main()
