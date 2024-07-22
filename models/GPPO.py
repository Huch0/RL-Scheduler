import os

from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from torch_geometric.data import Data, Batch

import time

import models.core as core
from utils.logger import EpochLogger
from scheduler_env.GraphJSSPEnv import GraphJSSPEnv


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, size, gamma=0.99, lam=0.95, device='cpu'):
        self.obs_buf = None
        self.can_buf = [[] for _ in range(size)]
        self.act_buf = th.zeros(size, dtype=th.int64).to(device)
        self.adv_buf = th.zeros(size, dtype=th.float32).to(device)
        self.rew_buf = th.zeros(size, dtype=th.float32).to(device)
        self.ret_buf = th.zeros(size, dtype=th.float32).to(device)
        self.val_buf = th.zeros(size, dtype=th.float32).to(device)
        self.logp_buf = th.zeros(size, dtype=th.float32).to(device)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs: Data, can, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        # Handle obs_buf as a Batch object
        if self.obs_buf is None:
            self.obs_buf = Batch.from_data_list([obs])
        else:
            self.obs_buf = Batch.from_data_list([*self.obs_buf.to_data_list(), obs])

        self.can_buf[self.ptr] = can
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        # Select the part of the buffers that belong to the current trajectory
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = th.cat((self.rew_buf[path_slice], th.tensor([last_val])))
        vals = th.cat((self.val_buf[path_slice], th.tensor([last_val])))

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, can=self.can_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return data


def train(
    env_fn=GraphJSSPEnv,
    instance_configs=dict(),
    actor_critic=core.GPPO,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=50,
    gamma=0.99,
    clip_ratio=0.2,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_pi_iters=80,
    train_v_iters=80,
    lam=0.97,
    max_ep_len=1000,
    target_kl=0.01,
    logger_kwargs=dict(),
    save_freq=10,
    val_freq=10
):
    """
    PPO-Clip with GNN feature extractor

    based on the implementation of Spinning Up
    (https://spinningup.openai.com/en/latest/algorithms/ppo.html)
    (https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py)

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())

    # Random seed
    seed = 0
    th.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    train_envs = [env_fn(config) for config in instance_configs['train']]
    val_envs = [env_fn(config) for config in instance_configs['val']]
    test_envs = [env_fn(config) for config in instance_configs['test']]

    # Create actor_critic model
    ac = actor_critic

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = steps_per_epoch
    buf = PPOBuffer(local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, can, act, adv, logp_old = data['obs'], data['can'], data['act'], data['adv'], data['logp']

        # Policy loss
        pis, logp = ac(obs, can, act)

        ratio = th.exp(logp - logp_old)
        clip_adv = th.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(th.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = th.mean(th.stack([pi.entropy() for pi in pis])).item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = th.as_tensor(clipped, dtype=th.float32).to(ac.device).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        v = ac.compute_v(obs)
        return ((v - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            # mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            # mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    start_time = time.time()
    best_model = None
    best_val_ret = -np.inf
    i, num_envs = 0, len(train_envs)
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        # Prepare for interaction with environment
        o = [env.reset()[0] for env in train_envs]
        ep_ret = [0 for _ in train_envs]
        ep_len = [0 for _ in train_envs]
        t = 0
        while t < local_steps_per_epoch:
            env = train_envs[i]
            done = False
            while not done:
                graph, can = o[i]['graph'].data, o[i]['candidate_op_indices']
                a, v, logp = ac.step(graph, can)

                next_o, r, tr, te, _ = env.step(a)
                ep_ret[i] += r
                ep_len[i] += 1
                # env.render() # For debugging

                # save and log
                a, v, logp = a.cpu(), v.cpu(), logp.cpu()
                buf.store(graph, can, a, r, v, logp)
                logger.store(VVals=v)

                # Update obs (critical!)
                o[i] = next_o

                done = tr or te
                timeout = ep_len[i] >= max_ep_len
                terminal = done or timeout
                epoch_ended = t == local_steps_per_epoch-1

                if terminal or epoch_ended:
                    if epoch_ended and not (terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len[i], flush=True)

                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        _, v, _ = ac.step(graph, can)
                    else:
                        v = 0
                    buf.finish_path(v)
                    if terminal:
                        # only save EpRet / EpLen if trajectory finished
                        logger.store(EpRet=ep_ret[i], EpLen=ep_len[i])
                    o[i], ep_ret[i], ep_len[i] = env.reset()[0], 0, 0

                t += 1
                if epoch_ended:
                    break

            i = (i + 1) % num_envs

        # Perform PPO update!
        update()

        # Validation
        if epoch % val_freq == 0:
            o = [env.reset()[0] for env in val_envs]
            ep_ret = [0 for _ in val_envs]
            ep_len = [0 for _ in val_envs]
            for i, env in enumerate(val_envs):
                while True:
                    a = ac.act(o[i]['graph'].data, o[i]['candidate_op_indices'])

                    next_o, r, tr, te, _ = env.step(a)
                    ep_ret[i] += r
                    ep_len[i] += 1

                    o[i] = next_o

                    done = tr or te
                    timeout = ep_len[i] >= max_ep_len
                    terminal = done or timeout

                    if terminal:
                        break
            mean_val_ret = np.mean(ep_ret)
            mean_val_len = np.mean(ep_len)

            # Log info about validation
            logger.store(
                ValMeanEpRet=mean_val_ret,
                ValMeanEpLen=mean_val_len
            )

            if mean_val_ret > best_val_ret:
                best_val_ret = mean_val_ret
                best_model = ac.state_dict()
                logger.log('New best model found!')

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        # validation
        logger.log_tabular('ValMeanEpRet', average_only=True)
        logger.log_tabular('ValMeanEpLen', average_only=True)
        logger.dump_tabular()

    # Return best model
    return best_model


if __name__ == "__main__":
    dir_path = os.path.join(os.path.dirname(__file__), '../instances')
    instance_configs = {
        'train': [],
        'val': [],
        'test': []
    }

    config = {
        'type': 'standard',
        'path': os.path.join(dir_path, f'standard/ta01'),
        'repeat': [1] * 15
    }
    instance_configs['train'].append(config)
    instance_configs['val'].append(config)
    instance_configs['test'].append(config)

    # for i in range(1, 11):
    #     config = {
    #         'type': 'standard',
    #         'path': os.path.join(dir_path, f'standard/ta{str(i).zfill(2)}'),
    #         'repeat': [1] * 15
    #     }
    #     if i < 7:
    #         instance_configs['train'].append(config)
    #     elif i < 9:
    #         instance_configs['val'].append(config)
    #     else:
    #         instance_configs['test'].append(config)

    device = 'cpu'
    # if th.cuda.is_available():
    #     device = 'cuda'
    # elif th.backends.mps.is_available():
    #     device = 'mps'
    print(f'Using {device} device')

    gppo = core.GPPO(device=device)
    best_model = train(actor_critic=gppo,
                       instance_configs=instance_configs,
                       steps_per_epoch=2250,
                       epochs=10)

    # Save best model
    th.save(best_model, 'best_model.pth')

    # Load best model
    # model = core.GPPO()
    # model.load_state_dict(th.load('best_model.pth'))
    # model.eval()

    # # Test
    # test_envs = [GraphJSSPEnv(config) for config in instance_configs['test']]
    # for env in test_envs:
    #     step = 0
    #     obs, _ = env.reset()
    #     done = False
    #     total_reward = 0

    #     while not done:
    #         step += 1

    #         action = model.act(obs['graph'].data, obs['candidate_op_indices'])
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         done = terminated or truncated
    #         total_reward += reward

    #         if done:
    #             print("Goal reached!")
    #             print(step, info, total_reward)
    #             obs['graph'].visualize_graph()
    #             env.render()
