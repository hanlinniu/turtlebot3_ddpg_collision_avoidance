# -*- coding: utf-8 -*-
"""Prioritized Replay buffer for algorithms.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1511.05952.pdf
         https://arxiv.org/pdf/1707.08817.pdf
"""

import random
from typing import Any, List, Tuple
import numpy as np
from collections import deque
import time

# from segment_tree import MinSegmentTree, SumSegmentTree

# def retrieve(a, upperbound):
#     b = np.zeros(a.shape)
#     for i in range(0,len(a)):
#         #b[i]=np.power((a[i]-upperbound), 2)
#         b[i] = abs(a[i]-upperbound)
#     # print("a is %s", a)
#     # print("min b is %s", np.min(b))
#     # print("upperbound is %s", upperbound)
#     min_positions = np.argmin(b) #[i for i, x in enumerate(b) if x == np.min(b)]
#     # if len(min_positions)>1:
#     #     one_min_positions = random.choice(min_positions)
#     # else: one_min_positions = min_positions[0]
#     # print("one_min_positions is %s", one_min_positions)
#     return min_positions


def stack_samples(samples):
    array = np.array(samples)
    #before_current_states = np.stack(array[:,0])
    current_states = np.stack(array[:,0]).reshape((array.shape[0],-1))
    actions = np.stack(array[:,1]).reshape((array.shape[0],-1))
    rewards = np.stack(array[:,2]).reshape((array.shape[0],-1))
    new_states = np.stack(array[:,3]).reshape((array.shape[0],-1))
    dones = np.stack(array[:,4]).reshape((array.shape[0],-1))


    # print("array.shape[0] is %s", array.shape[0])
    # print("before_current_states is %s", before_current_states)
    # print("before_current_states shape is %s", before_current_states.shape)
    # print("current_states is ",current_states) 
    # print("current_states shape is ",current_states.shape)   #(16, 28)

    
    # print("actions is ",actions)
    # print("actions shape is ",actions.shape)   #(256,1)
    # print("array shape is ", array.shape)
    # print("array shape [0] is ", array.shape[0])
    # print("array 1 is ", array[:,1])
    # print("array stack 1 is", np.stack(array[:,1]))

    return current_states, actions, rewards, new_states, dones

class PrioritizedReplayBuffer:
    """Create Prioritized Replay buffer.

    Refer to OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

    Attributes:
        alpha (float): alpha parameter for prioritized replay buffer
        epsilon_d (float): small positive constants to add to the priorities
        tree_idx (int): next index of tree
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        _max_priority (float): max priority
    """

    def __init__(
        self,
        buffer_size = 1000000,
        batch_size = 512,
        demo_size = 1000,
        gamma = 0.99,
        n_step = 1,
        alpha = 0.3,
        epsilon_d = 1.0,
        #demo= List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]],
    ):
        """Initialize.

        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            alpha (float): alpha parameter for prioritized replay buffer

        """
        assert alpha >= 0
        self.alpha = alpha
        self.epsilon_d = epsilon_d
        self.tree_idx = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.demo_size = demo_size

        # capacity must be positive and a power of 2.
        # tree_capacity = 1
        # while tree_capacity < self.buffer_size:
        #     tree_capacity *= 2

        # self.sum_tree remembers priority, its index will be used for sampling batch buffer
        self.sum_tree = np.zeros(self.buffer_size)

        # for i in range(self.batch_size):
        #     self.sum_tree[i] = random.uniform(0,1)
        # self.min_tree = np.zeros(self.buffer_size)
        self._max_priority = 1.0
        self.memory = deque(maxlen=self.buffer_size)




        # self.demo_size = 1000
        # if self.demo_size > 1:
        #     for i in range(self.demo_size):
        #         self.sum_tree[i] = self._max_priority ** self.alpha

        # just for testing
        # cur_state = np.zeros(5)
        # action = np.zeros(5)
        # reward = np.zeros(5)
        # new_state = np.zeros(5)
        # done = np.zeros(5)

        # for i in range(self.buffer_size):
        #     self.memory.append([cur_state, action, reward, new_state, done])

        # for init priority of demo
        # self.tree_idx = self.demo_size
        # for i in range(self.demo_size):
        #     self.sum_tree[i] = self._max_priority ** self.alpha
            # self.min_tree[i] = self._max_priority ** self.alpha
        

    def memory_data(self):
        return self.memory

    def add(self, cur_state, action, reward, new_state, done, indice, new_priorities):
        if indice < self.buffer_size:
            self.sum_tree[indice] = new_priorities
        else: 
            self.sum_tree[0:(self.buffer_size-1)] = self.sum_tree[1:self.buffer_size]
            self.sum_tree[self.buffer_size-1] = new_priorities

        # print("sum_tree is %s", self.sum_tree)
        print("priority memory length is %s", len(self.memory))
        self.memory.append([cur_state, action, reward, new_state, done])

    def _sample_proportional(self, batch_size):
        """Sample indices based on proportional."""
        # indices = []
        # sum_tree_batch = self.sum_tree[0:batch_size]
        # p_total = np.sum(sum_tree_batch) #self.sum_tree.sum(0, batch_size - 1)
        # segment = p_total / batch_size

        probility_total = np.sum(self.sum_tree)
        probility_tree = self.sum_tree/probility_total

        start_time = time.time()
        length_memory = len(self.memory)

        sampled_index  = np.random.choice(length_memory, batch_size, replace=False, p=probility_tree[0:length_memory])

        # a = np.amin(self.sum_tree[0:len(self.memory)])
        # b = np.amax(self.sum_tree[0:len(self.memory)])

        # for i in range(batch_size):
        #     upperbound = random.uniform(a, b)
        #     idx = retrieve(self.sum_tree[0:len(self.memory)], upperbound) 
        #     indices.append(idx)

        end_time = time.time()
        print("retrieve time is %s", (end_time - start_time))
        print("self.memory length is %s", len(self.memory))
        return sampled_index

    def sample(self, beta, batch_size):  # self.sum_tree and self.memory must match
        self.batch_size = batch_size

        """Sample a batch of experiences."""
        assert len(self.memory) >= self.batch_size
        assert beta > 0

        alpha = 0.3
        length_memory = len(self.memory)
        priority_tree = self.sum_tree[0:length_memory]

        for i in range(0,length_memory):
            priority_tree[i] = priority_tree[i] ** alpha

        probability_total = np.sum(priority_tree)
        probability_tree = priority_tree/probability_total
        indices  = np.random.choice(length_memory, batch_size, replace=False, p=probability_tree)


        max_weight = (np.amin(probability_tree) * batch_size) ** (-beta)
        # calculate weights
        weights_, eps_d = [], []
        for i in indices:
            eps_d.append(self.epsilon_d if i < self.demo_size else 0.0)
            weight = (probability_tree[i] * batch_size) ** (-beta)
            weights_.append(weight / max_weight)

        weights = np.array(weights_)
        eps_d = np.array(eps_d)
        # print("weights is %s", weights[0:20])


        # indices = self._sample_proportional(self.batch_size)
        # print("final indices is %s", indices)

        # # get max weight
        # length_memory = len(self.memory)
        # sum_sum_tree = np.sum(self.sum_tree)
        # p_min = np.amin(self.sum_tree) / sum_sum_tree
        # max_weight = (p_min * length_memory) ** (-beta)      # len(self) is for the buffer_size

        # # calculate weights
        # weights_, eps_d = [], []
        # for i in indices:
        #     eps_d.append(self.epsilon_d if i < self.demo_size else 0.0)
        #     # print("i is %s", i)
        #     p_sample = self.sum_tree[i] / sum_sum_tree  #self.sum_tree.sum()
        #     weight = (p_sample * length_memory) ** (-beta)
        #     weights_.append(weight / max_weight)

        # weights = np.array(weights_)
        # eps_d = np.array(eps_d)

        # sample from self.memory
        # print("indices is %s", indices)
        # print("indices length is %s", len(indices))
        # print("self.memory length is %s", len(self.memory))

        memory_sample = random.sample(self.memory, self.batch_size)
        for i in range(0, len(indices)):
            # print("indices index is %s", indices[i])
            memory_sample[i] = self.memory[indices[i]]

        states, actions, rewards, next_states, dones = stack_samples(memory_sample)
        # print("states is %s", states)
        # print("states shape is %s", states.shape)
        # print("sum_tree is %s", self.sum_tree)

        return states, actions, rewards, next_states, dones, weights, indices, eps_d

    def update_priorities(self, indices, priority):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priority)

        hyper_parameters_eps_d = 0.4

        if self.demo_size==0:
            for i in range(0,len(indices)):
                self.sum_tree[indices[i]] = priority[i] ** self.alpha

        if self.demo_size>0:
            for i in range(0,len(indices)):
                if indices[i]<=self.demo_size:
                    priority[i] = priority[i] + hyper_parameters_eps_d
                self.sum_tree[indices[i]] = priority[i] ** self.alpha

        # print("self.tree now is %s", self.sum_tree[0:1000])
        # if len(self.memory)>1010:
        #     print("self.tree after 1000 is %s", self.sum_tree[1010])
        



            
        
        # self._max_priority = max(self._max_priority, priority[i])


        # print("updated_sum_tree is %s", self.sum_tree)




if __name__ == '__main__':
    haha  = PrioritizedReplayBuffer()
    # indice = haha._sample_proportional(32)
    states, actions, rewards, next_states, dones, weights, indices, eps_d = haha.sample(0.4)
    print("sampled_states is %s", states)
    indice = [0, 1, 2, 3]
    priority = [0, 5, 5, 5]
    haha.update_priorities(indice, priority)