#! /usr/bin/env python

import ddpg_turtlebot_human_action
import gym
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, merge
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import random
from collections import deque
import os.path
import timeit
import csv
import math
import time
import sys
import rospy
import scipy.io as sio

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class ActorCritic:
	def __init__(self, env, sess):
		self.env  = env
		self.sess = sess

		self.learning_rate = 0.0001
		self.epsilon = .9
		self.epsilon_decay = .99995
		self.gamma = .90
		self.tau   = .01

		# ===================================================================== #
		#                               Actor Model                             #
		# Chain rule: find the gradient of chaging the actor network params in  #
		# getting closest to the final value network predictions, i.e. de/dA    #
		# Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
		# ===================================================================== #

		self.memory = deque(maxlen=40000)




	def remember(self, cur_state, action, reward, new_state, done):
		# self.memory.append([cur_state, action, reward, new_state, done])
		# print("memory size is %s", [cur_state, action, reward, new_state, done])
		# print("cur_state shape is %s", cur_state.shape)
		# #action = action.reshape(2)
		# print("action shape is %s", action.shape)
		# self.test_data = np.arange(3)
		# print("test_data shape is %s", self.test_data.shape)
		cur_state = cur_state.reshape(28)
		action = action.reshape(2)
		self.array_reward = np.array(reward)
		self.array_reward = self.array_reward.reshape(1)
		new_state = new_state.reshape(28)
		done = np.array(done)
		done = done.reshape(1)
		self.memory_pack = np.concatenate((cur_state, action))
		self.memory_pack = np.concatenate((self.memory_pack, self.array_reward))
		self.memory_pack = np.concatenate((self.memory_pack, new_state))
		self.memory_pack = np.concatenate((self.memory_pack, done))
		self.memory.append(self.memory_pack)

		print("self.memory length is %s", len(self.memory))


		if len(self.memory)%10==0:
			sio.savemat('human_data.mat',{'data':self.memory},True,'5', False, False,'row') 
		
		# if len(self.memory)>100:
		# 	print("Human data is enough")

		# print("memory pack is %s", self.memory_pack.shape)
		# self.f = open("data.txt", "a")
		# np.savetxt('data.txt', self.memory, delimiter=',')
		# self.f.close()


def main():
	
	sess = tf.Session()
	K.set_session(sess)

	########################################################
	game_state= ddpg_turtlebot_human_action.GameState()   # game_state has frame_step(action) function
	actor_critic = ActorCritic(game_state, sess)
	########################################################
	num_trials = 1000
	trial_len  = 500
	record_human = 1
	current_state = game_state.reset()

	if (record_human==1):
		for i in range(num_trials):
			print("trial:" + str(i))
		##############################################################################################
			total_reward = 0
			for j in range(trial_len):

				###########################################################################################
				current_state = game_state.read_state()
				current_state = current_state.reshape((1, game_state.observation_space.shape[0]))
				action = game_state.read_action()
				action = action.reshape((1, game_state.action_space.shape[0]))
				reward, new_state, crashed_value = game_state.read_game_step(0.1, action[0][1], action[0][0])

				#print("reward is %s", reward)
				#print("current_state is %s", current_state)
				#print("new_state is %s", new_state)
				actor_critic.remember(current_state, action, reward, new_state, crashed_value)

				time.sleep(2)
				if rospy.is_shutdown():
					print('shutdown')
					break
		
		##########################################################################################

if __name__ == "__main__":
	main()