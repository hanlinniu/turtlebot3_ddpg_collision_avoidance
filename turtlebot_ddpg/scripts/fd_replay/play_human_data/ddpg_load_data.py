import numpy as np
import scipy.io as sio
# state_num = 25 #685                 # when you change this value, remember to change the reset default function as well
# action_num = 2
# observation_space = np.empty(state_num)
# action_space = np.empty(action_num)

# return_action = np.zeros(action_num)

# print("return_action shape is %s ", return_action.shape)
# return_action = return_action.reshape((1, action_space.shape[0]))
# print("return_action shape is %s ", return_action.shape)

# print("return_action[0][0] is %s ", return_action[0][0])
# print("return_action[0][1] is %s ", return_action[0][1])


mat_contents = sio.loadmat('human_data.mat')
# print("human_data is %s", mat_contents['data'])
a = mat_contents['data']


print("a length is %s", len(a))


for i in range(60):
	# print("a %s length %s", i, len(a[i]))
	# print("a %s current state is %s", i, len(a[i][0:28]))
	# print("a %s current state is %s", i, a[i][0:28])
	print("a %s action is %s", i, a[i][28], a[i][29])
	# print("a %s reward is %s", i, a[i][30])
	# print("a %s new state is %s", i, a[i][31:59])
	# print("a %s done is %s", i, a[i][59])
	# print("a %s total is %s", i, a[i])

# i = 10
# print("a %s current state is %s", i, len(a[i][0:24]))
# print("a %s current state is %s", i, a[i][0:2])
# print("a %s current state is %s", i, a[i][2])
# print("a %s action is %s", i, a[i][25], a[i][26])
# print("a %s reward is %s", i, a[i][27])
# print("a %s new state is %s", i, a[i][27:52])
# print("a %s done is %s", i, a[i][53])