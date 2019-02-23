import tensorflow as tf
import numpy as np

# list our bandidts.
bandits = [0.3, 0.4, 0.9, 0.4]
num_bandits = len(bandits) 

def pullBandit(bandit):
    # get a random number
    rand = np.random.random()
    if bandit > rand:
        # retrn a positive reward
        return 1
    else:
        # return a negative reward
        return -1

tf.reset_default_graph()

# 2 lines est the feed-forward part of the network, does the actual choosing
weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights)

# 6 lines est training procedure, feed the reward and actio into the 
# next to compute the loss then update network
reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
responsible_weight = tf.slice(weights, action_holder, [1])
loss = -(tf.log(responsible_weight)*reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

# start to train the agent
total_episodes = 20 # set the episodes to train on
total_reward = np.zeros(num_bandits) # set scoreboard for bandits to 0
print(f'Total reward after 0 episodes: {total_reward}')

# execute the TF session
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
e = 0.1 # set the chance of taking a random action, 0 for plunger lift!
i = 0

while i < total_episodes:
    # choose either a random action of one from networkd
    if np.random.rand(1) > e:
        action = sess.run(chosen_action)
    else:
        action = np.random.randint(num_bandits)

    reward = pullBandit(bandits[action]) # get our reward form picking a bandit

    # update the network
    _,resp,ww = sess.run([update,responsible_weight,weights], feed_dict={reward_holder:[reward], action_holder:[action]})

    # update the running tally of scores
    total_reward[action] += reward

    i += 1
    if i % i == 0:
        print(f'Total reward after {i} episodes: {total_reward}')


print(f'The agent guessed that the best bandit is: {np.argmax(ww)+1}')
print(f'The known best bandit is: {np.argmax(bandits)+1}')
