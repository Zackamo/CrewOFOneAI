import redis
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow.keras import layers
import keras

r = redis.Redis(host='127.0.0.1', port=6379, db=0, decode_responses=True)
startSub = r.pubsub()
startSub.psubscribe("start")

#size of the input vector, what the network "sees"
num_actions = 2

upper_bound = 2 * np.pi
lower_bound = 0

size = 36
num_tracked = 5
extra_params = 13
num_states = (size*size * num_tracked) + extra_params


################### Helper Functions ###############

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, size, size, num_tracked))
        self.player_buffer = np.zeros((self.buffer_capacity, extra_params))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, size, size, num_tracked))
        self.next_player_buffer = np.zeros((self.buffer_capacity, extra_params))
        
# Takes (s,p,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.player_buffer[index] = obs_tuple[1]
        self.action_buffer[index] = obs_tuple[2]
        self.reward_buffer[index] = obs_tuple[3]
        self.next_state_buffer[index] = obs_tuple[4]
        self.next_player_buffer[index] = obs_tuple[5]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, player_batch, action_batch, reward_batch, next_state_batch, next_player_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor([next_state_batch, next_player_batch], training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, next_player_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, player_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model([state_batch, player_batch], training=True)
            critic_value = critic_model([state_batch, player_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        player_batch = tf.convert_to_tensor(self.player_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        next_player_batch = tf.convert_to_tensor(self.next_player_buffer[batch_indices])

        self.update(state_batch, player_batch, action_batch, reward_batch, next_state_batch, next_player_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    state = layers.Input(shape=(size, size, num_tracked))
    player = layers.Input(shape=(extra_params))
    conv = layers.Conv2D(5, 8, strides=4)(state)
    conv = layers.Flatten()(conv)
    concat = layers.Concatenate(axis=1)([conv, player])
    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(2, activation="sigmoid", kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model([state, player], outputs)
    return model



def get_critic():
    # State as input
    state_input = layers.Input(shape=(size, size, num_tracked))
    player_input = layers.Input(shape=(extra_params))
    conv = layers.Conv2D(5, 8, strides=4)(state_input)
    conv = layers.Flatten()(conv)
    concat = layers.Concatenate(axis=1)([conv, player_input])
    state_out = layers.Dense(16, activation="relu")(concat)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, player_input, action_input], outputs)

    return model
        
    
def policy(state, player):
    sampled_actions = tf.squeeze(actor_model([state, player]))
   
    #sig_std_dev = 0.2

    noise = np.random.normal(0, sig_std_dev, 1)
    # Adding noise to action
    sampled_actions = sampled_actions + noise
    sampled_actions = sampled_actions % upper_bound

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    return [np.squeeze(legal_action)]




def getState():
    shootMode = int(r.get("mode"))
    swapTime = float(r.get("swapTime"))
    xPos = float(r.get("xPos"))
    x2Pos = xPos**2
    yPos = float(r.get("yPos"))
    y2Pos = yPos**2
    shotTime = float(r.get("shotTime"))
    powerupDuration = float(r.get("powerupDuration"))
    
    activePowerup = [0,0,0,0,0]
    index = int(r.get("powerupActive"))
    if index != 0:
        activePowerup[index - 1] = 1                     
                            
    reward = float(r.get("reward"))
    done = int(r.get("dead"))
    
    
    gunners = parsePositions("gunners")
    chargers = parsePositions("chargers")
    rocks = parsePositions("rocks")
    holes = parsePositions("holes")
    powerups = parsePositions("powerups")
    
    
    player = [shootMode, swapTime, xPos, x2Pos, yPos, y2Pos, shotTime, powerupDuration] + activePowerup
    state = np.dstack((gunners, chargers, rocks, holes, powerups))
    
    return state, player, reward, done

def setAction(action):
    #print("Action:", action)
    r.set("angle", float(action))

            
def start(ep):
    #r.set("ready?","yes")
    print("Waiting for start")
    for update in startSub.listen():
        if(int(update['data']) == ep):
            print("starting")
            state, player, reward, done = getState()
            return state, player

def parsePositions(redisKey):
    try: 
        temp = r.lpop(redisKey)
    except:
        print(redisKey, "is nil!")
        
    result = np.zeros((size, size))
    while temp != None:
        temp = int(temp)
        result[temp // size,temp % size] = 1
        temp = r.lpop(redisKey)
    return result 

'''########### Hyper Parameters ##################'''
#std_dev = 0.2

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 2000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)

############# Training Loop ##################
# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

action_one_list = []
action_two_list = []

# Takes about 4 min to train
for ep in range(total_episodes):

    prev_state, prev_player = start(ep)
    episodic_reward = 0
    sig_std_dev = 2/(1 + np.exp((1/total_episodes)*(ep - total_episodes/2)))
    print(sig_std_dev)
    
    tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state, dtype=tf.float32), 0)
    tf_prev_player = tf.expand_dims(tf.convert_to_tensor(prev_player, dtype=tf.float32), 0)

    
    action = policy(tf_prev_state, tf_prev_player)
    
    if prev_player[0] == 0:
        action_one_list.append(action[0][0])
        setAction(action[0][0])
    else:
        action_one_list.append(action[0][1])
        setAction(action[0][1])
        
    tf_action = tf.expand_dims(tf.convert_to_tensor(action, dtype=tf.float32), 0)
    
    #recorded_prev_state = tf.concat([tf.reshape(tf_prev_state, [-1]),tf.reshape(tf_prev_player,[-1])],0)
    
    #for update in updateSub.listen():
    while True:
        # Recieve state and reward from environment.
        state, player, totReward, done = getState()
        reward = totReward - episodic_reward
        
        tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        tf_player = tf.expand_dims(tf.convert_to_tensor(player, dtype=tf.float32), 0)
        
        #recorded_state = tf.concat([tf.reshape(tf_state, [-1]),tf.reshape(tf_player,[-1])],0)

        buffer.record((tf_prev_state, tf_prev_player, tf_action, reward, tf_state, tf_player))
        episodic_reward = totReward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        # End this episode when `done` is True
        if done:
            break

        tf_prev_state = tf_state
        tf_prev_player = tf_player
        #recorded_prev_state = recorded_state
        

        action = policy(tf_prev_state, tf_prev_player)
        
        if player[0] == 0:
            action_one_list.append(action[0][0])
            setAction(action[0][0])
        else:
            action_one_list.append(action[0][1])
            setAction(action[0][1])
            
        tf_action = tf.expand_dims(tf.convert_to_tensor(action, dtype=tf.float32), 0)

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
#plt.hist(action_one_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()

# Save the weights
actor_model.save_weights("actor.h5")
critic_model.save_weights("critic.h5")

target_actor.save_weights("target_actor.h5")
target_critic.save_weights("target_critic.h5")



