
from pyparsing import alphas
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import datetime

#####################  hyper parameters  ####################

LR_A = 0.0005    # learning rate for actor
LR_C = 0.0005    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 5
BATCH_SIZE = 32

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0
        self.start_steps = 20

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        std_dev = 0.8
        self.noise = OUActionNoise(mean=np.zeros(self.a_dim), std_deviation=float(std_dev) * np.ones(self.a_dim))

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s, step_num=0, epsiode_num=0, train=True):
        dt = 0.2
        # if self.start_steps != 0:
        #     h = 180             #height of a climber
        #     d1 = h/6            #length from shoulder to elbow or from shoulder to shoulder
        #     d0 = d1*1.5         #length from elbow to the tip of the fingers or length from shoulder to heap or legngth from heap to knee (they are the same)
        #     d2 = d0 + 0.5*d1    #length from knee to the tip of the fingers
        #     veird_angle = np.arccos(d1/np.sqrt(d1**2 + d2**2))
        #     angle_bound = [
        #         [-1*np.pi, 1*np.pi],
        #         [-1*np.pi, 0*np.pi],
        #         [-1/2*np.pi, 1/4*np.pi],
        #         [-1/2*np.pi, 1/4*np.pi],
        #         [-1*np.pi, 0*np.pi],
        #         [veird_angle-1/2*np.pi, veird_angle+1/4*np.pi], 
        #         [veird_angle-1/2*np.pi, -veird_angle+1/2*np.pi],
        #         [0*np.pi, 1*np.pi],
        #         [0*np.pi, 3/4*np.pi], 
        #         [0*np.pi, 1*np.pi],
        #         [-1*np.pi, 0*np.pi],
        #     ] 
        #     return []
        a = self.sess.run(self.a, {self.S: s[None, :]})[0]
        print(a)
        # if train:
        #     if step_num <= 20:
        #         return (np.random.rand(self.a_dim)*2*np.pi - np.pi)
        #     return (a + (np.random.rand(self.a_dim)*2*np.pi - np.pi)*(((1/((epsiode_num//500)+1))*0.7+(1/((step_num//30)+1))*0.3)))*dt
        # else: 
        #     return a
        if train:
            if step_num <= 20:
                return self.noise()
            return (a + self.noise()*(((1/((epsiode_num//500)+1))*0.7+(1/((step_num//30)+1))*0.3)))*dt
        else: 
            return a*dt

    def learn(self):
        # soft target replacement
        print("learning")
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:      # indicator for learning
            self.memory_full = True

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 100
            alp = 0.2
            net_l1 = tf.layers.dense(s, n_l1, name='l1', trainable=trainable)
            net_l1 = tf.nn.leaky_relu(net_l1, alpha=alp)
            net_l2 = tf.layers.dense(net_l1, n_l1, name='l2', trainable=trainable)
            net_l2 = tf.nn.leaky_relu(net_l2, alpha=alp)
            net_l3 = tf.layers.dense(net_l2, n_l1, name='l3', trainable=trainable)
            net_l3 = tf.nn.leaky_relu(net_l3, alpha=alp)
            net_l4 = tf.layers.dense(net_l3, n_l1, name='l4', trainable=trainable)
            net_l4 = tf.nn.leaky_relu(net_l4, alpha=alp)
            net_l5 = tf.layers.dense(net_l4, n_l1, name='l5', trainable=trainable)
            net_l5 = tf.nn.leaky_relu(net_l5, alpha=alp)
            a = tf.layers.dense(net_l5, self.a_dim, name='a', trainable=trainable)
            return tf.clip_by_value(a, -1, 1)
 
    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 100
            alp = 0.2
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_l1 = tf.nn.leaky_relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1, alpha=0.2)
            net_l2 = tf.layers.dense(net_l1, n_l1, trainable=trainable)
            net_l2 = tf.nn.leaky_relu(net_l2, alpha=alp)
            net_l3 = tf.layers.dense(net_l2, n_l1, trainable=trainable)
            net_l3 = tf.nn.leaky_relu(net_l3, alpha=alp)
            net_l4 = tf.layers.dense(net_l3, n_l1, trainable=trainable)
            net_l4 = tf.nn.leaky_relu(net_l4, alpha=alp)
            net_l5 = tf.layers.dense(net_l4, n_l1, trainable=trainable)
            net_l5 = tf.nn.leaky_relu(net_l5, alpha=alp)
            return tf.layers.dense(net_l5, 1, trainable=trainable)  # Q(s,a)

    def save(self):
        date, time = str(datetime.datetime.now()).split()
        h, m, _ = time.split(':')
        saver = tf.train.Saver()
        saver.save(self.sess, './params_'+date+'_'+h+m, write_meta_graph=False)

    def restore(self, name):
        saver = tf.train.Saver()
        saver.restore(self.sess, name)



# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import numpy as np

# #####################  hyper parameters  ####################

# LR_A = 0.001    # learning rate for actor
# LR_C = 0.001    # learning rate for critic
# GAMMA = 0.9    # reward discount
# TAU = 0.01      # soft replacement
# MEMORY_CAPACITY = 10000
# BATCH_SIZE = 32


# class DDPG(object):
#     def __init__(self, a_dim, s_dim, a_bound,):
#         self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
#         self.pointer = 0
#         self.memory_full = False
#         self.sess = tf.Session()
#         self.a_replace_counter, self.c_replace_counter = 0, 0

#         self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]
#         self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
#         self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
#         self.R = tf.placeholder(tf.float32, [None, 1], 'r')

#         with tf.variable_scope('Actor'):
#             self.a = self._build_a(self.S, scope='eval', trainable=True)
#             a_ = self._build_a(self.S_, scope='target', trainable=False)
#         with tf.variable_scope('Critic'):
#             # assign self.a = a in memory when calculating q for td_error,
#             # otherwise the self.a is from Actor when updating Actor
#             q = self._build_c(self.S, self.a, scope='eval', trainable=True)
#             q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

#         # networks parameters
#         self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
#         self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
#         self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
#         self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

#         # target net replacement
#         self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
#                              for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

#         q_target = self.R + GAMMA * q_
#         # in the feed_dic for the td_error, the self.a should change to actions in memory
#         td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
#         self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

#         a_loss = - tf.reduce_mean(q)    # maximize the q
#         self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

#         self.sess.run(tf.global_variables_initializer())

#     def soft_replace_fun(self):
#         # networks parameters
#         self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
#         self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
#         self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
#         self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

#         self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
#                              for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

#     def choose_action(self, s):
#         return self.sess.run(self.a, {self.S: s[None, :]})[0]

#     def learn(self):
#         # soft target replacement
#         self.sess.run(self.soft_replace)

#         indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
#         bt = self.memory[indices, :]
#         bs = bt[:, :self.s_dim]
#         ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
#         br = bt[:, -self.s_dim - 1: -self.s_dim]
#         bs_ = bt[:, -self.s_dim:]

#         self.sess.run(self.atrain, {self.S: bs})
#         self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

#     def store_transition(self, s, a, r, s_):
#         transition = np.hstack((s, a, [r], s_))
#         index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
#         self.memory[index, :] = transition
#         self.pointer += 1
#         if self.pointer > MEMORY_CAPACITY:      # indicator for learning
#             self.memory_full = True

#     def _build_a(self, s, scope, trainable):
#         with tf.variable_scope(scope):
#             net = tf.layers.dense(s, 100, activation=tf.nn.relu, name='l1', trainable=trainable)
#             a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
#             return tf.multiply(a, self.a_bound, name='scaled_a')

#     def _build_c(self, s, a, scope, trainable):
#         with tf.variable_scope(scope):
#             n_l1 = 100
#             w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
#             w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
#             b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
#             net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
#             return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

#     def save(self):
#         saver = tf.train.Saver()
#         saver.save(self.sess, './params', write_meta_graph=False)

#     def restore(self):
#         saver = tf.train.Saver()
#         saver.restore(self.sess, './params')
