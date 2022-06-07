"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
from time import sleep
from env import ClimberEnv
from rl import DDPG

MAX_EPISODES = 3000
MAX_EP_STEPS = 250
ON_TRAIN = True

# set env
env = ClimberEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []
def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            env.render()

            a = rl.choose_action(s, j, i, train=True)

            s_, r, done = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                with open('overnight.txt', 'a') as f:
                    f.write('Ep: %i | %s | ep_r: %.1f | step: %i \n' % (i, '---' if not done else 'done', ep_r, j))
                break
    rl.save()


def eval(name):
    rl.restore(name)
    env.render()
    env.viewer.set_vsync(True)
    while True:
        s = env.reset()
        for i in range(200):
            env.render()
            a = rl.choose_action(s, train=False)
            s, r, done = env.step(a)
            sleep(0.03)
            if done:
                print(i)
                break
        else:
            print(200)

ON_TRAIN = True
name = "./4/params_2022-06-04_0233"
if ON_TRAIN:
    train()
else:
    eval(name)

