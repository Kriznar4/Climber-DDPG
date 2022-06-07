# env.py
from time import sleep
import pyglet
import numpy as np
from scipy.spatial import distance

class ClimberEnv(object):
    viewer = None                           # first no viewer
    dt = 1                               # speed of rotation dt
    h = 180             #height of a climber
    d1 = h/6            #length from shoulder to elbow or from shoulder to shoulder
    d0 = d1*1.5         #length from elbow to the tip of the fingers or length from shoulder to heap or legngth from heap to knee (they are the same)
    d2 = d0 + 0.5*d1    #length from knee to the tip of the fingers
    veird_angle = np.arccos(d1/np.sqrt(d1**2 + d2**2))
    action_bound = [-1, 1]    # Rotation angle range   
    angle_bound = [
        [-1*np.pi, 1*np.pi],
        [-1*np.pi, 0*np.pi],
        [-1/2*np.pi, 1/4*np.pi],
        [-1/2*np.pi, 1/4*np.pi],
        [-1*np.pi, 0*np.pi],
        [veird_angle-1/2*np.pi, veird_angle+1/4*np.pi], 
        [veird_angle-1/2*np.pi, -veird_angle+1/2*np.pi],
        [0*np.pi, 1*np.pi],
        [0*np.pi, 3/4*np.pi], 
        [0*np.pi, 1*np.pi],
        [-1*np.pi, 0*np.pi],
    ]        
    goal = { 
        'll' : {'x': 100., 'y': 100., 'l': 40},
        'lh' : {'x': 90., 'y': 260.,  'l': 40},
        'rl' : {'x': 230., 'y': 100., 'l': 40},
        'rh' : {'x': 200., 'y': 240., 'l': 40},
        }  # blue goal (x,y) and lengths l
    state_dim = 17   + 9 +2                      # seven observations (x, y coordinates of fingers, if finger is at goal, if all the fingers are at goal) and 11 angles
    action_dim = 9                          # nine actions

    def __init__(self):
        self.time_on_goal = {'rl': 0, 'lh': 0, 'rh': 0}
        self.arm_info = np.zeros(
            11, dtype=[('l', np.float32), ('r', np.float32)])
        # generated matrix of (2,2)
        self.arm_info['l'] = [
            self.d2,
            self.d0,
            self.d1,
            self.d0,
            self.d2,
            np.sqrt(self.d1**2 + self.d2**2),
            self.d1,
            self.d0,
            self.d2,
            self.d1,
            self.d0,
            ]      
        self.arm_info['r'] = 0   # The end angles of the arms
        self.arm_info['r'][5] = self.arm_info['r'][2] + self.veird_angle
        self.arm_info['r'][8] = self.arm_info['r'][2] + (np.pi/2)

        self.points = np.zeros((11, 2, 2))
        self.compute_points()

    def step(self, action): 
        action = action[0]
        action = np.insert(action, 5, action[2])
        action = np.insert(action, 8, action[2])
        done = False
        r = 0.

        # Calculate the angle of rotation in unit time dt, limiting the angle to within 360 degrees
        action = np.clip(action, *self.action_bound)
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] = [np.clip(r, clip[0], clip[1]) for r, clip in zip(self.arm_info['r'], self.angle_bound)]  # bound
        # self.arm_info['r'] %= np.pi * 2
        # If the finger touches the blue goal, we decide to end the round (done)
        # So you need to calculate the coordinates of the fingers

        self.compute_points()

        in_goal = []
        # Get done and reward based on the coordinates of endpoint and goal
        for endpoint, goal, end_str in zip((self.rl_end, self.rh_end, self.lh_end), (self.goal['rl'], self.goal['rh'], self.goal['lh']), ('rl', 'rh', 'lh')):
            in_goal.append(0)
            if (goal['x'] - goal['l']/2 < endpoint[0] < goal['x'] + goal['l']/2) and (goal['y'] - goal['l']/2 < endpoint[1] < goal['y'] + goal['l']/2):
                in_goal.pop()
                in_goal.append(1)
                self.time_on_goal[end_str] += 1
                r += 10 + self.time_on_goal[end_str]
            else:
                #          r -= distance.euclidean([goal['x'], goal['y']], endpoint)/1000
                # r -= 50*self.time_on_goal[end_str]
                self.time_on_goal[end_str] = 0
        # print(sum(in_goal))
        if sum(in_goal) == 3 and all([self.time_on_goal[end_str] > 10 for end_str in ('rl', 'rh', 'lh')]):
            r += 100000
            done = True

        for angle, bound in zip(self.arm_info['r'], self.angle_bound):
            if angle in bound:
                pass
                #       r -= 1
        
        for a in action:
            if a in self.action_bound:
                pass
                #        r -= 1000
            

        #state
        s = np.concatenate([
            self.arm_info['r'], 
            [self.goal['ll']['x'], 
            self.goal['ll']['y'], 
            self.goal['rl']['x'], 
            self.goal['rl']['y'], 
            self.goal['rh']['x'], 
            self.goal['rh']['y'],
            self.goal['lh']['x'], 
            self.goal['lh']['y']],  
            self.rl_end, self.rh_end, self.lh_end, 
            in_goal
            ])
        # s = np.concatenate([
        #     [self.goal['ll']['x'], 
        #     self.goal['ll']['y'], 
        #     self.goal['rl']['x'], 
        #     self.goal['rl']['y'], 
        #     self.goal['rh']['x'], 
        #     self.goal['rh']['y'],
        #     self.goal['lh']['x'], 
        #     self.goal['lh']['y']],  
        #     self.rl_end, self.rh_end, self.lh_end, 
        #     in_goal
        #     ])
        return s, r, done

    def reset(self):
        self.arm_info['r'] = [np.random.uniform(low, high) for low, high in self.angle_bound]
        self.arm_info['r'][5] = self.arm_info['r'][2] + self.veird_angle
        self.arm_info['r'][8] = self.arm_info['r'][2] + (np.pi/2)

        self.compute_points()

        in_goal = []
        # Get done and reward based on the coordinates of endpoint and goal
        for endpoint, goal in zip((self.rl_end, self.rh_end, self.lh_end), (self.goal['rl'], self.goal['rh'], self.goal['lh'])):
            in_goal.append(0)
            if (goal['x'] - goal['l']/2 < endpoint[0] < goal['x'] + goal['l']/2) and (goal['y'] - goal['l']/2 < endpoint[1] < goal['y'] + goal['l']/2):
                in_goal.pop()
                in_goal.append(1)

        #state
        s = np.concatenate([
            self.arm_info['r'], 
            [self.goal['ll']['x'], 
            self.goal['ll']['y'], 
            self.goal['rl']['x'], 
            self.goal['rl']['y'], 
            self.goal['rh']['x'], 
            self.goal['rh']['y'],
            self.goal['lh']['x'], 
            self.goal['lh']['y']],  
            self.rl_end, self.rh_end, self.lh_end, 
            in_goal
            ])
        # s = np.concatenate([
        #     [self.goal['ll']['x'], 
        #     self.goal['ll']['y'], 
        #     self.goal['rl']['x'], 
        #     self.goal['rl']['y'], 
        #     self.goal['rh']['x'], 
        #     self.goal['rh']['y'],
        #     self.goal['lh']['x'], 
        #     self.goal['lh']['y']],  
        #     self.rl_end, self.rh_end, self.lh_end, 
        #     in_goal
        #     ])
        return s

    def render(self):
        if self.viewer is None: # if called render and no viewer, generate viewer
            self.viewer = Viewer(self.points, self.goal)
        self.viewer.render()    # use Viewers' render function

    def sample_action(self):
        action = np.random.rand(self.action_dim)*2*np.pi - np.pi    # two radians
        # action = [np.random.uniform(low, high) for i, (low, high) in zip(range(11), self.angle_bound) if i not in [5, 8]]
        return action

    def compute_points(self):
        points = self.points

        al = self.arm_info['l']  # radius, arm length
        ar = self.arm_info['r']  # radian, angle

        # for right leg endpoint
        points[0][0] = np.array([self.goal['ll']['x'], self.goal['ll']['y']])    # a1 start (x0, y0)
        points[0][1] = points[1][0] = np.array([np.cos(sum(ar[0:1])), np.sin(sum(ar[0:1]))]) * al[0] + points[0][0]
        points[1][1] = points[2][0] = points[5][0] = points[8][0] = np.array([np.cos(sum(ar[0:2])), np.sin(sum(ar[0:2]))]) * al[1] + points[1][0]
        points[2][1] = points[3][0] = np.array([np.cos(sum(ar[0:3])), np.sin(sum(ar[0:3]))]) * al[2] + points[2][0]
        points[3][1] = points[4][0] = np.array([np.cos(sum(ar[0:4])), np.sin(sum(ar[0:4]))]) * al[3] + points[3][0]
        self.rl_end = points[4][1] = np.array([np.cos(sum(ar[0:5])), np.sin(sum(ar[0:5]))]) * al[4] + points[4][0]

        
        # for right hand endpoint
        points[5][1] = points[6][0] = np.array([np.cos(sum(np.concatenate([ar[0:2], ar[5:6]]))), np.sin(sum(np.concatenate([ar[0:2], ar[5:6]])))]) * al[5] + points[5][0]
        points[6][1] = points[7][0] = np.array([np.cos(sum(np.concatenate([ar[0:2], ar[5:7]]))), np.sin(sum(np.concatenate([ar[0:2], ar[5:7]])))]) * al[6] + points[6][0]
        self.rh_end = points[7][1] = np.array([np.cos(sum(np.concatenate([ar[0:2], ar[5:8]]))), np.sin(sum(np.concatenate([ar[0:2], ar[5:8]])))]) * al[7] + points[7][0]

        # for left hand endpoint
        points[8][1] = points[9][0] = np.array([np.cos(sum(np.concatenate([ar[0:2], ar[8:9]]))), np.sin(sum(np.concatenate([ar[0:2], ar[8:9]])))]) * al[8] + points[8][0]
        points[9][1] = points[10][0] = np.array([np.cos(sum(np.concatenate([ar[0:2], ar[8:10]]))), np.sin(sum(np.concatenate([ar[0:2], ar[8:10]])))]) * al[9] + points[9][0]
        self.lh_end = points[10][1] = np.array([np.cos(sum(np.concatenate([ar[0:2], ar[8:]]))), np.sin(sum(np.concatenate([ar[0:2], ar[8:]])))]) * al[10] + points[10][0]

        self.points = points

class Viewer(pyglet.window.Window):
    bar_thc = 5     # the thickness of the limb

    def __init__(self, points, goal):
        # draw arms etc.

        # Create window inheritance
        # vsync in the case of True, Refresh by screen frequency, Otherwise, not at that frequency
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Climber', vsync=False)

        # window background color
        pyglet.gl.glClearColor(1, 1, 1, 1)

        # Add arm information
        self.points = points

        # Put the drawing information of the arm into this batch
        self.batch = pyglet.graphics.Batch()    # display whole batch at once

        # Add goal as blue dot
        goal_ll, goal_rl, goal_rh, goal_lh = goal['ll'], goal['rl'], goal['rh'], goal['lh']

        black = [(0,0,0)]
        green = [(51, 204, 51)]
        yellow = [(204, 153, 0)]
        red = [(204, 0, 0)]
        blue = [(0, 153, 255)]

        # Add window center point, the root of the arm
        self.center_coord = np.array([goal_ll['x'], goal_ll['y']])

        # The information of the blue goal includes his x, y coordinates, and the length of the goal l
        self.goal_ll = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal_ll['x'] - goal_ll['l'] / 2, goal_ll['y'] - goal_ll['l'] / 2,                # location
                     goal_ll['x'] - goal_ll['l'] / 2, goal_ll['y'] + goal_ll['l'] / 2,
                     goal_ll['x'] + goal_ll['l'] / 2, goal_ll['y'] + goal_ll['l'] / 2,
                     goal_ll['x'] + goal_ll['l'] / 2, goal_ll['y'] - goal_ll['l'] / 2]),
            ('c3B', green[0] * 4))    # color

        self.goal_rl = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal_rl['x'] - goal_rl['l'] / 2, goal_rl['y'] - goal_rl['l'] / 2,                # location
                     goal_rl['x'] - goal_rl['l'] / 2, goal_rl['y'] + goal_rl['l'] / 2,
                     goal_rl['x'] + goal_rl['l'] / 2, goal_rl['y'] + goal_rl['l'] / 2,
                     goal_rl['x'] + goal_rl['l'] / 2, goal_rl['y'] - goal_rl['l'] / 2]),
            ('c3B', yellow[0] * 4))    # color

        self.goal_rh = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal_rh['x'] - goal_rh['l'] / 2, goal_rh['y'] - goal_rh['l'] / 2,                # location
                     goal_rh['x'] - goal_rh['l'] / 2, goal_rh['y'] + goal_rh['l'] / 2,
                     goal_rh['x'] + goal_rh['l'] / 2, goal_rh['y'] + goal_rh['l'] / 2,
                     goal_rh['x'] + goal_rh['l'] / 2, goal_rh['y'] - goal_rh['l'] / 2]),
            ('c3B', red[0] * 4))    # color

        self.goal_lh = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal_lh['x'] - goal_lh['l'] / 2, goal_lh['y'] - goal_lh['l'] / 2,                # location
                     goal_lh['x'] - goal_lh['l'] / 2, goal_lh['y'] + goal_lh['l'] / 2,
                     goal_lh['x'] + goal_lh['l'] / 2, goal_lh['y'] + goal_lh['l'] / 2,
                     goal_lh['x'] + goal_lh['l'] / 2, goal_lh['y'] - goal_lh['l'] / 2]),
            ('c3B', blue[0] * 4))    # color

        # add an arm
        self.arms_batch = []
        
        colors = green*2+black+yellow*2+black+red*2+black+blue*2+black*2
        for i in range(13):
            self.arms_batch.append(
                self.batch.add(
                    4, pyglet.gl.GL_QUADS, None,
                    ('v2f', [10, 10, 10, 10, 20, 20, 20, 20]),
                    ('c3B', colors[i] * 4,)     # color
                )    
            )

    def render(self):
        # refresh and render on screen

        self._update_arm()  # Update arm content (No change for now)
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        # Refresh arm and other positions

        self.clear()        # clear screen
        self.batch.draw()   # draw what's inside batch

    def _update_arm(self): 
        arms = np.concatenate([self.points, [[self.points[2][1], self.points[5][1]], [self.points[8][1], self.points[5][1]]]])
 

        for arm, arm_batch in zip(arms, self.arms_batch):
            arm_batch.vertices = np.concatenate([arm[0], arm[0] + [3, 3], arm[1]+ [3, 3], arm[1]])



if __name__ == '__main__':
    env = ClimberEnv()
    while True:
        env.render()
        env.step(env.sample_action())
        print(env.sample_action())
        #sleep(0.01)

# if __name__ == '__main__':
#     env = ArmEnv()
#     while True:
#         env.render()