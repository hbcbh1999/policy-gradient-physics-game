#!/usr/bin/env python
"""
Train a Pong AI using policy gradient-based reinforcement learning.

Based on Andrej Karpathy's "Deep Reinforcement Learning: Pong from Pixels"
http://karpathy.github.io/2016/05/31/rl/
and the associated code
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
"""

from __future__ import print_function
import argparse
import numpy as np
import gym
import matplotlib.pyplot as plt
from policy_network import Network
import sfml
import math
import random
import cPickle as pickle
from sfml import sf
import time

# game = 0 # pong
game = 1 # simulation


parser = argparse.ArgumentParser()
parser.add_argument('--hidden_layer_size', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--batch_size_episodes', type=int, default=5)
parser.add_argument('--checkpoint_every_n_episodes', type=int, default=100)
parser.add_argument('--load_checkpoint', action='store_true')
parser.add_argument('--discount_factor', type=int, default=0.99)
parser.add_argument('--render', type=int, default=True)
args = parser.parse_args()


window_size_x = 80
window_size_y = 80

# gravity = 9.81
gravity = 0.0
delta_t = 0.15

# Action values to send to gym environment to move paddle up/down
UP_ACTION = 2
DOWN_ACTION = 3
# Mapping from action values to outputs from the policy network
action_dict = {DOWN_ACTION: 0, UP_ACTION: 1}


class Simulation:

    def __init__(self):

        self.render = True
        self.nRounds = 0
        self.touched_this_round = False

        radius = 15.0
        self.ball = Ball(radius=radius,
                         pos_x=random.uniform(radius, window_size_x-radius),
                         pos_y=radius)
        self.ball.set_vel(random.uniform(-5.0, 5.0),10.0)
        controller_height = 1.0
        controller_width = 15.0
        self.controller = Controller(height=controller_height,
                                     width=controller_width,
                                     pos_x=window_size_x/2,
                                     pos_y=window_size_y-controller_height/2)

        if self.render:
            self.ball_draw = sfml.graphics.RectangleShape((radius, radius))
            self.ball_draw.origin = (radius/2, radius/2)
            self.ball_draw.fill_color = sfml.graphics.Color(255, 0, 0)
            self.controller_draw = sfml.graphics.RectangleShape((controller_width, controller_height))
            self.controller_draw.origin = (controller_width/2, controller_height/2)
            self.controller_draw.fill_color = sfml.graphics.Color(255, 0, 0)
            self.render_window = sf.RenderWindow(sf.VideoMode(window_size_x, window_size_y), "pySFML Window")


    def render_win(self):
        self.ball_draw.position = (self.ball.pos_x, self.ball.pos_y)
        self.controller_draw.position = (self.controller.pos_x, self.controller.pos_y)
    
        self.render_window.clear()
        self.render_window.draw(self.ball_draw)
        self.render_window.draw(self.controller_draw)
        self.render_window.display()
        time.sleep(0.01)

    def collision_detect(self, ball, controller):

        damp = 1.0
        touched = False

        disk_pos_x = ball.pos_x
        disk_pos_y = ball.pos_y
        disk_vel_x = ball.vel_x
        disk_vel_y = ball.vel_y

        controller_pos_center_x = controller.pos_x
        controller_pos_max_x = controller_pos_center_x + controller.width / 2.
        controller_pos_min_x = controller_pos_center_x - controller.width / 2.
        distance_y = window_size_y - disk_pos_y - ball.radius


        if disk_pos_x < controller_pos_max_x and \
            disk_pos_x > controller_pos_min_x and \
                abs(distance_y) < controller.height:
            touched = True

            ball.set_pos(disk_pos_x, disk_pos_y)
            ball.set_vel(disk_vel_x, -damp * disk_vel_y)

        return touched


    def step(self, action):

        # move controller according to action
        self.controller.move(action)

        # calculate if ball is out of bounds
        out_of_bound = self.ball.update_position()

        # calculate if ball was touched by controller
        touched = self.collision_detect(self.ball, self.controller)

        # Logic for reward calculation

        reward = 0
        done = False

        # if ball is out of bounds: negative reward and round is lost
        if out_of_bound == True:

            # if ball was touched this round
            if self.touched_this_round:
                reward = 1
            else:
                reward = -1

            # reset states
            self.touched_this_round = False
            self.nRounds += 1
            self.reset()

        # if controller touched ball: mark round as "won"
        elif touched ==  True:

            self.nRounds += 1
            self.touched_this_round = True

        # max number of rounds per episode reached
        if self.nRounds >= 20:

            # mark episode as done
            done = True

            # reset round counter
            self.nRounds = 0
            self.touched_this_round = False

        # convert reward to float
        reward = float(reward)

        obs = np.zeros((window_size_x, window_size_y), dtype=np.float32)
        pixel_pos_x = int(self.ball.pos_x)
        pixel_pos_y = int(self.ball.pos_y)
        r = int(self.ball.radius)
        obs[pixel_pos_x-r:pixel_pos_x+r, pixel_pos_y-r:pixel_pos_y+r] = 255.0

        observation = obs.ravel()

        return observation, reward, done

    def reset(self):

        controller_height = 2.0
        r = int(self.ball.radius)
        self.controller.set_pos(window_size_x/2, window_size_y-controller_height/2)
        self.ball.reset_pos(random.uniform(r, window_size_x-r), r)
        self.ball.set_vel(random.uniform(-5.0, 5.0), 10.0)

        obs = np.zeros((window_size_x, window_size_y), dtype=np.float32)
        pixel_pos_x = int(self.ball.pos_x)
        pixel_pos_y = int(self.ball.pos_y)
        obs[pixel_pos_x-r:pixel_pos_x+r, pixel_pos_y-r:pixel_pos_y+r] = 255.0
        observation = obs.ravel()

        return observation

class Controller:
    def __init__(self, height, width, pos_x, pos_y):
        self.height = height
        self.width = width
        self.pos_x = pos_x
        self.pos_y = pos_y

    def set_pos(self, x, y):
        self.pos_x = x
        self.pos_y = y

    def move(self, direction):
        if direction == 2:
            self.pos_x -= 2.0

        elif direction == 3:
            self.pos_x += 2.0

        # controller hits right boundary
        if self.pos_x + self.width / 2 > window_size_x:
            self.pos_x = window_size_x - self.width / 2

        # controller hits left boundary
        if self.pos_x - self.width / 2 < 0:
            self.pos_x = self.width / 2

class Ball:
    def __init__(self, radius, pos_x, pos_y):
        self.radius = radius
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.vel_x = 0.0
        self.vel_y = 0.0

    def reset_pos(self, x, y):
        self.pos_x = x
        self.pos_y = y
        self.vel_x = 0.0
        self.vel_y = 0.0

    def set_pos(self, x, y):
        self.pos_x = x
        self.pos_y = y

    def set_vel(self, vel_x, vel_y):
        self.vel_x = vel_x
        self.vel_y = vel_y

    def update_position(self):

        oob = False

        delta_velocity_y = gravity * delta_t
        self.vel_y += delta_velocity_y

        delta_velocity_x = 0.0
        self.vel_x += delta_velocity_x

        delta_pos_y = self.vel_y * delta_t
        delta_pos_x = self.vel_x * delta_t

        bounce = 1.0

        self.pos_y += delta_pos_y
        self.pos_x += delta_pos_x

        if self.pos_y + self.radius > window_size_y:  # disk is out of bounds
            oob = True

        if self.pos_y - self.radius < 0.0:
            self.pos_y = self.radius
            self.vel_y = -bounce*self.vel_y

        if self.pos_x + self.radius > window_size_x:
            self.pos_x = window_size_x - self.radius
            self.vel_x = -bounce*self.vel_x

        if self.pos_x - self.radius < 0.0:
            self.pos_x = self.radius
            self.vel_x = -bounce*self.vel_x

        return oob


# From Andrej's code
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()

def discount_rewards(rewards, discount_factor):
    discounted_rewards = np.zeros_like(rewards)
    for t in range(len(rewards)):
        discounted_reward_sum = 0
        discount = 1
        for k in range(t, len(rewards)):
            discounted_reward_sum += rewards[k] * discount
            discount *= discount_factor
            if rewards[k] != 0:
                # Don't count rewards from subsequent rounds
                break
        discounted_rewards[t] = discounted_reward_sum
    return discounted_rewards

if game == 0:
    env = gym.make('Pong-v0')
else:
    env = Simulation()


network = Network(
    args.hidden_layer_size, args.learning_rate, checkpoints_dir='checkpoints')
if args.load_checkpoint:
    network.load_checkpoint()

batch_state_action_reward_tuples = []
smoothed_reward = None
episode_n = 1

my_rewards = []
my_episodes = []

while True:
    print("Starting episode %d" % episode_n)

    episode_done = False
    episode_reward_sum = 0

    round_n = 1

    if game == 0:
        last_observation = env.reset()
        last_observation = prepro(last_observation)
    else:
        last_observation = env.reset()

    if game == 0:
        action = env.action_space.sample()  #random action
        observation, _, _, _ = env.step(action)
        observation = prepro(observation)
    else:
        if random.uniform(0.0,1.0) < 0.5:
            action = 2
        else:
            action = 3
        observation, _,_ = env.step(action)

    n_steps = 1

    while not episode_done:
        if args.render:
            if episode_n % 1 == 0:
                if game == 0:
                    env.render()
                else:
                    env.render_win()

        observation_delta = observation - last_observation
        last_observation = observation
        up_probability = network.forward_pass(observation_delta)[0]
        if np.random.uniform() < up_probability:
            action = UP_ACTION
        else:
            action = DOWN_ACTION

        if game == 0:
            observation, reward, episode_done, info = env.step(action)
            observation = prepro(observation)
        else:
            observation, reward, episode_done = env.step(action)

        episode_reward_sum += reward
        n_steps += 1

        tup = (observation_delta, action_dict[action], reward)
        batch_state_action_reward_tuples.append(tup)

        if reward == -1:
            print("Round %d: %d time steps; lost..." % (round_n, n_steps))
        elif reward == +1:
            print("Round %d: %d time steps; won!" % (round_n, n_steps))
        if reward != 0:
            round_n += 1
            n_steps = 0

    print("Episode %d finished after %d rounds" % (episode_n, round_n))

    # exponentially smoothed version of reward
    if smoothed_reward is None:
        smoothed_reward = episode_reward_sum
    else:
        smoothed_reward = smoothed_reward * 0.99 + episode_reward_sum * 0.01
    print("Reward total was %.3f; discounted moving average of reward is %.3f" \
        % (episode_reward_sum, smoothed_reward))

    if episode_n % args.batch_size_episodes == 0:
        states, actions, rewards = zip(*batch_state_action_reward_tuples)
        rewards = discount_rewards(rewards, args.discount_factor)
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards)
        batch_state_action_reward_tuples = list(zip(states, actions, rewards))
        network.train(batch_state_action_reward_tuples)
        batch_state_action_reward_tuples = []

    if episode_n % args.checkpoint_every_n_episodes == 0:
        network.save_checkpoint()

    episode_n += 1

    if episode_n % 5 == 0:
        my_rewards.append(smoothed_reward)
        my_episodes.append(episode_n)
        plt.plot(my_episodes, my_rewards, 'r-')
        plt.savefig('reward.png')
