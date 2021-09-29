# Copyright (c) 2018, Simon Brodeur
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  - Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#  - Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# Author: Simon Brodeur <simon.brodeur@usherbrooke.ca>
# Université de Sherbrooke, APP3 S8GIA, A2018

import math
import gym
import time
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

from skfuzzy import control as ctrl
from gym import spaces, logger
from gym.utils import seeding


# Adapted from: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num    Observation                 Min         Max
        0    Cart Position             -2.4            2.4
        1    Cart Velocity             -Inf            Inf
        2    Pole Angle                ~-180°         ~180°
        3    Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Box(1)
        Num    Action                     Min         Max
        0    Force applied to cart      -50           50

        Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        Cart position and velocity are null
        Pole angle is randomly sampled in [-5°, 5°]
        Pole velocity is randomly sampled in [-100°/sec, 100°/sec]
    Episode Termination:
        Pole Angle is more than ±90°
        Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 50.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 90 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            180.0,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(np.array([-self.force_mag]), np.array([self.force_mag]), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = action[0]
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x, x_dot, theta, theta_dot)
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        observation = np.array([x, x_dot, theta * 180 / np.pi, theta_dot * 180 / np.pi])
        assert self.observation_space.contains(observation), "%r (%s) invalid" % (observation, type(observation))
        return observation, reward, done, {}

    def reset(self):
        self.state = [0.0,
                      0.0,
                      self.np_random.uniform(low=-5, high=5) * np.pi / 180,
                      self.np_random.uniform(low=-100, high=100) * np.pi / 180]
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()


def createFuzzyController():

    # TODO: Create the fuzzy variables for inputs and outputs.
    # Defuzzification (defuzzify_method) methods for fuzzy variables:
    #    'centroid': Centroid of area
    #    'bisector': bisector of area
    #    'mom'     : mean of maximum
    #    'som'     : min of maximum
    #    'lom'     : max of maximum
    ant1 = ctrl.Antecedent(np.linspace(-1, 1, 1000), 'input1')
    ant2 = ctrl.Antecedent(np.linspace(-1, 1, 1000), 'input2')
    cons1 = ctrl.Consequent(np.linspace(-1, 1, 1000), 'output1', defuzzify_method='centroid')

    # Accumulation (accumulation_method) methods for fuzzy variables:
    #    np.fmax
    #    np.multiply
    cons1.accumulation_method = np.fmax

    # TODO: Create membership functions
    ant1['membership1'] = fuzz.trapmf(ant1.universe, [-1, -0.5, 0.5, 1])
    ant1['membership2'] = fuzz.trapmf(ant1.universe, [-0.75, -0.5, 0.5, 0.75])

    ant2['membership1'] = fuzz.trapmf(ant1.universe, [-1, -0.5, 0.5, 1])

    cons1['membership1'] = fuzz.trimf(cons1.universe, [-1, 0, 1])

    # TODO: Define the rules.
    rules = []
    rules.append(ctrl.Rule(antecedent=(ant1['membership1'] & ant2['membership1']), consequent=cons1['membership1']))

    # Conjunction (and_func) and disjunction (or_func) methods for rules:
    #     np.fmin
    #     np.fmax
    for rule in rules:
        rule.and_func = np.fmin
        rule.or_func = np.fmax

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    return sim


def main():

    # Create the environment and fuzzy controller
    env = CartPoleEnv()
    sim = createFuzzyController()

    # Display rules
    print('------------------------ RULES ------------------------')
    for rule in sim.ctrl.rules:
        print(rule)
    print('-------------------------------------------------------')

    # Display fuzzy variables
    for var in sim.ctrl.fuzzy_variables:
        var.view()
    plt.show()

    VERBOSE = False
    for episode in range(10):
        print('Episode no.%d' % (episode))
        env.reset()

        isSuccess = True
        action = np.array([0.0], dtype=np.float32)
        for _ in range(100):
            env.render()
            time.sleep(0.01)

            # Execute the action
            observation, _, done, _ = env.step(action)
            if done:
                # End the episode
                isSuccess = False
                break

            # Select the next action based on the observation
            cartPosition, cartVelocity, poleAngle, poleVelocityAtTip = observation

            # TODO: set the input to the fuzzy system
            sim.input['input1'] = 0
            sim.input['input2'] = 0

            sim.compute()
            if VERBOSE:
                sim.print_state()

            # TODO: get the output from the fuzzy system
            force = sim.output['output1']

            action = np.array([force], dtype=np.float32).flatten()

        if isSuccess:
            print('Controller succeeded!')
        else:
            print('Controller failed!')

    env.close()


if __name__ == "__main__":
    main()
