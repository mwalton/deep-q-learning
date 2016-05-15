'''
Classes based on work from Nervana Systems on a similar topic
https://github.com/NervanaSystems/simple_dqn/blob/master/src/environment.py
'''
import sys, os, logging, signal, sys
import cv2
log = logging.getLogger(__name__)

class Environment:
  def __init__(self):
    pass

  def numActions(self):
    # Returns number of actions
    raise NotImplementedError

  def reset(self):
    # Restarts environment
    raise NotImplementedError

  def act(self, action):
    # Performs action and returns reward
    raise NotImplementedError

  def getScreen(self):
    # Gets current game screen
    raise NotImplementedError

  def isTerminal(self):
    # Returns if game is done
    raise NotImplementedError

class Gym(Environment):
  '''Interface for OpenAI Gym'''
  def __init__(self, env_id, screen_width, screen_height):
    log.info("Using Gym Environment")
    import gym
    self.gym = gym.make(env_id)
    self.obs = None
    self.terminal = None
    # OpenCV expects width as first and height as second s
    self.dims = (screen_width, screen_height)

  def numActions(self):
    return self.gym.action_space.n

  def reset(self):
    self.gym.reset()
    self.obs = None
    self.terminal = None

  def act(self, action):
    self.obs, reward, self.terminal, _ = self.gym.step(action)
    return reward

  def getScreen(self):
    assert self.obs is not None
    return cv2.resize(cv2.cvtColor(self.obs, cv2.COLOR_RGB2GRAY), self.dims)

  def isTerminal(self):
    assert self.terminal is not None
    return self.terminal

