import random, logging
from tqdm import tqdm
import numpy as np
log = logging.getLogger(__name__)


class Agent:
    def __init__(self, environment, replay_memory, q_net=None,
                 display_screen=False, init_epsilon=1, end_epsilon=.1,
                 epsilon_decay_steps=1000000, train_frequency=4,
                 train_repeat=1,history_length=4,test_epsilon=0.05):
        self.env = environment
        self.mem = replay_memory
        self.net = q_net
        self.display_screen = display_screen
        self.init_epsilon = init_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.train_frequency = train_frequency
        self.train_repeat = train_repeat
        self.history_length = history_length

        self.test_epsilon = test_epsilon
        self.num_actions = self.env.numActions()

    def _restartRandom(self):
        self.env.reset()
        # TODO: perform random number of dummy actions (may not be a good idea)

    def _epsilon(self, t):
        # calculate decaying exploration rate
        if t < self.epsilon_decay_steps:
            return self.init_epsilon - t * (self.init_epsilon - self.end_epsilon) / self.epsilon_decay_steps
        else:
            return self.end_epsilon

    def step(self, epsilon):
        if random.random() < epsilon:
            action = random.randrange(self.num_actions)
        else:
            # select the action w/ highest estimated Q
            state = self.mem.getCurrentState()

            '''
            getCurrentState() returns minibatch where first item
            is the current state
            '''
            qvalues = self.net.predict(state)
            assert len(qvalues[0]) == self.num_actions
            # choose highest Q-value of first state
            action = np.argmax(qvalues[0])

        # render the display

        if self.display_screen:
            self.env.gym.render()

        # perform the action obtaining some reward
        reward = self.env.act(action)
        screen = self.env.getScreen()
        terminal = self.env.isTerminal()

        # add transition to buffer (otherwise we wouldn't have current state)
        self.mem.add(action, reward, screen, terminal)

        # restart if epsisode has ended
        if terminal:
            log.debug("Episode Terminated, restarting")
            self._restartRandom()

        # TODO: callback to record stats

        return terminal

    def play_random(self, num_steps):
        for i in tqdm(xrange(num_steps)):
            # let pr of random action epsilon = 1
            self.step(1.)

    def train(self, num_steps):
        # assert there is enough random steps in replay mem to support history
        assert self.mem.count >= self.history_length, "Not enough history in replay memory, increase random steps."
        # play number of steps
        for t in tqdm(xrange(num_steps)):
            self.step(t)
            if t % self.train_frequency == 0:
                # train for train_repeat times
                for i in xrange(self.train_repeat):
                    # sample minibatch
                    minibatch = self.mem.getMinibatch()
                    # train the network
                    self.net.train(minibatch)

    def test(self, num_steps):
        # just make sure there is history_length screens to form a state
        self._restartRandom()
        # play given number of steps
        for t in tqdm(xrange(num_steps)):
            # perform game step
            self.step(self.test_epsilon)

    def play(self, num_games):
        self._restartRandom()
        for i in xrange(num_games):
            # play until the episode terminates
            while not self.step(self.test_epsilon):
                pass

    def __del__(self):
        if self.display_screen:
            self.env.gym.render(close=True)
        pass
