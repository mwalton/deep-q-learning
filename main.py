import argparse
import random
from deep_qlean.environment import Gym
from deep_qlean.agent import Agent
from deep_qlean.replay_memory import ReplayMemory
from deep_qlean.q_networks import ConvNet, ConvAutoEncoder

import logging, signal, sys
logging.basicConfig(format='%(asctime)s %(message)s')


def close_gui(signal, frame):
    '''Set a callback for attempting to close display
    on SIGINT. Arcade Learning Environments / OpenAI Gym windows do
    not close automatically after program terminates or keyboard interrupt
    '''
    if env and args.display_screen:
        env.gym.render(close=True)
    sys.exit(0)

signal.signal(signal.SIGINT, close_gui)


def str2bool(v):
    '''map common argument semantics to bool'''
    return v.lower() in ("yes", "true", "t", "1")


def network_factory(type, **kwargs):
    if type == 'conv_net':
        return ConvNet(kwargs)
    elif type == 'conv_autoencoder':
        return ConvAutoEncoder(kwargs)
    else:
        raise TypeError, "Not a valid Q-Network Topology"

# Parse input arguments
parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Environment')
envarg.add_argument("env_id", help="OpenAI Gym environment id, eg Breakout-v0")
envarg.add_argument("--screen_width", type=int, default=84, help="Screen width after resize.")
envarg.add_argument("--screen_height", type=int, default=84, help="Screen height after resize.")
envarg.add_argument("--display_screen", type=str2bool, default=False, help="Display game screen during training and testing.")

memarg = parser.add_argument_group('Replay memory')
memarg.add_argument("--replay_size", type=int, default=1000000, help="Maximum size of replay memory.")
memarg.add_argument("--history_length", type=int, default=4, help="How many screen frames form a state.")
memarg.add_argument("--min_reward", type=float, default=-1, help="Minimum reward.")
memarg.add_argument("--max_reward", type=float, default=1, help="Maximum reward.")

netarg = parser.add_argument_group('Deep Q-learning network')
netarg.add_argument("--qnet_type", choices=['conv_net','conv_autoencoder'], default='conv_net', help="Q-Network topology")
netarg.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate.")
netarg.add_argument("--gamma", type=float, default=0.99, help="Discount rate for future rewards.")
netarg.add_argument("--batch_size", type=int, default=32, help="Batch size for neural network.")
netarg.add_argument('--optimizer', choices=['rmsprop', 'adam', 'adadelta'], default='rmsprop', help='Network optimization algorithm.')
netarg.add_argument("--decay_rate", type=float, default=0.95, help="Decay rate for RMSProp and Adadelta algorithms.")
netarg.add_argument("--clip_error", type=float, default=1, help="Clip error term in update between this number and its negative.")
netarg.add_argument("--target_steps", type=int, default=10000, help="Copy main network to target network after this many steps.")

agentarg = parser.add_argument_group('Agent')
agentarg.add_argument("--init_epsilon", type=float, default=1, help="Exploration rate at the beginning of decay.")
agentarg.add_argument("--end_epsilon", type=float, default=0.1, help="Exploration rate at the end of decay.")
agentarg.add_argument("--epsilon_decay_steps", type=float, default=1000000, help="How many steps to decay the exploration rate.")
agentarg.add_argument("--test_epsilon", type=float, default=0.05, help="Exploration rate used during testing.")
agentarg.add_argument("--train_frequency", type=int, default=4, help="Perform training after this many game steps.")
agentarg.add_argument("--train_repeat", type=int, default=1, help="Number of times to sample minibatch during training.")

mainarg = parser.add_argument_group('Main loop')
mainarg.add_argument("--random_steps", type=int, default=50000, help="Populate replay memory with random steps before starting learning.")
mainarg.add_argument("--train_steps", type=int, default=250000, help="How many training steps per epoch.")
mainarg.add_argument("--test_steps", type=int, default=125000, help="How many testing steps after each epoch.")
mainarg.add_argument("--epochs", type=int, default=200, help="How many epochs to run.")
mainarg.add_argument("--play_games", type=int, default=0, help="How many games to play, suppresses training and testing.")
mainarg.add_argument("--load_weights", help="Load network from file.")
mainarg.add_argument("--save_weights_prefix", help="Save network to given file. Network topology, epoch and extension will be appended.")
mainarg.add_argument("--csv_file", help="Write training progress to this file.")

comarg = parser.add_argument_group('Common')
comarg.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")
comarg.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Log level.")
args = parser.parse_args()

log = logging.getLogger()
log.setLevel(args.log_level)
log.handlers.pop()

if args.random_seed:
    random.seed(args.random_seed)

# Instantiate agent and environment.
env = Gym(args.env_id, args.screen_width, args.screen_height)
mem = ReplayMemory(args.replay_size, args)
q_net = network_factory(args.qnet_type, shape=(1, 1))
agent = Agent(env, mem, q_net, display_screen=args.display_screen,
              init_epsilon=args.init_epsilon, end_epsilon=args.end_epsilon,
              epsilon_decay_steps=args.epsilon_decay_steps,
              train_frequency=args.train_frequency,train_repeat=args.train_repeat,
              history_length=args.history_length,test_epsilon=args.test_epsilon)

# Load weights from trained network
if args.load_weights:
    log.info("Loading weights from %s" % args.load_weights)
    q_net.load_weights(args.load_weights)

# Play games without train / test
if args.play_games:
    log.info("Playing %d games" % args.play_games)
    agent.play(args.play_games)
    sys.exit(0)

# Populate the agent's replay memory with random steps before learning
if args.random_steps:
    log.info("Populating replay mem with %d random moves" % args.random_steps)
    agent.play_random(args.random_steps)

# Train test loop over epochs
for epoch in xrange(args.epochs):
    log.info("Epoch #%d / %d" % ((epoch + 1), args.epochs))

    if args.train_steps:
        log.info("Training for %d steps" % args.train_steps)
        agent.train(args.train_steps)

        if args.save_weights_prefix:
            filename = args.save_weights_prefix + "%s_%d.pkl" % (args.qnet_type, (epoch + 1))
            log.info("Saving weights to %s" % filename)
            q_net.save_weights(filename)

    if args.test_steps:
        log.info("Testing for %d steps" % args.test_steps)
        agent.test(args.test_steps)

