#!/usr/bin/env python3
import argparse
from contextlib import redirect_stderr
import os
import sys

from tensorflow.python.keras.backend import update
from tensorflow.python.ops.linalg_ops import norm
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import memory_game_environment
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--cards", default=8, type=int, help="Number of cards in the memory game.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--batch_size", default=16, type=int, help="Number of episodes to train on.")
parser.add_argument("--evaluate_each", default=512, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=25, type=int, help="Evaluate for number of episodes.")
parser.add_argument("--hidden_layer", default=None, type=int, help="Hidden layer size; default 8*`cards`")
parser.add_argument("--memory_cells", default=None, type=int, help="Number of memory cells; default 2*`cards`")
parser.add_argument("--memory_cell_size", default=None, type=int, help="Memory cell size; default 3/2*`cards`")

class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        self.args = args
        self.env = env

        # Define the agent inputs: a memory and a state.
        memory = keras.layers.Input(shape=[args.memory_cells, args.memory_cell_size], dtype=tf.float32)
        state = keras.layers.Input(shape=env.observation_space.shape, dtype=tf.int32)

        # Encode the input state, which is a (card, observation) pair,
        # by representing each element as one-hot and concatenating them, resulting
        # in a vector of length `sum(env.observation_space.nvec)`.
        encoded_input = tf.keras.layers.Concatenate()(
            [tf.one_hot(state[:, i], dim) for i, dim in enumerate(env.observation_space.nvec)])

        # TODO: Generate a read key for memory read from the encoded input, by using
        # a ReLU hidden layer of size `args.hidden_layer` followed by a dense layer
        # with `args.memory_cell_size` units and `tanh` activation (to keep the memory
        # content in limited range).
        hidden = keras.layers.Dense(args.hidden_layer, activation="relu")(encoded_input)
        read_key = keras.layers.Dense(args.memory_cell_size, activation="tanh")(hidden) 


        # TODO: Read the memory using the generated read key. Notably, compute cosine
        # similarity of the key and every memory row, apply softmax to generate
        # a weight distribution over the rows, and finally take a weighted average of
        # the memory rows.
        normalized_memory = tf.nn.l2_normalize(memory, 2)
        normalized_key = tf.nn.l2_normalize(read_key, 1)
        normalized_key = tf.expand_dims(normalized_key, 1)
        cos_similarity = tf.matmul(normalized_key, normalized_memory, transpose_b=[0, 2, 1])
        weight_dist = tf.nn.softmax(cos_similarity)
        read_value = tf.reduce_mean(weight_dist @ memory, 1) # just 1 element in the 1st dimension

        # TODO: Using concatenated encoded input and the read value, use a ReLU hidden
        # layer of size `args.hidden_layer` followed by a dense layer with
        # `env.action_space.n` units and `softmax` activation to produce a policy.
        conc = keras.layers.Concatenate()([encoded_input, read_value])
        hidden = keras.layers.Dense(args.hidden_layer, "relu")(conc)
        policy = keras.layers.Dense(env.action_space.n, "softmax")(hidden)

        # TODO: Perform memory write. For faster convergence, append directly
        # the `encoded_input` to the memory, i.e., add it as a first memory row, and drop
        # the last memory row to keep memory size constant.
        updated_memory = memory[:,:-1,:]
        expanded_encoded_input = tf.expand_dims(encoded_input, axis=1)
        updated_memory = keras.layers.Concatenate(axis=1)([expanded_encoded_input, updated_memory])

        # Create the agent
        self._agent = tf.keras.Model(inputs=[memory, state], outputs=[updated_memory, policy])
        self._agent.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.SparseCategoricalCrossentropy())

    def zero_memory(self):
        # TODO: Return an empty memory. It should be a TF tensor
        # with shape `[self.args.memory_cells, self.args.memory_cell_size]`.
        return tf.zeros(shape=[self.args.memory_cells, self.args.memory_cell_size])

    @wrappers.typed_np_function(np.float32, np.int32, np.int32, np.int32)
    @tf.function
    def _train(self, states, targets, max_episode_len, sample_weights):
        # TODO: Given a batch of sequences of `states` (each being a (card, symbol) pair),
        # train the network to predict the required `targets`.
        #
        # Specifically, start with a batch of empty memories, and run the agent
        # sequentially as many times as necessary, using `targets` as gold labels.
        #
        # Note that the sequences can be of different length, so you need to pad them
        # to same length and then somehow indicate the length of the individual episodes
        # (one possibility is to add another parameter to `_train`).
        
        batch_size = len(states)
        memories = tf.tile(
            tf.expand_dims(self.zero_memory(),0),
            [batch_size,1,1]
        )

        for i in range(max_episode_len):
            with tf.GradientTape() as tape:
                state = states[:,i,:]
                memories, policy = self._agent([memories, state])
                loss = self._agent.loss(targets[:, i], policy, sample_weight=sample_weights[:,i])
            self._agent.optimizer.minimize(
                loss, var_list=self._agent.trainable_variables, tape=tape)

    def train(self, episodes):
        # TODO: Given a list of episodes, prepare the arguments
        # of the self._train method, and execute it.
        def correct_ep_len(episode, correct_len):
            ep_len = len(episode)
            last_state_action = episode[-1]
            for _ in range(correct_len - ep_len):
                episode.append(last_state_action)
            return episode

        def correct_episodes_len(episodes):
            episode_max_length = 0
            for ep in episodes:
                ep_len = len(ep)
                if ep_len > episode_max_length:
                    episode_max_length = ep_len

            for i in range(len(episodes)):
                episodes[i] = correct_ep_len(episodes[i], episode_max_length)
            
            return episodes, episode_max_length
        
        def unzip_states_and_actions(episodes):
            states, actions = [], []
            for ep in episodes:
                s, a = list(zip(*ep)) 
                states.append(s)
                actions.append(a)
            return states, actions

        def get_episodes_end(episodes):
            episodes_end = []
            for ep in episodes:
                episodes_end.append(len(ep))
            return episodes_end

        def change_none_action_to_last_action(actions):
            for i in range(len(actions)):
                actions[i] = list(actions[i])
                for j in range(len(actions[i])):
                    if actions[i][j] is None:
                        actions[i][j] = self.env.action_space.n-1

        #def actions_to_one_hot(actions):
        #    for i in range(len(actions)):
        #        for j in range(len(actions[i])):
        #            actions[i][j] = tf.one_hot(actions[i][j], self.env.action_space.n)

        def create_sample_weights(episodes_end, max_len):
            sample_weights = np.ones((len(episodes_end),max_len))
            for i in range(sample_weights.shape[0]):
                sample_weights[i,episodes_end[i]:] = 0
            return sample_weights

        lengths = get_episodes_end(episodes)
        episodes_padded, max_len = correct_episodes_len(episodes)
        states, actions = unzip_states_and_actions(episodes_padded)
        change_none_action_to_last_action(actions)

        # TODO change action to one_hot representation
        sample_weights = create_sample_weights(lengths, max_len)

        self._train(states, actions, max_len, sample_weights)

    @wrappers.typed_np_function(np.float32, np.int32)
    @tf.function
    def predict(self, memory, state):
        return self._agent([memory, state])


def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Post-process arguments to default values if not overridden on the command line.
    if args.hidden_layer is None:
        args.hidden_layer = 8 * args.cards
    if args.memory_cells is None:
        args.memory_cells = 2 * args.cards
    if args.memory_cell_size is None:
        args.memory_cell_size = 3 * args.cards // 2
    assert sum(env.observation_space.nvec) == args.memory_cell_size

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(evaluating=False):
        rewards = 0
        state, memory, done = env.reset(evaluating), [network.zero_memory()], False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: Find out which action to use
            memory, policy = network.predict(memory, [state])
            action = np.argmax(policy)

            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    # Training
    training = True
    episode = 0
    while training:
        # Generate required number of episodes
        for _ in range(args.evaluate_each // args.batch_size):
            episode += 1
            episodes = []
            for _ in range(args.batch_size):
                episodes.append(env.expert_episode())

            # Train the network
            network.train(episodes)

        # TODO: Maybe evaluate the current performance, using
        # `evaluate_episode()` method returning the achieved return,
        # and setting `training=False` when the performance is high enough.
        rewards = []
        for _ in range(args.evaluate_for):
            rewards.append(evaluate_episode())
        rewards_mean = np.mean(rewards)
        #print(f"After {episode} training episodes return is {rewards_mean}")
        if rewards_mean > 1.5:
            training = False

    # Final evaluation
    while True:
        evaluate_episode(True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(memory_game_environment.make(args.cards), args.seed, evaluate_for=args.evaluate_for, report_each=args.evaluate_for)

    #sys.stderr = open('stderr.txt', 'w')
    main(env, args)
