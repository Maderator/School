#!/usr/bin/env python3

# IDs:
#edbe2dad-018e-11eb-9574-ea7484399335
#aa20f311-c7a2-11e8-a4be-00505601122b
#c716f6b0-25ab-11ec-986f-f39926f24a9c

import argparse
import collections
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import memory_game_environment
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--cards", default=6, type=int, help="Number of cards in the memory game.")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--learning_rate", default=0.0001, type=float, help="Gradient clipping.")
parser.add_argument("--batch_size", default=1024, type=int, help="Number of episodes to train on.")
parser.add_argument("--gradient_clipping", default=1.0, type=float, help="Gradient clipping.")
parser.add_argument("--entropy_regularization", default=0.1, type=float, help="Entropy regularization weight.")
parser.add_argument("--evaluate_each", default=1000, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=100, type=int, help="Evaluate for number of episodes.")
parser.add_argument("--hidden_layer", default=None, type=int, help="Hidden layer size; default 8*`cards`")
parser.add_argument("--memory_cells", default=None, type=int, help="Number of memory cells; default 2*`cards`")
parser.add_argument("--memory_cell_size", default=None, type=int, help="Memory cell size; default 3/2*`cards`")
parser.add_argument("--replay_buffer", default=None, type=int, help="Max replay buffer size; default batch_size")
parser.add_argument("--ema_momentum", default=0.01, type=float, help="Exponential moving average momentum")
parser.add_argument("--gamma", default=0.25, type=float, help="discounting factor")

class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        self.args = args
        self.env = env

        # Define the agent inputs: a memory and a state.
        memory = tf.keras.layers.Input(shape=[args.memory_cells, args.memory_cell_size], dtype=tf.float32)
        state = tf.keras.layers.Input(shape=env.observation_space.shape, dtype=tf.int32)

        # Encode the input state, which is a (card, observation) pair,
        # by representing each element as one-hot and concatenating them, resulting
        # in a vector of length `sum(env.observation_space.nvec)`.
        encoded_input = tf.keras.layers.Concatenate()(
            [tf.one_hot(state[:, i], dim) for i, dim in enumerate(env.observation_space.nvec)])

        # TODO(memory_game): Generate a read key for memory read from the encoded input, by using
        # a ReLU hidden layer of size `args.hidden_layer` followed by a dense layer
        # with `args.memory_cell_size` units and `tanh` activation (to keep the memory
        # content in limited range).
        hidden = keras.layers.Dense(args.hidden_layer, activation="relu")(encoded_input)
        read_key = keras.layers.Dense(args.memory_cell_size, activation="tanh")(hidden) 

        # TODO(memory_game): Read the memory using the generated read key. Notably, compute cosine
        # similarity of the key and every memory row, apply softmax to generate
        # a weight distribution over the rows, and finally take a weighted average of
        # the memory rows.
        normalized_memory = tf.nn.l2_normalize(memory, 2)
        normalized_key = tf.nn.l2_normalize(read_key, 1)
        normalized_key = tf.expand_dims(normalized_key, 1)
        cos_similarity = tf.matmul(normalized_key, normalized_memory, transpose_b=[0, 2, 1])
        weight_dist = tf.nn.softmax(cos_similarity)
        read_value = tf.reduce_mean(weight_dist @ memory, 1) # just 1 element in the 1st dimension

        # TODO(memory_game): Using concatenated encoded input and the read value, use a ReLU hidden
        # layer of size `args.hidden_layer` followed by a dense layer with
        # `env.action_space.n` units and `softmax` activation to produce a policy.
        conc = keras.layers.Concatenate()([encoded_input, read_value])
        hidden = keras.layers.Dense(args.hidden_layer, "relu")(conc)
        policy = keras.layers.Dense(env.action_space.n, "softmax")(hidden)

        # TODO(memory_game): Perform memory write. For faster convergence, append directly
        # the `encoded_input` to the memory, i.e., add it as a first memory row, and drop
        # the last memory row to keep memory size constant.
        updated_memory = memory[:,:-1,:]
        expanded_encoded_input = tf.expand_dims(encoded_input, axis=1)
        updated_memory = keras.layers.Concatenate(axis=1)([expanded_encoded_input, updated_memory])

        # Create the agent
        self._agent = tf.keras.Model(inputs=[memory, state], outputs=[updated_memory, policy])
        self._agent.compile(
            optimizer=tf.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=args.gradient_clipping),
            loss=tf.losses.SparseCategoricalCrossentropy(),
        )

    @classmethod
    def load(cls, path: str,  env: wrappers.EvaluationEnv, args: argparse.Namespace):
        # A static method returning a new Agent loaded from the given path.
        network = Network.__new__(Network)
        network._agent = tf.keras.models.load_model(path)
        network.env = env
        network.args = args
        return network

    def save(self, path: str, include_optimizer=True) -> None:
        # Save the agent model as a h5 file, possibly with/without the optimizer.
        self._agent.save(path, include_optimizer=include_optimizer, save_format="h5")

    def zero_memory(self, batch_size):
        # TODO(memory_game): Return an empty memory. It should be a TF tensor
        # with shape `[self.args.memory_cells, self.args.memory_cell_size]`.
        return tf.zeros(shape=[batch_size, self.args.memory_cells, self.args.memory_cell_size])

    @wrappers.typed_np_function(np.float32, np.int32, np.float32, np.int32, np.float32)
    @tf.function
    def _train(self, states, actions, returns, max_episode_len, sample_weights):
        # TODO: Train the network given a batch of sequences of `states`
        # (each being a (card, symbol) pair), sampled `actions` and observed `returns`.
        # Specifically, start with a batch of empty memories, and run the agent
        # sequentially as many times as necessary, using `actions` as actions.
        #
        # Use the REINFORCE algorithm, optionally with a baseline. Note that
        # I use a baseline, but not a baseline computed by a neural network;
        # instead, for every time step, I track exponential moving average of
        # observed returns, with momentum 0.01. Furthermore, I use entropy regularization
        # with coefficient `args.entropy_regularization`.
        #
        # Note that the sequences can be of different length, so you need to pad them
        # to same length and then somehow indicate the length of the individual episodes
        # (one possibility is to add another parameter to `_train`).

        @tf.function
        def log2(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
            return numerator / denominator

        memories = self.zero_memory(self.args.batch_size)
        
        ema = returns[:,0] 
        #ema = np.zeros(self.args.batch_size, dtype=np.float32)
        for i in range(max_episode_len):
            with tf.GradientTape() as tape:
                state = states[:,i,:]
                memories, policy = self._agent([memories, state])
                ema = ema * self.args.ema_momentum  + returns[:,i] * (1 - self.args.ema_momentum)
                loss = self._agent.loss(actions[:,i], policy, sample_weight=(returns[:,i]-ema)*sample_weights[:,i])
                policy_entropy = tf.math.reduce_sum((-policy * log2(policy)), axis=-1)
                loss -= self.args.entropy_regularization * policy_entropy
            self._agent.optimizer.minimize(
                loss, var_list=self._agent.trainable_variables, tape=tape)

    def train(self, episodes):
        # TODO: Given a list of episodes, prepare the arguments
        # of the self._train method, and execute it.
        states, actions, rewards, returns, sample_weights = [], [], [], [], []
        max_len = max(len(ep) for ep in episodes)
        for ep in episodes:
            ep_states, ep_actions, ep_rewards, ep_returns, ep_sweights = [], [], [], [], []
            for i, move in enumerate(ep):
                ep_states.append(move[0])
                # Action is never None -> no need to check
                ep_actions.append(move[1])
                ep_sweights.append(1)
                ep_rewards.append(move[2])
                ep_returns.append(move[3])
            while i < max_len-1:
                i += 1
                ep_states.append(ep_states[i-1])
                ep_actions.append(ep_actions[i-1])
                ep_rewards.append(0)
                ep_returns.append(0)
                ep_sweights.append(0)
            states.append(ep_states)
            actions.append(ep_actions)
            rewards.append(ep_rewards)
            returns.append(ep_returns)
            sample_weights.append(ep_sweights)

        self._train(states, actions, returns, max_len, sample_weights)

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
        args.hidden_layer = 32 * args.cards #8 * args.cards
    if args.memory_cells is None:
        args.memory_cells = 2 * args.cards
    if args.memory_cell_size is None:
        args.memory_cell_size = 3 * args.cards // 2
    if args.replay_buffer is None:
        args.replay_buffer = args.batch_size
    assert sum(env.observation_space.nvec) == args.memory_cell_size

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(evaluating=False):
        rewards = 0
        state, memory, done = env.reset(evaluating), network.zero_memory(1), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO(memory_game): Find out which action to use
            memory, policy = network.predict(memory, [state])
            action = np.argmax(policy)

            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    # Training
    replay_buffer = collections.deque(maxlen=args.replay_buffer)

    if args.recodex:
        network = Network.load(path=f"mem_game_rl_{args.cards}cards.model", env=env, args=args)
        training = False
    else:
        network = Network.load(path=f"mem_game_rl_{args.cards}cards.model", env=env, args=args)
        training = True

    while training:
        # Generate required number of episodes
        for _ in range(args.evaluate_each):
            state, memory, episode, done = env.reset(), network.zero_memory(1), [], False
            while not done:
                # TODO: Choose an action according to the generated distribution.
                memory, policy = network.predict(memory, [state])
                action = np.argmax(policy)

                next_state, reward, done, _ = env.step(action)
                episode.append([state, action, reward])
                state = next_state

            # TODO: In the `episode`, compute returns from the rewards.
            g = 0
            for i, step in reversed(list(enumerate(episode))):
                reward = step[2]
                g = g * args.gamma + reward
                episode[i].append(g)

            replay_buffer.append(episode)

            # Train the network if enough data is available
            if len(replay_buffer) >= args.batch_size:
                network.train([replay_buffer[i] for i in np.random.choice(len(replay_buffer), size=args.batch_size, replace=False)])

        # TODO(memory_game): Maybe evaluate the current performance, using
        # `evaluate_episode()` method returning the achieved return,
        # and setting `training=False` when the performance is high enough.
        rew = []
        for _ in range(args.evaluate_for):
            rew.append(evaluate_episode())
        rewards_mean = np.mean(rew)
        #print(f"After x training episodes return is {rewards_mean}")
        if rewards_mean > 0.5:
            Network.save(network, f"mem_game_rl_{network.args.cards}cards.model", include_optimizer=True)
            training = False

    # Final evaluation
    while True:
        evaluate_episode(True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(memory_game_environment.make(args.cards), args.seed, evaluate_for=args.evaluate_for, report_each=args.evaluate_for)

    main(env, args)
