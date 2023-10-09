#!/usr/bin/env python3
import argparse
import os
import sys
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=True, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=100, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--entropy_regularization", default=0.005, type=float, help="Entropy regularization weight.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=5, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=10, type=int, help="Size of hidden layer.")
parser.add_argument("--embedding_output_dim", default=1, type=int, help="Size of embedding output dimension.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--tiles", default=4, type=int, help="Tiles to use.")
parser.add_argument("--workers", default=16, type=int, help="Number of parallel workers.")
parser.add_argument("--model_filename", default="paac_cont.keras", type=str, help="Path to model file")
parser.add_argument("--target_return", default=20, type=int)

class EncodingLayer(tf.keras.layers.Layer):

    def __init__(self, one_hot_size, oh_indices, **kwargs):
        super(EncodingLayer, self).__init__(**kwargs)
        self._one_hot_size = int(one_hot_size)
        self._oh_indices = oh_indices

    def call(self, inputs):
        inputs = tf.math.add(inputs, self._oh_indices)
        inputs = tf.cast(inputs, tf.int32)
        oh = tf.one_hot(inputs, self._one_hot_size, dtype=tf.int32)
        return tf.reduce_max(oh, axis=1) # get tiles_hot_encoding
    
    def get_config(self):
        config = super(EncodingLayer, self).get_config()
        config.update({"one_hot_size": self._one_hot_size, "oh_indices": self._oh_indices})
        return config

class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Analogously to paac, your model should contain two components:
        # - actor, which predicts distribution over the actions
        # - critic, which predicts the value function
        #
        # The given states are tile encoded, so they are integral indices of
        # tiles intersecting the state. Therefore, you should convert them
        # to dense encoding (one-hot-like, with with `args.tiles` ones).
        # (Or you can even use embeddings for better efficiency.)
        #
        # The actor computes `mus` and `sds`, each of shape [batch_size, actions].
        # Compute each independently using states as input, adding a fully connected
        # layer with `args.hidden_layer_size` units and ReLU activation. Then:
        # - For `mus`, add a fully connected layer with `actions` outputs.
        #   To avoid `mus` moving from the required range, you should apply
        #   properly scaled `tf.tanh` activation.
        # - For `sds`, add a fully connected layer with `actions` outputs
        #   and `tf.nn.softplus` activation.
        #
        # The critic should be a usual one, passing states through one hidden
        # layer with `args.hidden_layer_size` ReLU units and then predicting
        # the value function.
        in_shape = env.observation_space.shape
        inputs = tf.keras.layers.Input(shape=in_shape)

        one_hot_size = tf.reduce_sum(env.observation_space.nvec)
        oh_indices = []
        cumulative_previous_oh_bins = 0
        for i in range(len(env.observation_space.nvec)):
            oh_indices.append(cumulative_previous_oh_bins)
            cumulative_previous_oh_bins += env.observation_space.nvec[i]
        encoding = EncodingLayer(one_hot_size+args.tiles, oh_indices)(inputs)
        #oh_indices = inputs + cumulative_previous_oh_bins
        #print(oh_indices.shape)
        #encoding = tf.one_hot(tf.cast(oh_indices, dtype=tf.int32), one_hot_size, dtype=tf.int32)
        
        #embedded = []
        #for i in range(len(env.observation_space)):
        #    embedded.append(
        #        tf.keras.layers.Embedding(
        #            env.observation_space.nvec[i], 
        #            args.embedding_output_dim
        #        )(inputs[:,i])
        #    )
        #embedding_concat = tf.concat(embedded, 1)
        
        # MUS
        hidden_mus = tf.keras.layers.Dense(args.hidden_layer_size, activation="relu")(encoding)
        output_mus = tf.keras.layers.Dense(env.action_space.shape[0], activation=tf.keras.activations.tanh, name="output_mus")(hidden_mus)
        #output_mus = tf.keras.activations.tanh(hidden_mus)
        #tanh_range = 2
        #hidden_mus_normalized = tf.math.divide(tf.math.add(hidden_mus, 1), tanh_range)
        #action_space_range = env.action_space.high - env.action_space.low
        #output_mus = tf.math.add(tf.math.multiply(hidden_mus_normalized, action_space_range), env.action_space.low)

        #SDS
        hidden_sds = tf.keras.layers.Dense(args.hidden_layer_size, activation="relu")(encoding)
        output_sds = tf.keras.layers.Dense(env.action_space.shape[0], activation=tf.nn.softplus, name="output_sds")(hidden_sds)
        self._actor = tf.keras.Model(inputs, [output_mus, output_sds])
        self._actor.compile(
            optimizer=tf.optimizers.Adam(learning_rate=args.learning_rate)
        )
        
        inputs = tf.keras.layers.Input(shape=in_shape)
        encoding = EncodingLayer(one_hot_size, oh_indices)(inputs)
        hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation="relu")(encoding)
        outputs = tf.keras.layers.Dense(1, activation=None)(hidden)
        self._critic = tf.keras.Model(inputs, outputs)

        self._critic.compile(
            optimizer=tf.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.losses.MeanSquaredError(),
            metrics=tf.metrics.MeanSquaredError()
        )


    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @tf.function
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # TODO: Run the model on given `states` and compute
        # sds, mus and predicted values. Then create `action_distribution` using
        # `tfp.distributions.Normal` class and computed mus and sds.
        # In PyTorch, the corresponding class is `torch.distributions.normal.Normal`.
        #
        # TODO: Train the actor using the sum of the following two losses:
        # - negative log likelihood of the `actions` in the `action_distribution`
        #   (using the `log_prob` method). You then need to sum the log probabilities
        #   of actions in a single batch example (using `tf.math.reduce_sum` with `axis=1`).
        #   Finally multiply the resulting vector by (returns - predicted values)
        #   and compute its mean. Note that the gradient must not flow through
        #   the predicted values (you can use `tf.stop_gradient` if necessary).
        # - negative value of the distribution entropy (use `entropy` method of
        #   the `action_distribution`) weighted by `args.entropy_regularization`.
        #
        # Train the critic using mean square error of the `returns` and predicted values.
        
        with tf.GradientTape() as tape:
            pred_mus, pred_sds = self._actor(states, training=True)
            action_dist = tfp.distributions.Normal(pred_mus, pred_sds)
            loss = - tf.math.reduce_sum(action_dist.log_prob(actions), axis=1)
            pred_values = self._critic(states) # TODO presunout stop gradient?
            loss = tf.math.reduce_mean(loss * tf.stop_gradient(returns - pred_values)) # TODO what axis if any?
            loss += - args.entropy_regularization * tf.math.reduce_sum(action_dist.entropy()) # TODO mean of entropy?
        self._actor.optimizer.minimize(loss, self._actor.trainable_variables, tape=tape)

        with tf.GradientTape() as tape:
            value_pred = self._critic(states, training=True)
            loss = self._critic.compiled_loss(y_true=returns, y_pred=value_pred)
        self._critic.optimizer.minimize(loss, self._critic.trainable_variables, tape=tape)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_actions(self, states: np.ndarray) -> np.ndarray:
        # TODO: Return predicted action distributions (mus and sds).
        return self._actor(states)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_values(self, states: np.ndarray) -> np.ndarray:
        # TODO: Return predicted state-action values.
        return self._critic(states)

def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    def evaluate_episode(start_evaluation:bool = False) -> float:
        rewards, state, done = 0, env.reset(start_evaluation), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: Predict the action using the greedy policy
            mus, sds = network.predict_actions([state])
            mus = np.clip(mus, env.action_space.low, env.action_space.high)
            state, reward, done, _ = env.step(mus)
            rewards += reward
        return rewards

    # Create the vectorized environment
    vector_env = gym.vector.AsyncVectorEnv(
        [lambda: wrappers.DiscreteMountainCarWrapper(gym.make("MountainCarContinuous-v0"), tiles=args.tiles)] * args.workers)
    vector_env.seed(args.seed)
    states = vector_env.reset()

    #rewards = np.zeros((states.shape[0]))
    training = not args.recodex
    while training:
        # Training
        for _ in range(args.evaluate_each):
            # TODO: Predict action distribution using `network.predict_actions`
            # and then sample it using for example `np.random.normal`. Do not
            # forget to clip the actions to the `env.action_space.{low,high}`
            # range, for example using `np.clip`.
            mus, sds = network.predict_actions(states)
            samples = np.random.normal(mus[:,0], sds[:,0])
            actions = np.clip(samples, env.action_space.low, env.action_space.high)
            actions = np.expand_dims(actions, 1)

            # TODO(paac): Perform steps in the vectorized environment
            next_states, step_rewards, done, _ = vector_env.step(actions)

            # TODO(paac): Compute estimates of returns by one-step bootstrapping
            pred_next_value = network.predict_values(next_states)
            #print(pred_next_value[0], mus[0], sds[0])
            pred_next_value_reshaped = np.reshape(pred_next_value, (pred_next_value.shape[0]))
            returns = step_rewards + args.gamma * (done == False) * pred_next_value_reshaped

            # TODO(paac): Train network using current states, chosen actions and estimated returns
            network.train(states, actions, returns)
            
            states = next_states

        # Periodic evaluation
        for _ in range(args.evaluate_for):
            evaluate_episode()

        if np.mean(env._episode_returns[-env._evaluate_for:]) > args.target_return:
            print("Target return reached -> ending training")
            network._actor.save(args.model_filename)
            break

    if args.recodex:
        network._actor = tf.keras.models.load_model(
            args.model_filename,
            custom_objects={"EncodingLayer":EncodingLayer}
        )

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(wrappers.DiscreteMountainCarWrapper(gym.make("MountainCarContinuous-v0"), tiles=args.tiles), args.seed)
    #tf.compat.v1.enable_eager_execution()

    main(env, args)
