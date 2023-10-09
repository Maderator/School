#!/usr/bin/env python3
import argparse
import collections
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=50, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--env", default="Pendulum-v0", type=str, help="Environment.")
parser.add_argument("--evaluate_each", default=50, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=50, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.95, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=100, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--noise_sigma", default=0.2, type=float, help="UB noise sigma.")
parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
parser.add_argument("--target_tau", default=0.05, type=float, help="Target network update weight.")
parser.add_argument("--target_return", default=-100, type=int, help="Needed score for training to end.")

def dpg_loss(actor, critic, states):
    action_pred = actor(states)
    value_pred = critic([states, action_pred])
    return - tf.reduce_mean(value_pred)

class ScaleLayer(tf.keras.layers.Layer):

    def __init__(self, new_min, new_max, old_min=-1, old_max=1, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        self._old_min = old_min
        self._old_max = old_max

        self._new_min = new_min
        self._new_max = new_max

        self._old_size =old_max - old_min
        self._new_size =new_max - new_min

    def call(self, inputs):
        return (self._new_size * (inputs - self._old_min)) / self._old_size + self._new_min

    def get_config(self):
        config = super(ScaleLayer, self).get_config()
        config.update(
            {"old_min" : self._old_min,
            "old_max"  : self._old_max,
            "new_min"  : self._new_min,
            "new_max"  : self._new_max}
        )
        return config



class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create:
        # - an actor, which starts with states and returns actions.
        #   Usually, one or two hidden layers are employed. As in the
        #   paac_continuous, to keep the actions in the required range, you
        #   should apply properly scaled `tf.tanh` activation.
        #
        # - a target actor as the copy of the actor using `tf.keras.models.clone_model`.
        #
        # - a critic, starting with given states and actions, producing predicted
        #   returns. The states and actions are usually concatenated and fed through
        #   two more hidden layers, before computing the returns with the last output layer.
        #
        # - a target critic as the copy of the critic using `tf.keras.models.clone_model`.
        in_shape = env.observation_space.shape
        inputs = tf.keras.layers.Input(shape=in_shape)
        hidden = tf.keras.layers.Dense(
            args.hidden_layer_size, activation=tf.keras.activations.relu)(inputs)
        hidden = tf.keras.layers.Dense(
            args.hidden_layer_size, activation=tf.keras.activations.relu)(hidden)
        output = tf.keras.layers.Dense(env.action_space.shape[0], activation=tf.keras.activations.tanh)(hidden)
        output_scaled = ScaleLayer(env.action_space.low, env.action_space.high, -1, 1)(output)

        self._actor = tf.keras.Model(inputs, output_scaled)
        self._actor.compile(
            optimizer=tf.optimizers.Adam(learning_rate=args.learning_rate)
        )

        self._target_actor = tf.keras.models.clone_model(self._actor)

        inputs_observations = tf.keras.layers.Input(shape=env.observation_space.shape)
        hidden_observations = tf.keras.layers.Dense(
            args.hidden_layer_size, activation=tf.keras.activations.relu)(inputs_observations)
        inputs_actions = tf.keras.layers.Input(shape=env.action_space.shape)
        concat = tf.keras.layers.Concatenate()([hidden_observations, inputs_actions])
        hidden = tf.keras.layers.Dense(
            args.hidden_layer_size, activation=tf.keras.activations.relu)(concat)
        hidden = tf.keras.layers.Dense(
            args.hidden_layer_size, activation=tf.keras.activations.relu)(hidden)
        output = tf.keras.layers.Dense(1, activation=None)(hidden)
        self._critic = tf.keras.Model([inputs_observations, inputs_actions], output)

        self._critic.compile(
            optimizer=tf.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.losses.MeanSquaredError()
        )

        self._target_critic = tf.keras.models.clone_model(self._critic)


    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @tf.function
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # TODO: Separately train:
        # - the actor using the DPG loss,
        # - the critic using MSE loss.
        #
        # Furthermore, update the target actor and critic networks by
        # exponential moving average with weight `args.target_tau`. A possible
        # way to implement it inside a `tf.function` is the following:
        #   for var, target_var in zip(network.trainable_variables, target_network.trainable_variables):
        #       target_var.assign(target_var * (1 - target_tau) + var * target_tau)
        with tf.GradientTape() as tape:
            action_pred = self._actor(states, training=True)
            value_pred = self._critic([states, action_pred])
            loss = -tf.reduce_mean(value_pred)
        self._actor.optimizer.minimize(loss, self._actor.trainable_variables, tape=tape)

        with tf.GradientTape() as tape:
            value_pred = self._critic([states, actions], training=True)
            loss = self._critic.compiled_loss(y_true=returns, y_pred=value_pred)
        self._critic.optimizer.minimize(loss, self._critic.trainable_variables, tape=tape)

        for var, target_var in zip(self._actor.trainable_variables, self._target_actor.trainable_variables):
            target_var.assign(target_var * (1 - args.target_tau) + var * args.target_tau)
        
        for var, target_var in zip(self._critic.trainable_variables, self._target_critic.trainable_variables):
            target_var.assign(target_var * (1 - args.target_tau) + var * args.target_tau)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_actions(self, states: np.ndarray) -> np.ndarray:
        # TODO: Return predicted actions by the actor.
        return self._actor(states)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_values(self, states: np.ndarray) -> np.ndarray:
        # TODO: Return predicted returns -- predict actions by the target actor
        # and evaluate them using the target critic.
        return self._target_critic([states, self._target_actor(states)])


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, mu, theta, sigma):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return self.state


def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    def evaluate_episode(start_evaluation:bool = False) -> float:
        rewards, state, done = 0, env.reset(start_evaluation), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: Predict the action using the greedy policy
            action = network.predict_actions([state])
            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    noise = OrnsteinUhlenbeckNoise(env.action_space.shape[0], 0, args.noise_theta, args.noise_sigma)
    training = not args.recodex
    while training:
        # Training
        for _ in range(args.evaluate_each):
            state, done = env.reset(), False
            noise.reset()
            while not done:
                # TODO: Predict actions by calling `network.predict_actions`
                # and adding the Ornstein-Uhlenbeck noise. As in paac_continuous,
                # clip the actions to the `env.action_space.{low,high}` range.
                action = network.predict_actions([state])
                action = np.clip(
                    action + noise.sample(),
                    env.action_space.low,
                    env.action_space.high
                )
                action = tf.reshape(action, -1)

                next_state, reward, done, _ = env.step(action)
                replay_buffer.append(Transition(state, action, reward, done, next_state))
                state = next_state

                if len(replay_buffer) >= args.batch_size:
                    batch = np.random.choice(len(replay_buffer), size=args.batch_size, replace=False)
                    states, actions, rewards, dones, next_states = map(np.array, zip(*[replay_buffer[i] for i in batch]))
                    # TODO: Perform the training
                    pred_next_value = network.predict_values(next_states)
                    pred_next_value_reshaped = np.reshape(pred_next_value, -1)
                    returns = rewards + args.gamma * (dones == False) * pred_next_value_reshaped
                    network.train(states, actions, returns)

        # Periodic evaluation
        for _ in range(args.evaluate_for):
            evaluate_episode()

        if np.mean(env._episode_returns[-env._evaluate_for:]) > args.target_return:
            network._actor.save_weights("actor_model.keras")
            break

    # Final evaluation
    if args.recodex:
        network._actor.load_weights("actor_model.keras")

    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed)

    main(env, args)
