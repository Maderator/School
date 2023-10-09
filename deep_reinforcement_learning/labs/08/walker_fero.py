#!/usr/bin/env python3
import argparse
import collections
import os

from tensorflow.python.keras.engine.training import concat
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfkeras
import tensorflow_probability as tfp

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--env", default="BipedalWalker-v3", type=str, help="Environment.")
parser.add_argument("--envs", default=8, type=int, help="Environments.")
parser.add_argument("--evaluate_each", default=200, type=int, help="Evaluate each number of updates.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=100, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="walker.model", type=str, help="Model path")
parser.add_argument("--target_entropy", default=-1, type=float, help="Target entropy per action component.")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")
parser.add_argument("--target_return", default=200, type=int)

def actor_loss(actor, critic1, critic2, states, minimize_network = True):
    # TODO: Separately train:
    # - the actor, by using two objectives:
    #   - the objective for the actor itself; in this objective, `tf.stop_gradient(alpha)`
    #     should be used (for the `alpha` returned by the actor) to avoid optimizing `alpha`,
    actions, log_probs, alpha = actor(states, sample=True)

    if minimize_network:
        val_critic1 = critic1([states, actions])[:, 0]
        val_critic2 = critic2([states, actions])[:, 0]
        val_critic_final = tf.math.minimum(val_critic1, val_critic2)

        actor_objective = tf.stop_gradient(alpha) * log_probs
        actor_objective -= val_critic_final
        actor_objective = tf.math.reduce_mean(actor_objective)

        return actor_objective

    #   - the objective for `alpha`, where `tf.stop_gradient(log_prob)` should be used
    #     to avoid computing gradient for other variables than `alpha`.
    #     Use `args.target_entropy` as the target entropy (the default of -1 per action
    #     component is fine and does not need to be tuned for the agent to train).
    alpha_objective = - alpha * tf.stop_gradient(log_probs)
    alpha_objective -= alpha * actor._target_entropy # per action component
    alpha_objective = tf.math.reduce_mean(alpha_objective)

    return alpha_objective

def actor_loss_network(actor, critic1, critic2, states):
    return actor_loss(actor, critic1, critic2, states, minimize_network=True)

def actor_loss_alpha(actor, states):
    return actor_loss(actor, None, None, states, minimize_network=False)

class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        # TODO: Create an actor. Because we will be sampling (and `sample()` from
        # `tfp.distributions` does not play nice with functional models) and because
        # we need the `alpha` variable, we use subclassing to create the actor.
        class Actor(tf.keras.Model):
            def __init__(self, hidden_layer_size: int, env: wrappers.EvaluationEnv):
                super().__init__()
                # TODO: Create
                # - two hidden layers with `hidden_layer_size` and ReLU activation
                # - a layer for generaing means with `env.action_space.shape[0]` units and no activation
                # - a layer for generaing sds with `env.action_space.shape[0]` units and `tf.math.exp` activation
                # - finally, create a variable represeting a logarithm of alpha, using for example the following:
                self._log_alpha = tf.Variable(np.log(0.1), dtype=tf.float32, name="log_alpha")
                self._target_entropy = args.target_entropy * env.action_space.shape[0]

                self._input_layer = tfkeras.layers.InputLayer(input_shape=env.observation_space.shape)
                self._hidden1 = tfkeras.layers.Dense(hidden_layer_size, activation=tfkeras.activations.relu)
                self._hidden2 = tfkeras.layers.Dense(hidden_layer_size, activation=tfkeras.activations.relu)
                self._means_output = tfkeras.layers.Dense(env.action_space.shape[0])
                self._sds_output = tfkeras.layers.Dense(env.action_space.shape[0], activation=tf.math.exp)
            
            def get_trainable_vars_without_alpha(self):
                return self._hidden1.trainable_variables + self._hidden2.trainable_variables + self._means_output.trainable_variables + self._sds_output.trainable_variables


            def call(self, inputs: tf.Tensor, sample: bool):
                # TODO: Perform the actor computation
                # - First, pass the inputs through the first hidden layer
                #   and then through the second hidden layer.
                # - From these hidden states, compute
                #   - `mus` (the means),
                #   - `sds` (the standard deviations).
                o = self._input_layer(inputs)
                o = self._hidden1(o)
                o = self._hidden2(o)
                means = self._means_output(o)
                sds = self._sds_output(o)

                # - Then, create the action distribution using `tfp.distributions.Normal`
                #   with the `mus` and `sds`. Note that to support computation without
                #   sampling, the easiest is to pass zeros as standard deviations when
                #   `sample == False`.
                if not sample:
                    return means, None, None

                actions_distribution = tfp.distributions.Normal(
                    means, sds
                )
                # - We then bijectively modify the distribution so that the actions are
                #   in the given range. Luckily, `tfp.bijectors` offers classes that
                #   can transform a distribution.
                
                #   - first run `tfp.bijectors.Tanh()(actions_distribution)` to squash the
                #     actions to [-1, 1] range,
                actions_distribution = tfp.bijectors.Tanh()(actions_distribution)
                
                #   - then run `tfp.bijectors.Scale((env.action_space.high - env.action_space.low) / 2)(actions_distribution)`
                #     to scale the action ranges to [-(high-low)/2, (high-low)/2],
                actions_distribution = tfp.bijectors.Scale( (env.action_space.high - env.action_space.low) / 2)(actions_distribution)
                
                #   - finally, `tfp.bijectors.Shift((env.action_space.high + env.action_space.low) / 2)(actions_distribution)`
                #     shifts the ranges to [low, high].
                actions_distribution = tfp.bijectors.Shift( (env.action_space.high + env.action_space.low) / 2)(actions_distribution)

                # - Sample the actions by a `sample()` call.
                actions = actions_distribution.sample()
                # - Then, compute the log-probabilities of the sampled actions by using `log_prob()`
                #   call. An action is actually a vector, so to be precise, compute for every batch
                #   element a scalar, an average of the log-probabilities of individual action components.
                log_probs = tf.math.reduce_sum(actions_distribution.log_prob(actions), axis=-1)
                # - Finally, compute `alpha` as exponentiation of `self._log_alpha`.
                alpha = tf.math.exp(self._log_alpha)
                # - Return actions, log_probs and alpha.
                return actions, log_probs, alpha

        # TODO: Instantiate the actor as `self._actor` and compile it.
        self._actor = Actor(args.hidden_layer_size, env)
        self._actor.compile(
            optimizer=tfkeras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=actor_loss_network
        )
        self._alpha_optimizer = tfkeras.optimizers.Adam(learning_rate=args.learning_rate)

        # TODO: Create a critic, which
        
        # - takes observations and actions as inputs,
        input_observations = tfkeras.layers.Input(shape=env.observation_space.shape)
        input_actions = tfkeras.layers.Input(shape=env.action_space.shape)
        
        # - concatenates them,
        concat_layer = tfkeras.layers.Concatenate()([input_observations, input_actions])
        
        # - passes the result through two dense layers with `args.hidden_layer_size` units
        #   and ReLU activation
        hidden = tfkeras.layers.Dense(args.hidden_layer_size, activation=tfkeras.activations.relu)(concat_layer)
        hidden = tfkeras.layers.Dense(args.hidden_layer_size, activation=tfkeras.activations.relu)(hidden)
        
        # - finally, using a last dense layer produces a single output with no activation
        output = tfkeras.layers.Dense(1)(hidden)
        self._critic1 = tfkeras.models.Model(inputs=[input_observations, input_actions], outputs=output)
        
        # This critic needs to be cloned so that two critics and two target critics are created.
        self._critic2 = tfkeras.models.clone_model(self._critic1)
        self._target_critic1 = tfkeras.models.clone_model(self._critic1)
        self._target_critic2 = tfkeras.models.clone_model(self._critic1)

        # compile non-target critics
        self._critic1.compile(
            optimizer=tfkeras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tfkeras.losses.MeanSquaredError()
        )
        self._critic2.compile(
            optimizer=tfkeras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tfkeras.losses.MeanSquaredError()
        )

    def save_actor(self, path: str):
        # Because we use subclassing for creating the actor, the easiest way of
        # serializing an actor is just to save weights.
        self._actor.save_weights(path, save_format="h5")

    def load_actor(self, path: str, env: wrappers.EvaluationEnv):
        # When deserializing, we need to make sure the variables are created
        # first -- we do so by processing a batch with a random observation.
        self.predict_actions([env.observation_space.sample()], sample=False)
        self._actor.load_weights(path)

    @wrappers.typed_np_function(np.float32, np.float32, np.float32, np.bool, np.float32)
    #@tf.function
    def train(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, next_states: np.ndarray) -> None:
        # TODO: Separately train:
        # - the actor

        next_actions, next_logprobs, alpha = self._actor(next_states, sample=True)
        ns_nq_values1 = self._target_critic1([next_states, next_actions])[:, 0]
        ns_nq_values2 = self._target_critic2([next_states, next_actions])[:, 0]
        ns_nq_values = tf.math.minimum(ns_nq_values1, ns_nq_values2) - alpha * next_logprobs
        returns = rewards + (1 - dones) * args.gamma * ns_nq_values

        self._actor.optimizer.minimize(
            lambda: self._actor.loss(self._actor, self._critic1, self._critic2, states),
            var_list=self._actor.get_trainable_vars_without_alpha()
        )

        self._alpha_optimizer.minimize(
            lambda: actor_loss_alpha(self._actor, states),
            var_list=[self._actor._log_alpha]
        )
        
        # - the critics using MSE loss.
        self._critic1.optimizer.minimize(
            lambda: self._critic1.loss(self._critic1([states, actions], training=True)[:, 0], returns),
            var_list=self._critic1.trainable_variables
        )
        self._critic2.optimizer.minimize(
            lambda: self._critic2.loss(self._critic2([states, actions], training=True)[:, 0], returns),
            var_list=self._critic2.trainable_variables
        )

        # Finally, update the two target critic networks exponential moving
        # average with weight `args.target_tau`, using something like
        #   for var, target_var in zip(critic.trainable_variables, target_critic.trainable_variables):
        #       target_var.assign(target_var * (1 - target_tau) + var * target_tau)
        for var, target_var in zip(self._critic1.trainable_variables, self._target_critic1.trainable_variables):
            target_var.assign(target_var * (1 - args.target_tau) + var * args.target_tau)
        
        for var, target_var in zip(self._critic2.trainable_variables, self._target_critic2.trainable_variables):
            target_var.assign(target_var * (1 - args.target_tau) + var * args.target_tau)

    # Note that wen calling `predict_actions`, the `sample` argument MUST always
    # be passed by name, so `sample=True/False`. This way it is not processed
    # by the wrappers.typed_np_function.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_actions(self, states: np.ndarray, sample: bool) -> np.ndarray:
        # Return predicted actions, assuming the actor is in `self._actor`.
        return self._actor(states, sample=sample)[0]

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_values(self, states: np.ndarray) -> np.ndarray:
        # TODO: Produce the predicted returns, which are the minimum of
        #    target_critic(s, a) - alpha * log_prob
        #  considering both target critics and actions sampled from the actor.
        actions, log_probs, alpha = self._actor(states, sample=True)
        
        value1 = self._target_critic1([states, actions])[:, 0]
        value1 -=  alpha * log_probs
        value2 = self._target_critic2([states, actions])[:, 0]
        value2 -= alpha * log_probs

        return tf.math.minimum(value1, value2)


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
            actions = network.predict_actions([state], sample=False)[0]
            state, reward, done, _ = env.step(actions)
            rewards += reward
        return rewards

    # Evaluation in ReCodEx
    if args.recodex:
        network.load_actor(args.model_path, env)
        while True:
            evaluate_episode(True)

    # Create the asynchroneous vector environment for training.
    venv = gym.vector.AsyncVectorEnv([lambda: gym.make(args.env)] * args.envs)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque(maxlen=1000000)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    state, training = venv.reset(), not args.recodex
    while training:
        for _ in range(args.evaluate_each):
            # Predict actions by calling `network.predict_actions` with sampling.
            action = network.predict_actions(state, sample=True)

            next_state, reward, done, _ = venv.step(action)
            for i in range(args.envs):
                replay_buffer.append(Transition(state[i], action[i], reward[i], done[i], next_state[i]))
            state = next_state

            # Training
            if len(replay_buffer) >= 4 * args.batch_size:
                # Note that until now we used `np.random.choice` with `replace=False` to generate
                # batch indices. However, this call is extremely slow for large buffers, because
                # it generates a whole permutation. With `np.random.randint`, indices may repeat,
                # but once the buffer is large, it happens with little probability.
                batch = np.random.randint(len(replay_buffer), size=args.batch_size)
                states, actions, rewards, dones, next_states = map(np.array, zip(*[replay_buffer[i] for i in batch]))
                # TODO: Perform the training
                network.train(states, actions, rewards, dones, next_states)

        # Periodic evaluation
        for _ in range(args.evaluate_for):
            evaluate_episode()

        if np.mean(env._episode_returns[-env._evaluate_for:]) > args.target_return:
            network.save_actor(args.model_path)
            break

    if args.recodex:
        network.load_actor(args.model_path, env)

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed)

    main(env, args)
