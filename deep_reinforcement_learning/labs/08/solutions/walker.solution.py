#!/usr/bin/env python3
import argparse
import collections
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--env", default="BipedalWalker-v3", type=str, help="Environment.")
parser.add_argument("--envs", default=8, type=int, help="Environments.")
parser.add_argument("--evaluate_each", default=1000, type=int, help="Evaluate each number of updates.")
parser.add_argument("--evaluate_for", default=10, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--fall_penalty", default=None, type=float, help="Override fall penalty.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="walker.model", type=str, help="Model path")
parser.add_argument("--target_entropy", default=-1, type=float, help="Target entropy per action component.")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")

class Network:
    def __init__(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        self._target_tau = args.target_tau

        # TODO: Create an actor. Because we will be sampling (and `sample()` from
        # `tfp.distributions` does not play nice with functional models) and because
        # we need the `alpha` variable, we use subclassing to create the actor.
        class Actor(tf.keras.Model):
            def __init__(self, hidden_layer_size: int):
                super().__init__()
                # TODO: Create
                # - two hidden layers with `hidden_layer_size` and ReLU activation
                # - a layer for generaing means with `env.action_space.shape[0]` units and no activation
                # - a layer for generaing sds with `env.action_space.shape[0]` units and `tf.math.exp` activation
                # - finally, create a variable represeting a logarithm of alpha, using for example the following:
                self._hidden_1 = tf.keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu)
                self._hidden_2 = tf.keras.layers.Dense(hidden_layer_size, activation=tf.nn.relu)
                self._mus_layer = tf.keras.layers.Dense(env.action_space.shape[0])
                self._sds_layer = tf.keras.layers.Dense(env.action_space.shape[0], activation=tf.math.exp)
                self._log_alpha = tf.Variable(np.log(0.1), dtype=tf.float32)

            def call(self, inputs: tf.Tensor, sample: bool):
                # TODO: Perform the actor computation
                # - First, pass the inputs through the first hidden layer
                #   and then through the second hidden layer.
                # - From these hidden states, compute
                #   - `mus` (the means),
                #   - `sds` (the standard deviations).
                # - Then, create the action distribution using `tfp.distributions.Normal`
                #   with the `mus` and `sds`. Note that to support computation without
                #   sampling, the easiest is to pass zeros as standard deviations when
                #   `sample == False`.
                # - We then bijectively modify the distribution so that the actions are
                #   in the given range. Luckily, `tfp.bijectors` offers classes that
                #   can transform a distribution.
                #   - first run `tfp.bijectors.Tanh()(actions_distribution)` to squash the
                #     actions to [-1, 1] range,
                #   - then run `tfp.bijectors.Scale((env.action_space.high - env.action_space.low) / 2)(actions_distribution)`
                #     to scale the action ranges to [-(high-low)/2, (high-low)/2],
                #   - finally, `tfp.bijectors.Shift((env.action_space.high + env.action_space.low) / 2)(actions_distribution)`
                #     shifts the ranges to [low, high].
                #   If you use PyTorch, I am not aware of any tfp.bijectors-like classes.
                #   In that case, sample from a normal distribution, pass the samples through the
                #   `tanh` and suitable scaling, and then compute the log_prob by using log_prob
                #   from the normal distribution and manually accounting for the `tanh` as shown in the slides.
                #   However, the formula from the slides is not numerically stable, for a better variant see
                #   https://github.com/tensorflow/probability/blob/ef1f64a434/tensorflow_probability/python/bijectors/tanh.py#L70-L81
                # - Sample the actions by a `sample()` call.
                # - Then, compute the log-probabilities of the sampled actions by using `log_prob()`
                #   call. An action is actually a vector, so to be precise, compute for every batch
                #   element a scalar, an average of the log-probabilities of individual action components.
                # - Finally, compute `alpha` as exponentiation of `self._log_alpha`.
                # - Return actions, log_probs and alpha.
                hidden = self._hidden_1(inputs)
                hidden = self._hidden_2(hidden)
                mus = self._mus_layer(hidden)
                sds = self._sds_layer(hidden)
                actions_distribution = tfp.distributions.Normal(mus, sds if sample else 0.)
                actions_distribution = tfp.bijectors.Tanh()(actions_distribution)
                actions_distribution = tfp.bijectors.Scale((env.action_space.high - env.action_space.low) / 2)(actions_distribution)
                actions_distribution = tfp.bijectors.Shift((env.action_space.high + env.action_space.low) / 2)(actions_distribution)
                actions = actions_distribution.sample()
                log_prob = tf.math.reduce_mean(actions_distribution.log_prob(actions), axis=-1)
                return actions, log_prob, tf.math.exp(self._log_alpha)

        # TODO: Instantiate the actor as `self._actor` and compile it.
        self._actor = Actor(args.hidden_layer_size)
        self._actor.compile(tf.keras.optimizers.Adam(args.learning_rate))

        # TODO: Create a critic, which
        # - takes observations and actions as inputs,
        # - concatenates them,
        # - passes the result through two dense layers with `args.hidden_layer_size` units
        #   and ReLU activation,
        # - finally, using a last dense layer produces a single output with no activation
        # This critic needs to be cloned so that two critics and two target critics are created.
        states, actions = tf.keras.layers.Input(shape=env.observation_space.shape), tf.keras.layers.Input(shape=env.action_space.shape[0])
        hidden = tf.keras.layers.Concatenate(axis=1)([states, actions])
        hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu)(hidden)
        hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation=tf.nn.relu)(hidden)
        returns = tf.keras.layers.Dense(1)(hidden)[:, 0]
        critic = tf.keras.Model(inputs=[states, actions], outputs=returns)
        self._critics, self._target_critics = [], []
        for _ in range(2):
            self._critics.append(tf.keras.models.clone_model(critic))
            self._target_critics.append(tf.keras.models.clone_model(critic))
            self._critics[-1].compile(tf.keras.optimizers.Adam(args.learning_rate))

    def save_actor(self, path: str):
        # Because we use subclassing for creating the actor, the easiest way of
        # serializing an actor is just to save weights.
        self._actor.save_weights(path, save_format="h5")

    def load_actor(self, path: str, env: wrappers.EvaluationEnv):
        # When deserializing, we need to make sure the variables are created
        # first -- we do so by processing a batch with a random observation.
        self.predict_actions([env.observation_space.sample()], sample=False)
        self._actor.load_weights(path)

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @tf.function
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        # TODO: Separately train:
        # - the actor, by using two objectives:
        #   - the objective for the actor itself; in this objective, `tf.stop_gradient(alpha)`
        #     should be used (for the `alpha` returned by the actor) to avoid optimizing `alpha`,
        #   - the objective for `alpha`, where `tf.stop_gradient(log_prob)` should be used
        #     to avoid computing gradient for other variables than `alpha`.
        #     Use `args.target_entropy` as the target entropy (the default of -1 per action
        #     component is fine and does not need to be tuned for the agent to train).
        # - the critics using MSE loss.
        #
        # Finally, update the two target critic networks exponential moving
        # average with weight `args.target_tau`, using something like
        #   for var, target_var in zip(critic.trainable_variables, target_critic.trainable_variables):
        #       target_var.assign(target_var * (1 - target_tau) + var * target_tau)
        def actor_loss():
            actions, log_prob, alpha = self._actor(states, sample=True, training=True)

            minimum = None
            for critic in self._critics:
                values = critic([states, actions])
                minimum = tf.math.minimum(minimum, values) if minimum is not None else values

            return tf.math.reduce_mean(tf.stop_gradient(alpha) * log_prob - minimum) + \
                tf.math.reduce_mean(- alpha * tf.stop_gradient(log_prob) - alpha * args.target_entropy)
        self._actor.optimizer.minimize(actor_loss, var_list=self._actor.trainable_variables)

        for critic in self._critics:
            critic.optimizer.minimize(
                lambda: tf.keras.losses.mean_squared_error(returns, critic([states, actions], training=True)),
                var_list=critic.trainable_variables
            )
        for critic, target_critic in zip(self._critics, self._target_critics):
            for var, target_var in zip(critic.trainable_variables, target_critic.trainable_variables):
                target_var.assign(target_var * (1 - self._target_tau) + var * self._target_tau)

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
        minimum = None
        actions, log_prob, alpha = self._actor(states, sample=True)
        for critic in self._target_critics:
            values = critic([states, actions]) - alpha * log_prob
            minimum = tf.math.minimum(minimum, values) if minimum is not None else values
        return minimum


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
            action = network.predict_actions([state], sample=False)[0]
            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    # Evaluation in ReCodEx
    if args.recodex:
        network.load_actor(args.model_path, env)
        while True:
            evaluate_episode(True)

    # Create the asynchroneous vector environment for training.
    venv = gym.vector.AsyncVectorEnv([lambda: gym.make(args.env)] * args.envs)
    venv_eval = gym.vector.AsyncVectorEnv([lambda: gym.make(args.env)] * args.envs)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque(maxlen=1000000)
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    state, training = venv.reset(), True
    updates, best_saved = 0, 0
    while training:
        for _ in range(args.evaluate_each):
            # Predict actions by calling `network.predict_actions` with sampling.
            action = network.predict_actions(state, sample=True)

            next_state, reward, done, _ = venv.step(action)
            for i in range(args.envs):
                if done[i] and args.fall_penalty is not None:
                    reward[i] = max(reward[i], args.fall_penalty)
                replay_buffer.append(Transition(state[i], action[i], reward[i], done[i], next_state[i]))
            state = next_state

            # Training
            if len(replay_buffer) >= 4 * args.batch_size:
                # Note that until now we used `np.random.choice` with `replace=False` to generate
                # batch indices. However, this call is extremely slow for large buffers, because
                # it generates a whole permutation. With `np.random.randint`, indices may repeat,
                # but once the buffer is large, it happend with little probability.
                batch = np.random.randint(len(replay_buffer), size=args.batch_size)
                states, actions, rewards, dones, next_states = map(np.array, zip(*[replay_buffer[i] for i in batch]))
                # TODO: Perform the training
                returns = rewards + args.gamma * (1 - dones) * network.predict_values(next_states)
                network.train(states, actions, returns)
                updates += 1

        # Periodic evaluation
        print("{:.3f}".format(np.exp(network._actor._log_alpha)), end=" ", flush=True)
        state_eval, rewards, returns = venv_eval.reset(), np.zeros(args.envs), []
        while len(returns) < args.evaluate_for or (np.mean(returns) > best_saved - 25 and len(returns) < 500):
            action = network.predict_actions(state_eval, sample=False)
            state_eval, reward, done, _ = venv_eval.step(action)
            rewards += reward
            for i in range(args.envs):
                if done[i]:
                    returns.append(rewards[i])
                    rewards[i] = 0
        print("Evaluation after {}k: {} {}".format(updates // 1000, np.mean(returns), len(returns)),
              *map("{:.2f}".format, returns[:5]),
              *map("{:.2f}".format, returns[-5:]))
        if np.mean(returns) > best_saved:
            best_saved = np.mean(returns)
            path = args.model_path + "{:06.2f}.h5".format(best_saved)
            network.save_actor(path)
            print("Saved agent as {}".format(path))
        if np.mean(returns) >= 250 and args.fall_penalty is not None and args.fall_penalty > -100:
            args.fall_penalty -= 1
            print("Decreasing fall penalty to {}".format(args.fall_penalty))

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make(args.env), args.seed)

    main(env, args)
