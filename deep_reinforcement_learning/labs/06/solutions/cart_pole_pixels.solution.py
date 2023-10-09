#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import cart_pole_pixels_environment
import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
parser.add_argument("--episodes", default=500, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="cart_pole_pixels.model", type=str, help="Model path")

class Network:
    def construct(self, env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
        self._model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(env.observation_space.shape),
            tf.keras.layers.Conv2D(16, 5, 3, "valid", activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(3, 2),
            tf.keras.layers.Conv2D(24, 5, 3, "valid", activation=tf.nn.relu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(env.action_space.n, activation=tf.nn.softmax),
        ])
        self._model.compile(optimizer=tf.optimizers.Adam(args.learning_rate),
                            loss=tf.losses.SparseCategoricalCrossentropy())

        self._baseline = tf.keras.Sequential([
            tf.keras.layers.InputLayer(env.observation_space.shape),
            tf.keras.layers.Conv2D(16, 5, 3, "valid", activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(3, 2),
            tf.keras.layers.Conv2D(24, 5, 3, "valid", activation=tf.nn.relu),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation=None),
        ])
        self._baseline.compile(optimizer=tf.optimizers.Adam(args.learning_rate),
                               loss=tf.losses.MeanSquaredError())

    def load(self, path: str):
        self._model = tf.keras.models.load_model(path)

    def save(self, path: str):
        self._model.save(path, include_optimizer=False, save_format="h5")

    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @tf.function(experimental_relax_shapes=True)
    def train(self, states: np.ndarray, actions: np.ndarray, returns: np.ndarray) -> None:
        baseline = self._baseline(states)[:, 0]
        self._model.optimizer.minimize(
            lambda: self._model.compiled_loss(actions, self._model(states, training=True), returns - baseline),
            var_list=self._model.trainable_variables
        )
        self._baseline.optimizer.minimize(
            lambda: self._baseline.compiled_loss(returns, self._baseline(states, training=True)[:, 0]),
            var_list=self._baseline.trainable_variables
        )

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states: np.ndarray) -> np.ndarray:
        return self._model(states)

def main(env: wrappers.EvaluationEnv, args: argparse.Namespace) -> None:
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    network = Network()
    if args.recodex:
        # TODO: Perform evaluation of a trained model.
        network.load(args.model_path)
        while True:
            state, done = env.reset(start_evaluation=True), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()
                # TODO: Choose an action
                action = np.argmax(network.predict([state])[0])
                state, reward, done, _ = env.step(action)

    else:
        # TODO: Perform training
        network.construct(env, args)

        num_envs, num_actions = args.threads, env.action_space.n
        aenv = gym.vector.AsyncVectorEnv([lambda: gym.make("CartPolePixels-v0")] * num_envs)

        batches, good_enough = 0, False
        state = aenv.reset()
        states = [[] for _ in range(num_envs)]
        actions = [[] for _ in range(num_envs)]
        rewards = [[] for _ in range(num_envs)]

        while True:
            returns = []

            batch_states, batch_actions, batch_returns = [], [], []
            while len(returns) < args.batch_size:
                action = np.array([np.random.choice(len(p), p=p) for p in network.predict(state)])

                next_state, reward, done, _ = aenv.step(action)

                for i in range(num_envs):
                    states[i].append(state[i])
                    actions[i].append(action[i])
                    rewards[i].append(reward[i])

                    if done[i]:
                        for j in reversed(range(len(rewards[i]) - 1)):
                            rewards[i][j] += args.gamma * rewards[i][j + 1]

                        batch_states += states[i]
                        batch_actions += actions[i]
                        batch_returns += rewards[i]
                        returns.append(rewards[i][0])
                        states[i], actions[i], rewards[i] = [], [], []

                state = next_state
            print("{:3.0f}".format(np.mean(returns)), end=" ", flush=True)
            network.train(batch_states, batch_actions, batch_returns)
            batches += 1

            if batches % (10 if not good_enough else 1) == 0:
                network.save(args.model_path + "{:06d}".format(batches))

                # Evalution
                state = aenv.reset()
                rewards, returns = np.zeros(num_envs), []
                while len(returns) < 10 or (len(returns) < 1000 and np.mean(returns) == 500):
                    action = np.argmax(network.predict(state), axis=-1)
                    state, reward, done, _ = aenv.step(action)
                    rewards += reward
                    for i in range(num_envs):
                        if done[i]:
                            returns.append(rewards[i])
                            rewards[i] = 0
                print("Batch {}, evaluation mean {}, episodes {}".format(batches, np.mean(returns), len(returns)))
                good_enough = np.mean(returns) >= 490

                state = aenv.reset()
                states = [[] for _ in range(num_envs)]
                actions = [[] for _ in range(num_envs)]
                rewards = [[] for _ in range(num_envs)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationEnv(gym.make("CartPolePixels-v0"), args.seed)

    main(env, args)
