#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
parser.add_argument("--episodes", default=200, type=int, help="Training episodes.")
parser.add_argument("--hidden_layer_size", default=200, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.02, type=float, help="Learning rate.")

class Agent:
    def __init__(self, env, args):
        # TODO: Create a suitable model. The predict method assumes
        # the policy network is stored as `self._model`.
        #
        # Apart from the model defined in `reinforce`, define also another
        # model for computing the baseline (with one output, using a dense layer
        # without activation). (Alternatively, this baseline computation can
        # be grouped together with the policy computation in a single tf.keras.Model.)
        #
        # Using Adam optimizer with given `args.learning_rate` for both models
        # is a good default.
        in_shape = env.observation_space.shape
        inputs = tf.keras.layers.Input(shape=in_shape)
        hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation="relu")(inputs)
        outputs = tf.keras.layers.Dense(env.action_space.n, "softmax")(hidden)
        self._model = tf.keras.Model(inputs, outputs)

        self._model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.losses.CategoricalCrossentropy(),
            metrics=tf.metrics.CategoricalCrossentropy()
        )
        
        inputs = tf.keras.layers.Input(shape=in_shape)
        hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation="relu")(inputs)
        outputs = tf.keras.layers.Dense(1, activation=None)(hidden)
        self._baseline_model = tf.keras.Model(inputs, outputs)

        self._baseline_model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.losses.MeanSquaredError(),
            metrics=tf.metrics.MeanSquaredError()
        )

    # Define a training method.
    #
    # Note that we need to use @tf.function for efficiency (using `train_on_batch`
    # on extremely small batches/networks has considerable overhead).
    #
    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @wrappers.typed_np_function(np.float32, np.int32, np.float32)
    @tf.function(experimental_relax_shapes=True)
    def train(self, states, actions, returns):
        # TODO: Perform training, using the loss from the REINFORCE with
        # baseline algorithm.
        # You should:
        # - compute the predicted baseline using the baseline model
        # - train the baseline model to predict `returns`
        # - train the policy model, using `returns - predicted_baseline` as
        #   the advantage estimate
        with tf.GradientTape() as tape:
            pred_baselines = self._baseline_model(states)
            loss = self._baseline_model.compiled_loss(y_true=returns, y_pred=pred_baselines)
        self._baseline_model.optimizer.minimize(loss, self._baseline_model.trainable_variables, tape=tape)
        self._baseline_model.compiled_metrics.update_state(y_true=returns, y_pred=pred_baselines)
        
        with tf.GradientTape() as tape:
            actions_dist = self._model(states)
            loss = self._model.compiled_loss(y_true=actions, y_pred=actions_dist, sample_weight=returns-pred_baselines)
        self._model.optimizer.minimize(loss, self._model.trainable_variables, tape=tape)
        self._model.compiled_metrics.update_state(y_true=actions, y_pred=actions_dist, sample_weight=returns-pred_baselines)
        


    # Predict method, again with manual @tf.function for efficiency.
    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, states):
        return self._model(states)

def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the agent
    agent = Agent(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                # TODO(reinforce): Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `agent.predict` and current `state`.
                dist = agent.predict([state])
                action = np.random.choice(env.action_space.n, p=dist[0])

                next_state, reward, done, _ = env.step(action)

                states.append(state)
                oh_action = np.zeros((env.action_space.n))
                oh_action[action] = 1
                actions.append(oh_action)
                rewards.append(reward)

                state = next_state

            # TODO(reinforce): Compute returns from the received rewards
            rewards = np.array(rewards)
            return_G = []
            for i, reward in enumerate(rewards):
                return_G.append(np.sum(rewards[i:])) # TODO add the discount factor if needed

            # TODO(reinforce): Add states, actions and returns to the training batch
            batch_states += states
            batch_actions += actions
            batch_returns = np.concatenate((batch_returns, return_G))

        # TODO(reinforce): Train using the generated batch.
        batch_returns = np.expand_dims(batch_returns, axis=1)
        agent.train(batch_states, batch_actions, batch_returns)

    # Final evaluation
    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            # TODO(reinforce): Choose greedy action
            dist = agent.predict([state])
            action = np.argmax(dist[0])
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("CartPole-v1"), args.seed)

    main(env, args)
