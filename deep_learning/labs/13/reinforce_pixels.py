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
parser.add_argument("--recodex", default=True, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=2, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--episodes", default=10000, type=int, help="Training episodes.")
parser.add_argument("--hidden_layer_size", default=200, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--activation", default="relu", type=str, help="Activation type")
parser.add_argument("--depth", default=8, type=int, help="Model depth")


# COPY of code from cifar_competition.solution.py lab04 (with few modifications)
# Author: Milan Straka
class Model(tf.keras.Model):
    def _activation(self, inputs, args):
        if args.activation == "relu":
            return tf.keras.layers.Activation(tf.nn.relu)(inputs)
        if args.activation == "lrelu":
            return tf.keras.layers.Activation(tf.nn.leaky_relu)(inputs)
        if args.activation == "elu":
            return tf.keras.layers.Activation(tf.nn.elu)(inputs)
        if args.activation == "swish":
            return tf.keras.layers.Activation(tf.nn.swish)(inputs)
        if args.activation == "gelu":
            return tf.keras.layers.Activation(tf.nn.gelu)(inputs)
        raise ValueError("Unknown activation '{}'".format(args.activation))

class ResNet(Model):
    def _cnn(self, inputs, args, filters, kernel_size, stride, activation):
        hidden = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(inputs)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = self._activation(hidden, args) if activation else hidden
        return hidden

    def _block(self, inputs, args, filters, stride):
        hidden = self._cnn(inputs, args, filters, 3, stride, activation=True)
        hidden = self._cnn(hidden, args, filters, 3, 1, activation=False)
        if stride > 1:
            residual = self._cnn(inputs, args, filters, 1, stride, activation=False)
        else:
            residual = inputs
        hidden = self._activation(hidden + residual, args)
        return hidden

    def __init__(self, args, input_shape):
        n = (args.depth - 2) // 6

        inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)
        hidden = self._cnn(inputs, args, 16, 3, 1, activation=True)
        for stage in range(3):
            for block in range(n):
                hidden = self._block(hidden, args, 16 * (1 << stage), 2 if stage > 0 and block == 0 else 1)
        outputs = tf.keras.layers.GlobalAvgPool2D()(hidden)
        # 128 neurons on output
        #outputs = tf.keras.layers.Dense(n_actions, activation=tf.nn.softmax)(hidden)
        super().__init__(inputs, outputs)
# End of copy

class Agent:
    def _cnn(self, inputs, filters, kernel_size, stride):
        hidden = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding="valid", use_bias=False)(inputs)
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.keras.layers.Activation(tf.nn.relu)(hidden)
        return hidden
    
    def __init__(self, env, args, model_dir, baseline_dir):            
        self._model_dir = model_dir
        self._baseline_dir = baseline_dir

        in_shape = env.observation_space.shape
        inputs = tf.keras.layers.Input(shape=in_shape)
        #resnet = ResNet(args, in_shape)
        #hidden = resnet(inputs)
        #outputs = tf.keras.layers.Dense(env.action_space.n, "softmax")(hidden)
        #self._model = tf.keras.Model(inputs, outputs)

        hidden = self._cnn(inputs, 16, 3, 2)
        hidden = self._cnn(hidden, 16, 3, 2)
        hidden = tf.keras.layers.Flatten()(hidden)
        outputs = tf.keras.layers.Dense(env.action_space.n, "softmax")(hidden)
        self._model = tf.keras.Model(inputs, outputs)

        self._model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.losses.CategoricalCrossentropy(),
            metrics=tf.metrics.CategoricalCrossentropy()
        )
        
        if os.path.exists(model_dir + '.index'):
            self._model.load_weights(model_dir)

        inputs = tf.keras.layers.Input(shape=in_shape)
        hidden = self._cnn(inputs, 16, 3, 2)
        hidden = self._cnn(hidden, 16, 3, 2)
        hidden = tf.keras.layers.Flatten()(hidden)
        hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation="relu")(hidden)
        #resnet = ResNet(args, in_shape)
        #hidden = resnet(inputs)
        outputs = tf.keras.layers.Dense(1, activation=None)(hidden)
        self._baseline_model = tf.keras.Model(inputs, outputs)

        self._baseline_model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=args.learning_rate),
            loss=tf.losses.MeanSquaredError(),
            metrics=tf.metrics.MeanSquaredError()
        )

        if os.path.exists(baseline_dir + '.index'):
            self._baseline_model.load_weights(baseline_dir)

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

    def training(self, env, args):
        for epoch in range(args.episodes // args.batch_size):
            batch_states, batch_actions, batch_returns = [], [], []
            for _ in range(args.batch_size):
                # Perform episode
                states, actions, rewards = [], [], []
                state, done = env.reset(), False
                while not done:
                    if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                        env.render()

                    dist = self.predict([state])
                    action = np.random.choice(env.action_space.n, p=dist[0])

                    next_state, reward, done, _ = env.step(action)

                    states.append(state)
                    oh_action = np.zeros((env.action_space.n))
                    oh_action[action] = 1
                    actions.append(oh_action)
                    rewards.append(reward)

                    state = next_state

                rewards = np.array(rewards)
                return_G = []
                for i, reward in enumerate(rewards):
                    return_G.append(np.sum(rewards[i:])) # TODO add the discount factor if needed

                batch_states += states
                batch_actions += actions
                batch_returns = np.concatenate((batch_returns, return_G))

            batch_returns = np.expand_dims(batch_returns, axis=1)
            self.train(batch_states, batch_actions, batch_returns)
            if (epoch+1) % 5 == 0:
                print("saving weights in after {} epochs".format(epoch+1))
                self._model.save_weights(self._model_dir)
                self._baseline_model.save_weights(self._baseline_dir)
        
        self._model.save_weights(self._model_dir)
        self._baseline_model.save_weights(self._baseline_dir)
        

def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    model_dir = './checkpoints/model_weights'
    baseline_dir = './checkpoints/baseline_weights'
    agent = Agent(env, args, model_dir, baseline_dir)

    if args.recodex:
        # TODO: Perform evaluation of a trained model.
        while True:
            state, done = env.reset(start_evaluation=True), False
            while not done:
                # TODO: Choose an action
                dist = agent.predict([state])
                action = np.argmax(dist[0])
                state, reward, done, _ = env.step(action)

    else:
        # TODO: Perform training
        agent.training(env, args)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("CartPolePixels-v0"), args.seed)

    main(env, args)
