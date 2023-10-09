#!/usr/bin/env python3

# IDs:
#edbe2dad-018e-11eb-9574-ea7484399335
#aa20f311-c7a2-11e8-a4be-00505601122b
#c716f6b0-25ab-11ec-986f-f39926f24a9c

from __future__ import annotations
import sys
import argparse
import collections
import math
import os
from re import L

from numpy.core.numeric import NaN
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2") # Report only TF errors by default

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras

from az_quiz import AZQuiz
import az_quiz_evaluator
import az_quiz_player_simple_heuristic
import wrappers

import timeit

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.3, type=float, help="MCTS root Dirichlet alpha")
parser.add_argument("--batch_size", default=512, type=int, help="Number of game positions to train on.")
parser.add_argument("--epsilon", default=0.25, type=float, help="MCTS exploration epsilon in root")
parser.add_argument("--evaluate_each", default=5, type=int, help="Evaluate each number of iterations.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--model_path", default="az_quiz.model", type=str, help="Model path")
parser.add_argument("--old_model_path", default="az_quiz150.model", type=str, help="Model path")
parser.add_argument("--num_simulations", default=200, type=int, help="Number of simulations in one MCTS.")
parser.add_argument("--sampling_moves", default=8, type=int, help="Sampling moves.")
parser.add_argument("--show_sim_games", default=True, action="store_true", help="Show simulated games.")
parser.add_argument("--sim_games", default=1, type=int, help="Simulated games to generate in every iteration.")
parser.add_argument("--train_for", default=1, type=int, help="Update steps in every iteration.")
parser.add_argument("--window_length", default=100000, type=int, help="Replay buffer max length.")
parser.add_argument("--save_each", default=50, type=int, help="Save model each number of iterations.")

# Model parameters
parser.add_argument("--conv_filters", default=20, type=int, help="Number of filters in convolutional layers.")
parser.add_argument("--kernel_size", default=3, type=int, help="Size of kernel in convolutional layers.")
parser.add_argument("--conv_layers", default=5, type=int, help="Number of convolutional layers.")

parser.add_argument("--l2_regularization", default=0.01, type=float, help="L2 regularization constant.")
parser.add_argument("--clipnorm", default=10, type=float, help="Clip gradient of Adam.")


ACTIONS = 28
N = 7
C = 4
CROSS_ENTROPY = tf.keras.losses.CategoricalCrossentropy()
MSE = tf.keras.losses.MeanSquaredError()

def custom_loss(y_true, y_pred):
    # Only counts the loss on the non-zero action in y_pred (we don't want to train the model to
    # predict zeros for the rest of actions)
    policy_true = y_true[0]
    policy_pred = y_pred[0]
    value_true = y_true[1]
    value_pred = y_pred[1]
    policy_loss = CROSS_ENTROPY(policy_true, policy_pred)
    value_loss = MSE(value_true, value_pred)
    return policy_loss + value_loss

#########
# Agent #
#########
class Agent:
    def __init__(self, args: argparse.Namespace):
        # TODO: Define an agent network in `self._model`.
        #
        # A possible architecture known to work consits of
        # - 5 convolutional layers with 3x3 kernel and 15-20 filters,
        # - a policy head, which first uses 3x3 convolution to reduce the number of channels
        #   to 2, flattens the representation, and finally uses a dense layer with softmax
        #   activation to produce the policy,
        # - a value head, which again uses 3x3 convolution to reduce the number of channels
        #   to 2, flattens, and produces expected return using an output dense layer with
        #   `tanh` activation.
        input = tf.keras.layers.Input(shape=(N, N, C))
        hidden_layer = tf.keras.layers.Conv2D(filters=20, kernel_size=3, padding='same', activation='relu')(input)
        hidden_layer = tf.keras.layers.Conv2D(filters=20, kernel_size=3, padding='same', activation='relu')(hidden_layer)
        hidden_layer = tf.keras.layers.Conv2D(filters=20, kernel_size=3, padding='same', activation='relu')(hidden_layer)
        hidden_layer = tf.keras.layers.Conv2D(filters=20, kernel_size=3, padding='same', activation='relu')(hidden_layer)
        hidden_layer = tf.keras.layers.Conv2D(filters=20, kernel_size=3, padding='same', activation='relu')(hidden_layer)

        policy = tf.keras.layers.Conv2D(filters=2, kernel_size=3, padding='same', activation='relu')(hidden_layer)
        policy = tf.keras.layers.Flatten()(policy)
        policy = tf.keras.layers.Dense(ACTIONS, activation='softmax')(policy)

        value = tf.keras.layers.Conv2D(filters=2, kernel_size=3, padding='same', activation='relu')(hidden_layer)
        value = tf.keras.layers.Flatten()(value)
        value = tf.keras.layers.Dense(1, activation='tanh')(value)

        self._model = tf.keras.models.Model(inputs=input, outputs=[policy, value])
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        self._model.compile(loss=custom_loss, optimizer=optimizer)

    @classmethod
    def load(cls, path: str, args: argparse.Namespace) -> Agent:
        # A static method returning a new Agent loaded from the given path.
        agent = Agent.__new__(Agent)
        agent._model = tf.keras.models.load_model(path, custom_objects={'custom_loss': custom_loss})
        return agent

    def save(self, path: str, include_optimizer=True) -> None:
        # Save the agent model as a h5 file, possibly with/without the optimizer.
        self._model.save(path, include_optimizer=include_optimizer, save_format="h5")

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @tf.function
    def train(self, boards: np.ndarray, target_policies: np.ndarray, target_values: np.ndarray) -> None:
        # TODO: Train the model based on given boards, target policies and target values.
        self._model.optimizer.minimize(
            lambda: self._model.loss([target_policies, target_values], self._model(boards, training=True)),
            var_list=self._model.trainable_variables
        )

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict(self, boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Return the predicted policy and the value function.
        return self._model(boards)

    def board(self, game: AZQuiz) -> np.ndarray:
        # TODO: Generate the boards from the current AZQuiz game.
        #
        # The `game.board` returns a board representation, but you also need to
        # somehow indicate who is the current player. You can either
        # - change the game so that the current player is always the same one
        #   (i.e., always 0 or always 1; `AZQuiz.swap_players` might come handy);
        # - indicate the current player by adding channels to the representation.
        if game.to_play != 0:
            game.swap_players()
            board = game.board
            game.swap_players()
            return board
        else:
            return game.board

########
# MCTS #
########

class MCTNode:
    def __init__(self, prior: float, parent: MCTNode):
        self.prior = prior # Prior probability from the agent.
        self.game = None   # If the node is evaluated, the corresponding game instance.
        self.children = {} # If the node is evaluated, mapping of valid actions to the child `MCTNode`s.
        self.visit_count = 0
        self.total_value = 0
        self.parent = parent

    def value(self) -> float:
        # TODO: Return the value of the current node, handling the
        # case when `self.visit_count` is 0.
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count


    def is_evaluated(self) -> bool:
        # A node is evaluated if it has non-zero `self.visit_count`.
        # In such case `self.game` is not None.
        return self.visit_count > 0

    def evaluate(self, game: AZQuiz, agent: Agent) -> None:
        # Each node can be evaluated at most once
        assert self.game is None
        self.game = game

        # TODO: Compute the value of the current game.
        # - If the game has ended, compute the value directly
        # - Otherwise, use the given `agent` to evaluate the current
        #   game. Then, for all valid actions, populate `self.children` with
        #   new `MCTNodes` with the priors from the policy predicted
        #   by the network.
        if game.winner is not None:
            value = -1
        else:
            policy, value = agent.predict([agent.board(game)])
            policy, value = policy[0], value[0, 0]
            valid_actions = game.valid_actions()
            for a in valid_actions:
                self.children[a] = MCTNode(prior=policy[a], parent=self)

        #print(value)
        self.visit_count, self.total_value = 1, value

    def add_exploration_noise(self, epsilon: float, alpha: float) -> None:
        # TODO: Update the children priors by exploration noise
        # Dirichlet(alpha), so that the resulting priors are
        #   epsilon * Dirichlet(alpha) + (1 - epsilon) * original_prior
        dir = np.random.dirichlet(np.ones(len(self.children))*alpha)
        for i, child in enumerate(self.children.values()):
            child.prior = epsilon * dir[i] + (1 - epsilon) * child.prior

    def select_child(self) -> tuple[int, MCTNode]:
        # Select a child according to the PUCT formula.
        def ucb_score(child):
            # TODO: For a given child, compute the UCB score as
            #   Q(s, a) + C(s) * P(s, a) * (sqrt(N(s)) / (N(s, a) + 1)),
            # where:
            # - Q(s, a) is the estimated value of the action stored in the
            #   `child` node. However, the value in the `child` node is estimated
            #   from the view of the player playing in the `child` node, which
            #   is usually the other player than the one playing in `self`,
            #   and in that case the estimated value must be "inverted";
            # - C(s) in AlphaZero is defined as
            #     log((1 + N(s) + 19652) / 19652) + 1.25
            #   Personally I used 1965.2 to account for shorter games, but I do not
            #   think it makes any difference;
            # - P(s, a) is the prior computed by the agent;
            # - N(s) is the number of visits of state `s`;
            # - N(s, a) is the number of visits of action `a` in state `s`.
            q = -child.value()
            #print(q)
            c = math.log((1 + self.visit_count + 1965.2) / 1965.2) + 1.25
            p = child.prior
            return q + c * p * (math.sqrt(self.visit_count)/(child.visit_count + 1))

        # TODO: Return the (action, child) pair with the highest `ucb_score`.
        max_ucb = float("-inf")
        #argmax_ucb = np.random.choice(list(self.children.keys()))
        argmax_ucb = next(iter(self.children))
        for action, child in self.children.items():
            ucb = ucb_score(child)
            if ucb > max_ucb:
                max_ucb, argmax_ucb = ucb, action
        return (argmax_ucb, self.children[argmax_ucb])

def mcts(game: AZQuiz, agent: Agent, args: argparse.Namespace, explore: bool) -> np.ndarray:
    # Run the MCTS search and return the policy proportional to the visit counts,
    # optionally including exploration noise to the root children.
    root = MCTNode(None, None)
    root.evaluate(game, agent)
    if explore:
        root.add_exploration_noise(args.epsilon, args.alpha)

    # Perform the `args.num_simulations` number of MCTS simulations.
    for _ in range(args.num_simulations):
        # TODO: Starting in the root node, traverse the tree using `select_child()`,
        # until a `node` without `children` is found.
        def traverse(node: MCTNode) -> MCTNode:
            while len(node.children) != 0:
                action, node = node.select_child()
            return action, node
        action, node = traverse(root)

        # If the node has not been evaluated, evaluate it.
        # Note that it is possible for a node to have no children and
        # be evaluated -- if the game ends in this node.
        if not node.is_evaluated():
            # TODO: Get the AZQuiz instance for this node by cloning
            # the `game` from its parent and performing a suitable action.
            game = node.parent.game.clone() 
            game.move(action)
            node.evaluate(game, agent)
        # Get the value of the node.
        value = node.value()

        # TODO: For all parents of the `node`, update their value estimate,
        # i.e., the `visit_count` and `total_value`.
        value_to_play = node.game.to_play
        node = node.parent
        while node is not None:
            node.visit_count += 1
            node.total_value += value if node.game.to_play == value_to_play else -value
            node = node.parent

    # TODO: Compute a policy proportional to visit counts of the root children.
    # Note that invalid actions are not the children of the root, but the
    # policy should still return 0 for them.
    policy = np.zeros(shape=(AZQuiz.actions))
    for action, child in root.children.items():
        policy[action] = child.visit_count / root.visit_count
    return policy / np.sum(policy)

############
# Training #
############
ReplayBufferEntry = collections.namedtuple("ReplayBufferEntry", ["board", "policy", "outcome"])

def sim_game(agent: Agent, args: argparse.Namespace) -> list[ReplayBufferEntry]:
    # Simulate a game, return a list of `ReplayBufferEntry`s.
    game = AZQuiz(randomized=False)
    i = 0
    boards = []
    policies = []
    players = []
    while game.winner is None:
        # TODO: Run the `mcts` with exploration.
        policy = mcts(game, agent, args, explore=True)

        # TODO: Select an action, either by sampling from the policy or greedily,
        # according to the `args.sampling_moves`.
        if i < args.sampling_moves:
            action = np.random.choice(len(policy), size=1, p=policy)[0]
            i += 1
        else:
            action = np.argmax(policy)

        boards.append(agent.board(game))
        policies.append(policy)
        players.append(1) if game.to_play == 0 else players.append(-1)
        game.move(action)

    # TODO: Return all encountered game states, each consisting of
    # - the board (probably via `agent.board`),
    # - the policy obtained by MCTS,
    # - the outcome based on the outcome of the whole game.
    outcome = 1.0 if game.winner == 0 else -1.0
    replay_buffer = []
    for board, policy, player_multiple in zip(boards, policies, players):
        cur_outcome = outcome * player_multiple
        replay_buffer.append(ReplayBufferEntry(board, policy, cur_outcome))
    return replay_buffer

def train(args: argparse.Namespace) -> Agent:
    # Perform training
    #agent = Agent(args)
    agent = Agent.load(args.model_path, args)
    replay_buffer = collections.deque(maxlen=args.window_length)

    iteration = 0
    training = True
    while training:
        iteration += 1
        
        # STATISTICS
        # Simulation zabere 40x vice casu nez Training
        # Evaluation zabere 1.5x vice casu nez Simulation

        # Generate simulated games
        for _ in range(args.sim_games):
            game = sim_game(agent, args)
            replay_buffer.extend(game)

            # If required, show the generated game, as 8 very long lines showing
            # all encountered boards, each field showing as
            # - `XX` for the fields belonging to player 0,
            # - `..` for the fields belonging to player 1,
            # - percentage of visit counts for valid actions.
            if args.show_sim_games and iteration % args.evaluate_each == 0:
                log = [[] for _ in range(8)]
                for i, (board, policy, outcome) in enumerate(game):
                    log[0].append("Move {}, result {}".format(i, outcome).center(28))
                    action = 0
                    for row in range(7):
                        log[1 + row].append("  " * (6 - row))
                        for col in range(row + 1):
                            log[1 + row].append(
                                " XX " if board[row, col, 0] else
                                " .. " if board[row, col, 1] else
                                "{:>3.0f} ".format(policy[action] * 100))
                            action += 1
                        log[1 + row].append("  " * (6 - row))
                original_stdout = sys.stdout
                with open(f'log{int(iteration / 100)}.txt', 'a') as f:
                    sys.stdout = f
                    print(*["".join(line) for line in log], sep="\n")
                    sys.stdout = original_stdout

        # Train
        for _ in range(args.train_for):
            # TODO: Perform training by sampling an `args.batch_size` of positions
            # from the `replay_buffer` and running `agent.train` on them.
            batch = np.random.randint(len(replay_buffer), size=args.batch_size)
            boards, target_policies, target_values = map(np.array, zip(*[replay_buffer[i] for i in batch]))
            agent.train(boards, target_policies, target_values)

        # Evaluate
        if iteration % args.evaluate_each == 0:
            # Run an evaluation on 2*56 games versus the simple heuristics,
            # using the `Player` instance defined below.
            # For speed, the implementation does not use MCTS during evaluation,
            # but you can of course change it so that it does.
            score = az_quiz_evaluator.evaluate(
                [Player(agent, argparse.Namespace(num_simulations=0)), az_quiz_player_simple_heuristic.Player()],
                games=56, randomized=False, first_chosen=False, render=False, verbose=False)
            print("Evaluation after iteration {}: {:.1f}%".format(iteration, 100 * score), flush=True)
            original_stdout = sys.stdout
            with open(f'log{int(iteration / 100)}.txt', 'a') as f:
                sys.stdout = f
                print("Evaluation after iteration {}: {:.1f}%".format(iteration, 100 * score), flush=True)
                sys.stdout = original_stdout
        
        if iteration % args.save_each == 0:
            Agent.save(agent, args.model_path, include_optimizer=True)
        
    return agent

#####################
# Evaluation Player #
#####################
class Player:
    def __init__(self, agent: Agent, args: argparse.Namespace):
        self.agent = agent
        self.args = args

    def play(self, game: AZQuiz) -> int:
        # Predict a best possible action.
        if self.args.num_simulations == 0:
            # TODO: If no simulations should be performed, use directly
            # the policy predicted by the agent on the current game board.
            policy = self.agent.predict([self.agent.board(game)])[0][0]
        else:
            # TODO: Otherwise run the `mcts` without exploration and
            # utilize the policy returned by it.
            policy = mcts(game, self.agent, self.args, explore=False)

        # Now select a valid action with the largest probability.
        return max(game.valid_actions(), key=lambda action: policy[action])

########
# Main #
########
def main(args: argparse.Namespace) -> Player:
    if args.recodex:
        # Load the trained agent
        agent = Agent.load(args.model_path, args)
    else:
        # Perform training
        agent = train(args)

    return Player(agent, args)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    player = main(args)

    # Run an evaluation versus the simple heuristic with the same parameters as in ReCodEx.
    az_quiz_evaluator.evaluate(
        [player, az_quiz_player_simple_heuristic.Player()],
        games=56, randomized=False, first_chosen=False, render=False, verbose=True,
    )
    
    #agent_old = Agent.load(args.old_model_path, args)
    #player_old = Player(agent_old, args)
    #az_quiz_evaluator.evaluate(
    #    [player, player_old],
    #    games=56, randomized=False, first_chosen=False, render=False, verbose=True,
    #)
