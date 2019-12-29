# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/6/20 15:48

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizers import RMSprop

from DQN.maze_env import Maze


class Eval_Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_q_network')
        self.layer1 = layers.Dense(10, activation='relu')
        self.logits = layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits


class Target_Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_q_network_1')
        self.layer1 = layers.Dense(10, trainable=False, activation='relu')
        self.logits = layers.Dense(num_actions, trainable=False, activation=None)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits


class DeepQNetwork:
    def __init__(self, n_actions, n_features, eval_model, target_model):

        self.params = {
            'n_actions': n_actions,
            'n_features': n_features,
            'learning_rate': 0.01,
            'reward_decay': 0.9,
            'e_greedy': 0.9,
            'replace_target_iter': 300,
            'memory_size': 500,
            'batch_size': 32,
            'e_greedy_increment': None
        }

        # total learning step

        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.epsilon = 0 if self.params['e_greedy_increment'] is not None else self.params['e_greedy']
        self.memory = np.zeros((self.params['memory_size'], self.params['n_features'] * 2 + 2))

        self.eval_model = eval_model
        self.target_model = target_model

        self.eval_model.compile(
            optimizer=RMSprop(lr=self.params['learning_rate']),
            loss='mse'
        )
        self.cost_his = []

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.params['memory_size']
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.eval_model.predict(observation)
            print(actions_value)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.params['n_actions'])
        return action

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.params['memory_size']:
            sample_index = np.random.choice(self.params['memory_size'], size=self.params['batch_size'])
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.params['batch_size'])

        batch_memory = self.memory[sample_index, :]

        q_next = self.target_model.predict(batch_memory[:, -self.params['n_features']:])
        q_eval = self.eval_model.predict(batch_memory[:, :self.params['n_features']])

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.params['batch_size'], dtype=np.int32)
        eval_act_index = batch_memory[:, self.params['n_features']].astype(int)
        reward = batch_memory[:, self.params['n_features'] + 1]

        q_target[batch_index, eval_act_index] = reward + self.params['reward_decay'] * np.max(q_next, axis=1)

        # check to replace target parameters
        if self.learn_step_counter % self.params['replace_target_iter'] == 0:
            for eval_layer, target_layer in zip(self.eval_model.layers, self.target_model.layers):
                target_layer.set_weights(eval_layer.get_weights())
            print('\ntarget_params_replaced\n')

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]
        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]
        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network

        self.cost = self.eval_model.train_on_batch(batch_memory[:, :self.params['n_features']], q_target)

        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.params['e_greedy_increment'] if self.epsilon < self.params['e_greedy'] \
            else self.params['e_greedy']
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


def run_maze():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()
            # RL choose action based on observation
            action = RL.choose_action(observation)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            if (step > 200) and (step % 5 == 0):
                RL.learn()
            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done:
                break
            step += 1
    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    eval_model = Eval_Model(num_actions=env.n_actions)
    target_model = Target_Model(num_actions=env.n_actions)
    RL = DeepQNetwork(env.n_actions, env.n_features, eval_model, target_model)
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()
