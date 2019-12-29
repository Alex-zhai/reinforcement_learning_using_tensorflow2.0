# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/6/27 15:27

import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from tensorflow.python.keras import layers
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizers import RMSprop


class ddqn_eval_model(Model):
    def __init__(self, num_actions):
        super().__init__('mlp_double_q_network')
        self.layer1 = layers.Dense(20, activation='relu', kernel_initializer=tf.random_normal_initializer(0, 0.3),
                                   bias_initializer=tf.constant_initializer(0.1))
        self.logits = layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits


class ddqn_target_model(Model):
    def __init__(self, num_actions):
        super().__init__('mlp_double_q_network_tar')
        self.layer1 = layers.Dense(20, activation='relu', kernel_initializer=tf.random_normal_initializer(0, 0.3),
                                   bias_initializer=tf.constant_initializer(0.1))
        self.logits = layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits


class DDQN:
    def __init__(self, n_actions, n_features, eval_model, target_model):
        self.params = {
            'n_actions': n_actions,
            'n_features': n_features,
            'learning_rate': 0.005,
            'reward_decay': 0.9,
            'e_greedy': 0.9,
            'replace_target_iter': 200,
            'memory_size': 3000,
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
        index = self.memory_counter % self.params['memory_size']
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.eval_model.predict(observation)
        action = np.argmax(actions_value)

        if not hasattr(self, 'q'):  # record action value it gets
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q * 0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:  # choosing action
            action = np.random.randint(0, self.params['n_actions'])
        return action

    def learn(self):

        if self.memory_counter > self.params['memory_size']:
            sample_index = np.random.choice(self.params['memory_size'], size=self.params['batch_size'])
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.params['batch_size'])
        batch_memory = self.memory[sample_index, :]

        q_next = self.target_model.predict(batch_memory[:, -self.params['n_features']:])
        q_eval4next = self.eval_model.predict(batch_memory[:, -self.params['n_features']:])
        q_eval = self.eval_model.predict(batch_memory[:, :self.params['n_features']])

        q_target = q_eval.copy()

        batch_index = np.arange(self.params['batch_size'], dtype=np.int32)
        eval_act_index = batch_memory[:, self.params['n_features']].astype(int)
        reward = batch_memory[:, self.params['n_features'] + 1]

        max_act4next = np.argmax(q_eval4next, axis=1)  # the action that brings the highest value is evaluated by q_eval
        selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions

        q_target[batch_index, eval_act_index] = reward + self.params['reward_decay'] * selected_q_next

        self.cost = self.eval_model.train_on_batch(batch_memory[:, :self.params['n_features']], q_target)
        print(self.cost)
        self.cost_his.append(self.cost)

        if self.learn_step_counter % self.params['replace_target_iter'] == 0:
            for eval_layer, target_layer in zip(self.eval_model.layers, self.target_model.layers):
                target_layer.set_weights(eval_layer.get_weights())
            print('\ntarget_params_replaced\n')

        self.epsilon = self.epsilon + self.params['e_greedy_increment'] if self.epsilon < self.params['e_greedy'] \
            else self.params['e_greedy']
        self.learn_step_counter += 1


if __name__ == '__main__':

    env = gym.make('Pendulum-v0')
    env = env.unwrapped

    env.seed(1)
    ACTION_SPACE = 11

    eval_model = ddqn_eval_model(ACTION_SPACE)
    target_model = ddqn_target_model(ACTION_SPACE)


    def train(RL):
        total_steps = 0
        observation = env.reset()
        while True:
            # if total_steps - MEMORY_SIZE > 8000: env.render()
            # env.render()
            action = RL.choose_action(observation)

            f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # convert to [-2 ~ 2] float actions
            observation_, reward, done, info = env.step(np.array([f_action]))

            reward /= 10  # normalize to a range of (-1, 0). r = 0 when get upright
            # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
            # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > 3000:  # learning
                RL.learn()

            if total_steps - 3000 > 20000:  # stop game
                break

            observation = observation_
            total_steps += 1
        return RL.q

    RL = DDQN(ACTION_SPACE, 3, eval_model, target_model)
    q_ = train(RL)

    plt.plot(np.array(q_), c='b', label='double')
    plt.legend(loc='best')
    plt.ylabel('Q eval')
    plt.xlabel('training steps')
    plt.grid()
    plt.show()
