# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/6/24 14:53

# reference: https://github.com/keon/policy-gradient/blob/master/pg.py

import gym
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.optimizers import Adam


class PG_model(Model):
    def __init__(self, num_actions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reshape = layers.Reshape((1, 80, 80))
        self.conv = layers.Conv2D(32, 6, 6, activation='relu', kernel_initializer='he_uniform', padding='same')
        self.flatten = layers.Flatten()
        self.dense_layer1 = layers.Dense(64, activation='relu', kernel_initializer='he_uniform')
        self.dense_layer2 = layers.Dense(64, activation='relu', kernel_initializer='he_uniform')
        self.all_act_prob = layers.Dense(num_actions, activation=tf.nn.softmax)

    def call(self, inputs, training=None, mask=None):
        x = tf.convert_to_tensor(inputs)
        x = self.reshape(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        all_act_prob = self.all_act_prob(x)
        return all_act_prob


class PG_Agent:
    def __init__(self, n_actions, n_features, pg_model):
        self.params = {
            'n_actions': n_actions,
            'n_features': n_features,
            'learning_rate': 0.01,
            'reward_decay': 0.95
        }
        self.pg_model = pg_model
        self.pg_model.compile(loss='categorical_crossentropy', optimizer=Adam(self.params['learning_rate']))
        self.probs = []
        self.gradients = []
        self.states = []
        self.rewards = []

    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        aprob = self.pg_model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.params['n_actions'], 1, p=prob)[0]
        return action, prob

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.params['n_actions']])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.params['reward_decay'] + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = rewards / np.std(rewards - np.mean(rewards))
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.params['learning_rate'] * np.squeeze(np.vstack([gradients]))
        print(np.squeeze(np.vstack([gradients])))
        print(self.probs)
        self.pg_model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []


def preprocess(I):
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()


if __name__ == "__main__":
    env = gym.make("Pong-v0")
    state = env.reset()
    prev_x = None
    score = 0
    episode = 0

    state_size = 80 * 80
    action_size = env.action_space.n
    pg_model = PG_model(env.action_space.n)
    agent = PG_Agent(state_size, action_size, pg_model)
    while True:
        env.render()

        cur_x = preprocess(state)
        x = cur_x - prev_x if prev_x is not None else np.zeros(state_size)
        prev_x = cur_x

        action, prob = agent.act(x)
        state, reward, done, info = env.step(action)
        score += reward
        agent.remember(x, action, prob, reward)

        if done:
            episode += 1
            agent.train()
            print('Episode: %d - Score: %f.' % (episode, score))
            score = 0
            state = env.reset()
            prev_x = None
