import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

# Parameter
MODEL = '../tmp/checkpoints/cp_target_network_episode_1000.ckpt.index'
POST_PROCESS_IMAGE_SIZE = (105, 80, 1)
NUM_FRAMES = 4
ENV = 'SpaceInvaders-v0'


class DQModel(keras.Model):
    def __init__(self, hidden_size: int, num_actions: int, dueling: bool):
        super(DQModel, self).__init__()
        self.dueling = dueling
        self.conv1 = keras.layers.Conv2D(16, (8, 8), (4, 4), activation='relu')
        self.conv2 = keras.layers.Conv2D(32, (4, 4), (2, 2), activation='relu')
        self.flatten = keras.layers.Flatten()
        self.adv_dense = keras.layers.Dense(hidden_size, activation='relu',
                                            kernel_initializer=keras.initializers.he_normal())
        self.adv_out = keras.layers.Dense(num_actions,
                                          kernel_initializer=keras.initializers.he_normal())
        if dueling:
            self.v_dense = keras.layers.Dense(hidden_size, activation='relu',
                                              kernel_initializer=keras.initializers.he_normal())
            self.v_out = keras.layers.Dense(1, kernel_initializer=keras.initializers.he_normal())
            self.lambda_layer = keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))
            self.combine = keras.layers.Add()

    def call(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.flatten(x)
        adv = self.adv_dense(x)
        adv = self.adv_out(adv)
        if self.dueling:
            v = self.v_dense(x)
            v = self.v_out(v)
            norm_adv = self.lambda_layer(adv)
            combined = self.combine([v, norm_adv])
            return combined
        return adv



def image_preprocess(image, new_size=(105, 80)):
    # convert to greyscale, resize and normalize the image
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, new_size)
    image = image / 255
    return image


def choose_action(state, primary_network):
    return np.argmax(primary_network(tf.reshape(state, (1,
                                                        POST_PROCESS_IMAGE_SIZE[0],
                                                        POST_PROCESS_IMAGE_SIZE[1],
                                                        NUM_FRAMES)).numpy()))


def process_state_stack(state_stack, state):
    for i in range(1, state_stack.shape[-1]):
        state_stack[:, :, i - 1].assign(state_stack[:, :, i])
    state_stack[:, :, -1].assign(state[:, :, 0])
    return state_stack


def main():

    # Environment
    env = gym.make(ENV)
    env.seed(0)
    num_actions = env.action_space.n

    # Model
    model = DQModel(256, num_actions, True)
    model.compile(optimizer=keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
    model.load_weights(MODEL)

    # Evaluate
    for i in range(5):

        state = env.reset()
        state = image_preprocess(state)
        state_stack = tf.Variable(np.repeat(state.numpy(), NUM_FRAMES).reshape((POST_PROCESS_IMAGE_SIZE[0],
                                                                                POST_PROCESS_IMAGE_SIZE[1],
                                                                                NUM_FRAMES)))

        while True:

            env.render()
            time.sleep(0.01)

            action = choose_action(state_stack, model)
            next_state, _, done, _ = env.step(action)
            next_state = image_preprocess(next_state)
            state_stack = process_state_stack(state_stack, next_state)

            if done:
                break

    env.close()


if __name__ == '__main__':
    main()
