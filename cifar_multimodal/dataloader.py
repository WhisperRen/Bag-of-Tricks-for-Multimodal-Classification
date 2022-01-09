import argparse
import copy
import os

import pandas as pd
import numpy as np
import tensorflow as tf

from utils import encode_examples


class DataLoader:
    def __init__(self):
        self.args = None
        self.global_args = None
        self.train_data_encode = tf.data.Dataset.from_tensor_slices([])
        self.tokenizer = None

    def parse_args(self, global_args, args):
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_path_text', type=str, default=r'./data', help='Base path of the input text data')
        self.global_args = global_args

        self.args, remaining_args = parser.parse_known_args(args=args)
        return copy.deepcopy(self.args), remaining_args

    def prepare(self):
        # load image data
        (x_train_image, y_train_image), (x_test_image, y_test_image) = tf.keras.datasets.cifar10.load_data()

        if self.global_args.data_training:
            text_path = os.path.join(self.args.data_path_text, 'cifar_text_train.txt')
            x_image = x_train_image
            y_image = y_train_image
        else:
            text_path = os.path.join(self.args.data_path_text, 'cifar_text_test.txt')
            x_image = x_test_image
            y_image = y_test_image

        x_image = x_image / 255.0
        x_image = x_image.astype(np.float32)

        # load text data and concat data to make multi-modal dataset
        x_text = pd.read_csv(text_path, names=['text'])
        y_image = pd.DataFrame(y_image[:, -1], columns=['label'])
        x_text = pd.concat([x_text, y_image], axis=1)

        self.train_data_encode, self.tokenizer = encode_examples(x_text, x_image)
        self.train_data_encode = self.train_data_encode.shuffle(5000).batch(
            self.global_args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return self.tokenizer
