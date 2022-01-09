import argparse
import copy
import os

import numpy as np
import tensorflow as tf

from embracenet import EmbraceNet
from single_modality_models.base_image_model import make_image_model
from single_modality_models.base_text_model import make_text_model
from utils import MAX_LEN


class EarlyFusionModel:
    def __init__(self):
        self.args = None
        self.model = None
        self.global_step = 0
        self.optimizer = None
        self.loss_fn = None
        self.global_args = None

    def parse_args(self, global_args, args):
        parser = argparse.ArgumentParser()

        parser.add_argument('--model_learning_rate', type=float, default=1e-3, help='Initial learning rate.')
        parser.add_argument('--model_dropout', action='store_true',
                            help='Specify this to employ modality dropout during training.')
        parser.add_argument('--model_drop_left', action='store_true', help='Specity this to drop left-side modality.')
        parser.add_argument('--model_drop_right', action='store_true', help='Specity this to drop right-side modality.')

        self.global_args = global_args

        self.args, remaining_args = parser.parse_known_args(args=args)
        return copy.deepcopy(self.args), remaining_args

    def prepare(self, tokenizer, is_training, global_step=0):
        # config parameters
        self.global_step = global_step

        # main model
        self.model = EmbraceNetBimodalModel(tokenizer, is_training=is_training, global_args=self.global_args,
                                            args=self.args, name='embracenet_bimodal')

        # self.model = SimpleConcatBimodalModel(tokenizer, is_training=is_training, global_args=self.global_args,
        #                                       args=self.args, name='simple_concat_bimodal')

        if is_training:
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.args.model_learning_rate
            )
            self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def save(self, base_path):
        save_path = os.path.join(base_path, 'ckpt_{}.h5'.format(self.global_step))
        self.model.save_weights(save_path)

    def restore(self, ckpt_path):
        text = tf.zeros(shape=[1, MAX_LEN, 1], dtype=tf.int32)
        image = tf.zeros(shape=[1, 32, 32, 3])
        self.model(text, image)
        self.model.load_weights(ckpt_path)
        tf.print(self.model.summary())

    def get_model(self):
        return self.model

    def train_step(self, input_text, input_image, summary=None):
        # do forward propagation, index 0 is data, index 1 is label
        with tf.GradientTape() as tape:
            output_tensor = self.model(input_text[0], input_image[0])
            loss = self.loss_fn(input_text[1], output_tensor)

        # do back propagation
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # finalize
        self.global_step += 1

        # write summary
        if summary is not None:
            with summary.as_default():
                tf.summary.scalar('loss', loss, step=self.global_step)

        return loss

    def predict(self, input_text, input_image):
        # get output
        output_list = self.model(input_text, input_image)
        prob_list = tf.nn.softmax(output_list)
        class_list = tf.math.argmax(prob_list, axis=-1)

        # finalize
        return prob_list, class_list


class EmbraceNetBimodalModel(tf.keras.Model):
    def __init__(self, tokenizer, is_training, global_args, args, **kwargs):
        super(EmbraceNetBimodalModel, self).__init__(**kwargs)

        self.is_training = is_training
        self.args = args
        self.global_args = global_args
        self.tokenizer = tokenizer

        # pre docking layers
        # text model
        self.pre_output_text_size = 16
        self.text_model = make_text_model(self.tokenizer, self.pre_output_text_size, self.is_training)

        # image model
        self.pre_output_image_size = 128
        self.image_model = make_image_model(self.pre_output_image_size)

        # embracenet
        self.embracenet = EmbraceNet(modality_num=2, embracement_size=128, name='embracenet')

        # post embracement layers, reflect to label classes
        self.post = tf.keras.layers.Dense(units=self.global_args.num_classes)

    def call(self, input_text, input_image):
        x_text = self.text_model(input_text)
        x_image = self.image_model(input_image)

        # drop left or right modality
        availabilities = None
        if self.args.model_drop_left or self.args.model_drop_right:
            availabilities_npy = np.ones([input_image.shape[0], 2])
            if self.args.model_drop_left:
                availabilities_npy[:, 0] = 0
            if self.args.model_drop_right:
                availabilities_npy[:, 1] = 0
            availabilities = tf.convert_to_tensor(availabilities_npy)

        # dropout during training
        if self.is_training and self.args.model_dropout:
            dropout_prob = tf.random.uniform([])

            def _dropout_modalities():
                target_modalities = tf.cast(tf.math.round(tf.random.uniform([x.shape[0]]) * 2.0), tf.dtypes.int64)
                return tf.one_hot(target_modalities, depth=2, dtype=tf.dtypes.float32)

            availabilities = tf.cond(tf.less(dropout_prob, 0.5), _dropout_modalities, lambda: tf.ones([x.shape[0], 2]))

        # adjust probability weights of modals
        text_p = 0.5
        image_p = 0.5
        selection_probabilities = tf.repeat([[text_p, image_p]], repeats=input_image.shape[0], axis=0)

        # embrace
        x_embrace = self.embracenet([x_text, x_image], availabilities=availabilities,
                                    selection_probabilities=selection_probabilities)

        # employ final layers
        x = self.post(x_embrace)

        # finalize
        return x


class SimpleConcatBimodalModel(tf.keras.Model):
    def __init__(self, tokenizer, is_training, global_args, args, **kwargs):
        super(SimpleConcatBimodalModel, self).__init__(**kwargs)

        self.is_training = is_training
        self.args = args
        self.global_args = global_args
        self.tokenizer = tokenizer

        # pre docking layers
        # text model
        self.pre_output_text_size = 16
        self.text_model = make_text_model(self.tokenizer, self.pre_output_text_size, self.is_training)

        # image model
        self.pre_output_image_size = 128
        self.image_model = make_image_model(self.pre_output_image_size)

        # reflect to label classes
        self.post = tf.keras.layers.Dense(units=self.global_args.num_classes)

    def call(self, input_text, input_image):
        x_text = self.text_model(input_text)
        x_image = self.image_model(input_image)

        concat_x = tf.concat([x_text, x_image], axis=-1)

        # employ final layers
        x = self.post(concat_x)

        # finalize
        return x


class LateFusionModel:
    def __init__(self):
        self.args = None
        self.model = None
        self.global_step = 0
        self.optimizer = None
        self.loss_fn = None
        self.global_args = None

    def parse_args(self, global_args, args):
        parser = argparse.ArgumentParser()

        parser.add_argument('--model_learning_rate', type=float, default=1e-3, help='Initial learning rate.')
        parser.add_argument('--model_dropout', action='store_true',
                            help='Specify this to employ modality dropout during training.')
        parser.add_argument('--model_drop_left', action='store_true', help='Specity this to drop left-side modality.')
        parser.add_argument('--model_drop_right', action='store_true', help='Specity this to drop right-side modality.')

        self.global_args = global_args

        self.args, remaining_args = parser.parse_known_args(args=args)
        return copy.deepcopy(self.args), remaining_args

    def prepare(self, tokenizer, is_training, global_step=0):
        # config parameters
        self.global_step = global_step

        # main model
        self.model = FixWeightsBimodalModel(tokenizer, is_training=is_training, global_args=self.global_args,
                                            args=self.args, name='fix_weights_bimodal')

        # self.model = TrainableWeightsBimodalModel(tokenizer, is_training=is_training, global_args=self.global_args,
        #                                           args=self.args, name='trainable_weights_bimodal')

        if is_training:
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.args.model_learning_rate
            )
            self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def save(self, base_path):
        save_path = os.path.join(base_path, 'ckpt_{}.h5'.format(self.global_step))
        self.model.save_weights(save_path)

    def restore(self, ckpt_path):
        text = tf.zeros(shape=[1, MAX_LEN, 1], dtype=tf.int32)
        image = tf.zeros(shape=[1, 32, 32, 3])
        self.model(text, image)
        self.model.load_weights(ckpt_path)
        tf.print(self.model.summary())

    def get_model(self):
        return self.model

    def train_step(self, input_text, input_image, summary=None):
        # do forward propagation, index 0 is data, index 1 is label
        with tf.GradientTape() as tape:
            output_tensor = self.model(input_text[0], input_image[0])
            loss = self.loss_fn(input_text[1], output_tensor)

        # do back propagation
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # finalize
        self.global_step += 1

        # write summary
        if summary is not None:
            with summary.as_default():
                tf.summary.scalar('loss', loss, step=self.global_step)

        return loss

    def predict(self, input_text, input_image):
        # get output
        output_list = self.model(input_text, input_image)
        prob_list = tf.nn.softmax(output_list)
        class_list = tf.math.argmax(prob_list, axis=-1)

        # finalize
        return prob_list, class_list


class FixWeightsBimodalModel(tf.keras.Model):
    def __init__(self, tokenizer, is_training, global_args, args, **kwargs):
        super(FixWeightsBimodalModel, self).__init__(**kwargs)

        self.is_training = is_training
        self.args = args
        self.global_args = global_args
        self.tokenizer = tokenizer

        self.text_weight = 0.3
        self.image_weight = 0.7

        # text model
        self.pre_output_text_size = self.global_args.num_classes
        self.text_model = make_text_model(self.tokenizer, self.pre_output_text_size, self.is_training)
        self.text_model.trainable = False

        # image model
        self.pre_output_image_size = self.global_args.num_classes
        self.image_model = make_image_model(self.pre_output_image_size)
        self.image_model.trainable = False

    def call(self, input_text, input_image):
        x_text = tf.math.softmax(self.text_model(input_text))
        x_image = tf.math.softmax(self.image_model(input_image))

        x = self.text_weight * x_text + self.image_weight * x_image
        # finalize
        return x


class TrainableWeightsBimodalModel(tf.keras.Model):
    def __init__(self, tokenizer, is_training, global_args, args, **kwargs):
        super(TrainableWeightsBimodalModel, self).__init__(**kwargs)

        self.is_training = is_training
        self.args = args
        self.global_args = global_args
        self.tokenizer = tokenizer

        # text model
        self.pre_output_text_size = self.global_args.num_classes
        self.text_model = make_text_model(self.tokenizer, self.pre_output_text_size, self.is_training)
        self.text_model.trainable = False

        # image model
        self.pre_output_image_size = self.global_args.num_classes
        self.image_model = make_image_model(self.pre_output_image_size)
        self.image_model.trainable = False

        self.weights_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.global_args.num_classes * 2, activation='relu'),
            tf.keras.layers.Dense(units=self.global_args.num_classes)
        ])

    def call(self, input_text, input_image):
        x_text = tf.math.softmax(self.text_model(input_text))
        x_image = tf.math.softmax(self.image_model(input_image))

        x = tf.concat([x_text, x_image], axis=-1)
        x = self.weights_layer(x)
        # finalize
        return x
