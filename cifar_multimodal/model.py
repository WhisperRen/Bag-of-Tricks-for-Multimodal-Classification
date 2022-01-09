import argparse
import copy
import os

import numpy as np
import tensorflow as tf
from transformers import TFBertForSequenceClassification

from embracenet import EmbraceNet


class BimodalModel:
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

    def prepare(self, is_training, global_step=0):
        # config parameters
        self.global_step = global_step

        # main model
        self.model = EmbraceNetBimodalModel(is_training=is_training, global_args=self.global_args,
                                            args=self.args, name='embracenet_bimodal')
        if is_training:
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.args.model_learning_rate
            )
            self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def save(self, base_path):
        save_path = os.path.join(base_path, 'ckpt_{}.h5'.format(self.global_step))
        self.model.save_weights(save_path)

    def restore(self, ckpt_path):
        text = {'input_ids': tf.zeros(shape=[1, 32], dtype=tf.int32),
                'token_type_ids': tf.zeros(shape=[1, 32], dtype=tf.int32),
                'attention_mask': tf.zeros(shape=[1, 32], dtype=tf.int32)}
        image = tf.zeros(shape=[1, 32, 32, 3])
        self.model(text, image)
        self.model.load_weights(ckpt_path)
        tf.print(self.model.summary())

    def get_model(self):
        return self.model

    # @tf.function
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

    # todo: inference is not available now !!
    def predict(self, input_text, input_image):
        # get output
        output_list = self.model(input_text, input_image)
        prob_list = tf.nn.softmax(output_list)
        class_list = tf.math.argmax(prob_list, axis=-1)

        # finalize
        return prob_list, class_list


class EmbraceNetBimodalModel(tf.keras.Model):
    def __init__(self, is_training, global_args, args, **kwargs):
        super(EmbraceNetBimodalModel, self).__init__(**kwargs)

        # input parameters
        self.is_training = is_training
        self.args = args
        self.global_args = global_args

        # pre embracement layers
        # text model
        # a trick, assign pre_out_size to bert model num_labels for modal fusion
        self.pre_output_text_size = 1024
        self.text_model = TFBertForSequenceClassification.from_pretrained(
            self.global_args.pretrained_model_path_text, num_labels=self.pre_output_text_size)

        # image model
        self.pre_output_image_size = 512
        self.image_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=self.pre_output_image_size)
        ])

        # embracenet
        self.embracenet = EmbraceNet(modality_num=2, embracement_size=512, name='embracenet')

        # post embracement layers
        self.post = tf.keras.layers.Dense(units=self.global_args.num_classes)

    def call(self, input_text, input_image):
        # text model output has been reflect to a vector, somehow it is a tuple, so get index 0
        x_text = self.text_model(input_text, training=self.is_training)[0]
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
