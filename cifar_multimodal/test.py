import argparse
import os

import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from dataloader import DataLoader
from model import EarlyFusionModel, LateFusionModel


def main():
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16, help='Size of the batches for each training step.')
    parser.add_argument('--data_training', action='store_true', default=False,
                        help='Specify this if it is for training.')
    parser.add_argument('--cuda_device', type=str, default='-1',
                        help='CUDA device index to be used in training.'
                             'This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. '
                             'Specify it as -1 to disable GPUs.')

    parser.add_argument('--restore_path', type=str, help='Checkpoint path to be restored.',
                        default=r'./restore_path/ckpt_400.h5')
    parser.add_argument('--global_step', type=int, default=0,
                        help='Global step of the restored model. Some models may require to specify this.')
    parser.add_argument('--num_classes', type=int, default=10, help='Class number for classification task')

    args, remaining_args = parser.parse_known_args()

    # initialize
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    # data loader
    tf.print('prepare data loader')
    dataloader = DataLoader()
    dataloader_args, remaining_args = dataloader.parse_args(args, remaining_args)
    tokenizer = dataloader.prepare()

    # model
    tf.print('prepare model')
    model = EarlyFusionModel()
    model_args, remaining_args = model.parse_args(args, remaining_args)
    model.prepare(tokenizer, is_training=False, global_step=args.global_step)

    # check remaining args
    if len(remaining_args) > 0:
        tf.print('WARNING: found unhandled arguments: %s' % remaining_args)

    # model > restore
    model.restore(ckpt_path=args.restore_path)
    tf.print('restored the model')

    # validate
    tf.print('begin testing')
    predictions = []
    labels = []

    # input text: [text, label]; input_image: [image, label]
    for input_text, input_image in dataloader.train_data_encode:
        predict_prob, predict_class = model.predict(input_text[0], input_image[0])
        predictions.extend(predict_class.numpy().tolist())
        labels.extend(tf.squeeze(input_text[1]).numpy().tolist())
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    tf.print('accuracy: {}'.format(accuracy))
    tf.print('f1 score: {}'.format(f1))


if __name__ == '__main__':
    main()
