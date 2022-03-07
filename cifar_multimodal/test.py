import argparse
import os
import logging

import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from dataloader import DataLoader
from early_fusion_model import EarlyFusionModel
from late_fusion_model import LateFusionModel

logger = logging.Logger(__name__)


def main():
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16, help='Size of the batches for each training step.')
    parser.add_argument('--cuda_device', type=str, default='-1',
                        help='CUDA device index to be used in training.'
                             'This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. '
                             'Specify it as -1 to disable GPUs.')
    parser.add_argument('--fusion_type', type=str, default='early_fusion', help='Either early_fusion or late_fusion')
    parser.add_argument('--specific_frame', type=str, default='simple_concat',
                        help='Specific multi-modal classification architecture.'
                             'for early fusion: simple_concat, embrace_net;'
                             'for late_fusion: fix_weight, trainable_weight')
    parser.add_argument('--num_classes', type=int, default=10, help='Class number for classification task')

    parser.add_argument('--restore_path', type=str, help='Checkpoint path to be restored.',
                        default=r'./restore_path/ckpt_400.h5')
    parser.add_argument('--data_training', action='store_true', default=False)

    args, remaining_args = parser.parse_known_args()

    # initialize
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    # data loader
    logger.warning('prepare data loader')
    dataloader = DataLoader()
    dataloader_args, remaining_args = dataloader.parse_args(args, remaining_args)
    tokenizer = dataloader.prepare()

    # model
    logger.warning('prepare model')
    model = EarlyFusionModel() if args.fusion_type == 'early_fusion' else LateFusionModel()
    model_args, remaining_args = model.parse_args(args, remaining_args)
    model.prepare(tokenizer, is_training=False)

    # check remaining args
    if len(remaining_args) > 0:
        logger.warning('WARNING: found unhandled arguments: %s' % remaining_args)

    # model > restore
    model.restore(ckpt_path=args.restore_path)
    logger.warning('restored the model')

    # validate
    logger.warning('begin testing')
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
