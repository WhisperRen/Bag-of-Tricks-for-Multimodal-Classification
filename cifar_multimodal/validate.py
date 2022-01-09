import argparse
import os

import numpy as np
import tensorflow as tf

from dataloader import DataLoader
from model import BimodalModel


def main():
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=1, help='Size of the batches for each training step.')
    parser.add_argument('--data_training', action='store_true', default=False,
                        help='Specify this if it is for training.')
    parser.add_argument('--cuda_device', type=str, default='-1',
                        help='CUDA device index to be used in training.'
                             'This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. '
                             'Specify it as -1 to disable GPUs.')

    parser.add_argument('--restore_path', type=str, help='Checkpoint path to be restored.',
                        default=r'D:\tmp\train\ckpt_100.h5')
    parser.add_argument('--global_step', type=int, default=0,
                        help='Global step of the restored model. Some models may require to specify this.')
    parser.add_argument('--num_classes', type=int, default=10, help='Class number for classification task')
    parser.add_argument('--pretrained_model_path_text', type=str,
                        default=r'D:\Codes\Pycharm\embracenet\bert_base_uncased', help='Pre-trained model path (text)')

    args, remaining_args = parser.parse_known_args()

    # initialize
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

    # data loader
    tf.print('prepare data loader')
    dataloader = DataLoader()
    dataloader_args, remaining_args = dataloader.parse_args(args, remaining_args)
    dataloader.prepare()

    # model
    tf.print('prepare model')
    model = BimodalModel()
    model_args, remaining_args = model.parse_args(args, remaining_args)
    model.prepare(is_training=False, global_step=args.global_step)

    # check remaining args
    if len(remaining_args) > 0:
        tf.print('WARNING: found unhandled arguments: %s' % remaining_args)

    # model > restore
    model.restore(ckpt_path=args.restore_path)
    tf.print('restored the model')

    # validate
    tf.print('begin validation')
    num_correct_data = 0

    for input_text, input_image in dataloader.train_data_encode:
        model.predict(input_text[0], input_image[0])

    # for data_index in range(num_data):
    #     input_data, truth_label, data_name = dataloader.get_data_pair(data_index=data_index)
    #
    #     model_input_list = np.repeat(np.array([input_data]), repeats=args.ensemble_repeats, axis=0)
    #
    #     output_prob, output_class = model.predict(input_list=model_input_list)
    #     output_prob = np.mean(output_prob, axis=0)
    #     output_class = np.bincount(output_class).argmax()
    #
    #     is_correct = (output_class == truth_label)
    #     num_correct_data += 1 if is_correct else 0
    #
    #     if data_index % 100 == 0:
    #         tf.print('%d/%d, %s (acc: %f)' % (
    #         data_index + 1, num_data, ('O' if is_correct else 'X'), (num_correct_data / (data_index + 1))))
    #
    # # finalize
    # tf.print('finished')
    # tf.print('- accuracy: %f' % (num_correct_data / num_data))
    # tf.print('- error rate: %f' % ((num_data - num_correct_data) / num_data))


if __name__ == '__main__':
    main()
