import argparse
import json
import os
import traceback
import time
from tqdm import tqdm
import logging

import tensorflow as tf

from dataloader import DataLoader
from early_fusion_model import EarlyFusionModel
from late_fusion_model import LateFusionModel

logger = logging.Logger(__name__)


def main():
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16, help='Size of the batches for each training step.')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--cuda_device', type=str, default='0',
                        help='CUDA device index to be used in training.'
                             'This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. '
                             'Specify it as -1 to disable GPUs.')
    parser.add_argument('--fusion_type', type=str, default='early_fusion', help='Either early_fusion or late_fusion')
    parser.add_argument('--specific_frame', type=str, default='simple_concat',
                        help='Specific multi-modal classification architecture.'
                             'for early fusion: simple_concat, embrace_net;'
                             'for late_fusion: fix_weight, trainable_weight')
    parser.add_argument('--num_classes', type=int, default=10, help='Class number for classification task')

    parser.add_argument('--train_path', type=str, default='./restore_path',
                        help='Base path of the trained model to be saved.')
    parser.add_argument('--restore_path', type=str,
                        help='Checkpoint path to be restored. '
                             'Specify this to resume the training or use pre-trained parameters.')
    parser.add_argument('--save_freq', type=int, default=200, help='The frequency of saving the trained model.')
    parser.add_argument('--sleep_ratio', type=float, default=0.05,
                        help='The ratio of sleeping time for each training step, '
                             'which prevents overheating of GPUs. Specify 0 to disable sleeping.')
    parser.add_argument('--data_training', action='store_true', default=True)
    parser.add_argument('--global_step', type=int, default=0,
                        help='Initial global step. Specify this to resume the training.')

    args, remaining_args = parser.parse_known_args()

    # initialize
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    os.makedirs(args.train_path, exist_ok=True)

    # data loader
    logger.warning('prepare data loader')
    dataloader = DataLoader()
    dataloader_args, remaining_args = dataloader.parse_args(args, remaining_args)
    tokenizer = dataloader.prepare()

    # model
    logger.warning('prepare model')
    model = EarlyFusionModel() if args.fusion_type == 'early_fusion' else LateFusionModel()
    model_args, remaining_args = model.parse_args(args, remaining_args)
    model.prepare(tokenizer, is_training=True, global_step=args.global_step)

    # check remaining args
    if len(remaining_args) > 0:
        logger.warning('WARNING: found unhandled arguments: %s' % remaining_args)

    # model > restore
    if args.restore_path is not None:
        model.restore(ckpt_path=args.restore_path)
        logger.warning('restored the model')

    # model > summary
    summary_path = os.path.join(args.train_path, 'summary')
    summary_writer = tf.summary.create_file_writer(summary_path)

    # save arguments
    arguments_path = os.path.join(args.train_path, 'arguments.json')
    all_args = {**vars(args), **vars(dataloader_args), **vars(model_args)}
    with open(arguments_path, 'w') as f:
        f.write(json.dumps(all_args, sort_keys=True, indent=2))

    # train
    logger.warning('begin training')
    local_train_step = 0
    try:
        for _ in tqdm(range(args.epochs)):
            for input_text, input_image in dataloader.train_data_encode:
                global_train_step = model.global_step + 1
                local_train_step += 1

                start_time = time.time()

                summary = summary_writer if local_train_step % 1000 == 0 else None

                loss = model.train_step(input_text=input_text, input_image=input_image, summary=summary)

                duration = time.time() - start_time

                if args.sleep_ratio > 0 and duration > 0:
                    time.sleep(min(10.0, duration * args.sleep_ratio))

                if local_train_step % 10 == 0:
                    logger.warning('step %d, loss %.6f (%.3f sec/batch)' % (global_train_step, loss, duration))

                if local_train_step % args.save_freq == 0:
                    model.save(base_path=args.train_path)
                    logger.warning('saved a model checkpoint at step %d' % global_train_step)

    except KeyboardInterrupt:
        logger.error('interrupted (KeyboardInterrupt)')
        pass
    except Exception as ex:
        logger.error(ex)
        logger.error(traceback.format_exc())

    # finalize
    logger.warning('finished')
    summary_writer.close()


if __name__ == '__main__':
    main()
