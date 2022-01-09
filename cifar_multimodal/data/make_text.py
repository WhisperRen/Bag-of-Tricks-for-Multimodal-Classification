import tensorflow as tf
import numpy as np
import pandas as pd


def make_text_modality(labels, mode='train'):
    label_dict = {0: ['airplane', 'plane', 'aeroplane', 'aircraft', 'people'],
                  1: ['automobile', 'car', 'vehicle'],
                  2: ['bird'],
                  3: ['cat', 'kitten'],
                  4: ['deer', 'fawn', 'stag'],
                  5: ['dog', 'doggy', 'puppy', 'people'],
                  6: ['frog', 'toad'],
                  7: ['horse', 'pony', 'mare'],
                  8: ['ship', 'boat', 'vessel'],
                  9: ['truck', 'lorry']}

    pre_lines = ['This is a {} next to the car', 'Look, there is a {} in front of the boat',
                 'There are many {} in the city.',
                 'There are a lot of people in this country who are {} fans', 'Twenty {} are running towards a plane',
                 'There is a man with a dog behinds {}', "{} is Tom's favorite",
                 'There is a woman with a kitten and {}']
    lines = []
    for label in labels:
        names = label_dict.get(label[0])
        ind = np.random.randint(0, len(names))
        ind_pre = np.random.randint(0, len(pre_lines))
        lines.append(pre_lines[ind_pre].format(names[ind]))

    df = pd.DataFrame(lines, columns=['text'])
    if mode == 'train':
        df.to_csv('./cifar_text_train.txt', header=False, index=False)
    elif mode == 'test':
        df.to_csv('./cifar_text_test.txt', header=False, index=False)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    make_text_modality(y_train)
    make_text_modality(y_test, mode='test')
