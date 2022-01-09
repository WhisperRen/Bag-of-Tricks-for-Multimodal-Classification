import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 1000
MAX_LEN = 100


def map_example_to_dict(text_list, image_list, label):
    return (text_list, label), (image_list, label)


def encode_examples(texts, image_list):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(texts['text'])
    text_tokenized = tokenizer.texts_to_sequences(texts['text'])
    text_tokenized = pad_sequences(text_tokenized, maxlen=MAX_LEN)

    # prepare list, so that can build up final tensorflow dataset from slices
    text_list = []
    label_list = []
    for index, row in texts.iterrows():
        label = row["label"]
        text_list.append(text_tokenized[index].reshape(-1, 1))
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices(
        (text_list, image_list, label_list)).map(map_example_to_dict, num_parallel_calls=4), tokenizer
