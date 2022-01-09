import tensorflow as tf


def make_text_model(tokenizer, output_dim=16, trainable=True):
    text_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(tokenizer.index_word) + 1, 8),
        tf.keras.layers.Dropout(rate=0.5, trainable=trainable),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)),
        tf.keras.layers.Dense(units=output_dim, activation='relu')
    ])
    return text_model
