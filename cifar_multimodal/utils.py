import tensorflow as tf

MAX_LENGTH = 32


def convert_example_to_feature(review, tokenizer):
    # combine step for tokenization, WordPiece vector mapping,
    # adding special tokens as well as truncating reviews longer than the max length
    return tokenizer.encode_plus(review,
                                 add_special_tokens=True,  # add [CLS], [SEP]
                                 max_length=MAX_LENGTH,  # max length of the text that can go to BERT
                                 pad_to_max_length=True,  # add [PAD] tokens
                                 return_attention_mask=True,  # add attention mask to not focus on pad tokens
                                 truncation=True
                                 )


# map to the expected input to TFBertForSequenceClassification, see here
def map_example_to_dict(input_ids, attention_masks, token_type_ids, image_list, label):
    return ({
               "input_ids": input_ids,
               "token_type_ids": token_type_ids,
               "attention_mask": attention_masks,
           }, label), (image_list, label)


def encode_examples(ds, image_list, tokenizer, limit=-1):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    if limit > 0:
        ds = ds.take(limit)

    for index, row in ds.iterrows():
        review = row["text"]
        label = row["label"]
        bert_input = convert_example_to_feature(review, tokenizer)

        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices(
        (input_ids_list, attention_mask_list, token_type_ids_list, image_list, label_list)).map(map_example_to_dict)
