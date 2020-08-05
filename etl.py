import os
import sys
import json
import numpy
import click
import random
import embendder
import pandas as pd
import tensorflow as tf
from tensorflow.python.lib.io.tf_record import TFRecordWriter
from transformers import *
from typing import List, Optional, Union
from keras_preprocessing import text
from sklearn.metrics.pairwise import cosine_similarity

BATCH_SIZE = 32
LEN_TF_DATASET = 0
MAX_SEQ_LENGHT = 256
NBR_FEATURE_PREFIX = 'NL_nbr_'
NBR_WEIGHT_SUFFIX = '_weight'
GRAPH_NUM_NEIGHBORS = 2
GRAPH_REGULARIZATION_MULTIPLIER = 0.1
feature_spec = {
    'idx': tf.io.FixedLenFeature([], tf.int64),
    'sentence': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}


def torso_highlighter(text, start_mark, end_mark):
    try:
        start = text.index(start_mark) + len(start_mark)
        text = text[start:]

        try:
            end = text.index(end_mark)
            text = text[:end]
        except:
            pass

        text_split = text.split(" ")
        text = ""
        for word in text_split:
            if len(word) > 0:
                text = text + word + " "
        text = text[:len(text)-1]

    except:
        text = "EMPTY"

    return text


def pad_sequence(sequence):
    pad_size = tf.maximum([0], MAX_SEQ_LENGHT - tf.shape(sequence)[0])
    padded = tf.concat([sequence.values,
                        tf.fill((pad_size),
                                tf.cast(0, sequence.dtype))],
                       axis=0)

    return tf.slice(padded, [0], [MAX_SEQ_LENGHT])


def parse_example_2(example_proto):

    feature_spec = {
        'words': tf.io.VarLenFeature(tf.int64),
        'label': tf.io.FixedLenFeature((), tf.int64, default_value=-1),
    }

    for i in range(GRAPH_NUM_NEIGHBORS):
        nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX,
                                           i,
                                           'words')
        nbr_weight_key = '{}{}{}'.format(NBR_FEATURE_PREFIX,
                                         i,
                                         NBR_WEIGHT_SUFFIX)
        feature_spec[nbr_feature_key] = tf.io.VarLenFeature(tf.int64)

        feature_spec[nbr_weight_key] = tf.io.FixedLenFeature([1],
                                                             tf.float32,
                                                             default_value=tf.constant([0.0]))

    features = tf.io.parse_single_example(example_proto,
                                          feature_spec)

    features['words'] = pad_sequence(features['words'])
    for i in range(GRAPH_NUM_NEIGHBORS):
        nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX,
                                           i,
                                           'words')

        features[nbr_feature_key] = pad_sequence(features[nbr_feature_key])

    labels = features.pop('label')

    return features, labels


def parse_example(example_proto):
      # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_spec)


def max_token(tokens, mark):
    max_value = 0

    for item in tokens:
        if item[mark] > max_value:
            max_value = item[mark]

    return max_value


def min_token(tokens, mark):
    min_value = sys.maxsize

    for item in tokens:
        if item[mark] < min_value:
            min_value = item[mark]

    return min_value


def create_dataset(feauters, labels, batch_size=100, shuffle=False):

    len_data = len(labels)
    dataset = tf.data.Dataset.from_tensor_slices((feauters, labels))

    if shuffle:
        dataset = dataset.shuffle(len_data,
                                  reshuffle_each_iteration=False)

    dataset = dataset.batch(batch_size)

    return dataset


def load_test_and_train_data(train, test, pad_maxlen_cover=0):
    (train_labels, train_feauters) = load_feauters(train)
    (test_labels, test_feauters) = load_feauters(test)

    tokenization = embendder.tokenizer_test_and_train(train=train_feauters,
                                                      test=test_feauters,
                                                      pad_maxlen_cover=pad_maxlen_cover)
    
    tokenization["train_labels"] = numpy.asarray(train_labels, dtype=numpy.int32)
    tokenization["test_labels"] = numpy.asarray(test_labels, dtype=numpy.int32)

    return tokenization


def load_test_and_train_data_bert(train, train_size, test, test_size, batch_size, pretrained="bert-base-multilingual-cased"):
    tokenizer = BertTokenizer.from_pretrained(pretrained)

    train = convert_dataset_to_transformer(dataset=train,
                                           batch_size=batch_size,
                                           tokenizer=tokenizer)
    print("CONVERTED: TRAIN")

    validation = convert_dataset_to_transformer(dataset=test,
                                                batch_size=batch_size,
                                                tokenizer=tokenizer)
    print("CONVERTED: VALIDATION")

    result = {
        "train": train,
        "train_steps_per_epoch": (train_size // batch_size),
        "len_train": train_size,
        "validation": validation,
        "validation_steps_per_epoch": (test_size // batch_size),
        "len_validation": test_size,
    }

    return result


def k_fold_split(k_fold, train, test, batch_size, pad_maxlen_cover=0.99, pad_maxlen_limit=10000):

    splits = []
    tokenization = []

    if type(train) is str and type(test) is str:
        (train_labels, train_feauters) = load_feauters(train)
        (test_labels, test_feauters) = load_feauters(test)

        tokenization = embendder.tokenizer_test_and_train(train=train_feauters,
                                                          test=test_feauters,
                                                          pad_maxlen_cover=pad_maxlen_cover,
                                                          pad_maxlen_limit=pad_maxlen_limit)

        tokenization["train_labels"] = numpy.asarray(train_labels)
        tokenization["test_labels"] = numpy.asarray(test_labels)

        size = len(tokenization["train_labels"])
        print("DATA: LOADED")

    part = int(size * (1 / k_fold))
    with click.progressbar(length=k_fold, label="K-FOLD SPLIT: ", fill_char=click.style('=', fg='white')) as bar:
        for i in range(0, k_fold):

            val_start = (i*part)
            val_end = val_start + part

            train = {}
            validation = {}
            if i == 0:
                train["feauters"] = tokenization["train_feauters"][part:]
                train["labels"] = numpy.asarray(
                    tokenization["train_labels"][part:])
            else:
                train["feauters"] = numpy.concatenate(
                    (tokenization["train_feauters"][:val_start], tokenization["train_feauters"][val_end:]), axis=0)
                train["labels"] = numpy.concatenate(
                    (tokenization["train_labels"][:val_start], tokenization["train_labels"][val_end:]), axis=0)

            validation["feauters"] = tokenization["train_feauters"][val_start:val_end]
            validation["labels"] = numpy.asarray(
                tokenization["train_labels"][val_start:val_end])

            size_train = len(train["feauters"])
            size_validation = len(validation["feauters"])

            item = {
                "train": train,
                "train_steps_per_epoch": (size_train // batch_size),
                "len_train": size_train,
                "validation": validation,
                "validation_steps_per_epoch": (size_validation // batch_size),
                "len_validation": size_validation,
            }

            splits.append(item)

            bar.update(1)

    return (splits, tokenization)


def k_fold_split_old(k_fold, data, batch_size, size=0, tokenization_method_type="", pad_maxlen=0, pretrained=""):

    splits = []
    tokenization = []

    if type(data) is str:
        (labels, feauters) = load_feauters(data)

        tokenization = embendder.tokenizer(data=feauters,
                                           pad_maxlen=pad_maxlen,
                                           tokenization_method_type=tokenization_method_type)

        feauters = tokenization["feauters"]

        size = len(feauters)
        print("DATA: LOADED")

    part = int(size * (1 / k_fold))

    with click.progressbar(length=k_fold, label="K-FOLD SPLIT: ", fill_char=click.style('=', fg='white')) as bar:
        for i in range(0, k_fold):

            val_start = (i*part)
            val_end = val_start + part

            if type(data) is not str:
                if i == 0:
                    train = data.skip(part)
                else:
                    start = data.range(0, val_start)
                    end = data.range(val_end, size)
                    train = data.take(val_start).concatenate(
                        data.skip(val_end))

                validation = data.skip(val_start).take(part)

                size_train = sum([1 for i in train])
                size_validation = sum([1 for i in validation])

                if "bert" in pretrained:
                    tokenizer = BertTokenizer.from_pretrained(pretrained)
                elif "albert" in pretrained:
                    tokenizer = BertTokenizer.from_pretrained(pretrained)

                if len(pretrained) > 0:
                    train = convert_dataset_to_transformer(dataset=train,
                                                           batch_size=batch_size,
                                                           tokenizer=tokenizer)

                    validation = convert_dataset_to_transformer(dataset=validation,
                                                                batch_size=batch_size,
                                                                tokenizer=tokenizer)
                else:
                    train = train.batch(batch_size)
                    validation = validation.batch(batch_size)
            else:
                train = {}
                validation = {}
                if i == 0:
                    train["feauters"] = feauters[part:]
                    train["labels"] = numpy.asarray(labels[part:])
                else:
                    train["feauters"] = numpy.concatenate(
                        (feauters[:val_start], feauters[val_end:]), axis=0)
                    train["labels"] = numpy.concatenate(
                        (labels[:val_start], labels[val_end:]), axis=0)

                validation["feauters"] = feauters[val_start:val_end]
                validation["labels"] = numpy.asarray(
                    labels[val_start:val_end])

                size_train = len(train["feauters"])
                size_validation = len(validation["feauters"])

            item = {
                "train": train,
                "train_steps_per_epoch": (size_train // batch_size),
                "len_train": size_train,
                "validation": validation,
                "validation_steps_per_epoch": (size_validation // batch_size),
                "len_validation": size_validation,
            }

            splits.append(item)

            bar.update(1)

    return (splits, tokenization)


def convert_dataset_to_transformer(dataset, batch_size, tokenizer, max_sequence_length=256):
    dataset = dataset.map(parse_example)
    dataset = my_glue_convert_examples_to_features(examples=dataset,
                                                   tokenizer=tokenizer,
                                                   max_length=max_sequence_length)
    dataset = dataset.batch(batch_size).repeat(-1)

    return dataset


def my_glue_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None
):
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        return _my_tf_glue_convert_examples_to_features(examples, tokenizer, max_length=max_length)

    return _my_glue_convert_examples_to_features(examples, tokenizer, max_length=max_length)


def _my_tf_glue_convert_examples_to_features(
    examples: tf.data.Dataset, tokenizer: PreTrainedTokenizer, max_length: Optional[int] = None,
) -> tf.data.Dataset:
    processor = MyProcessor()
    examples = [processor.tfds_map(
        processor.get_example_from_tensor_dict(example)) for example in examples]
    features = my_glue_convert_examples_to_features(
        examples, tokenizer, max_length=max_length)

    def gen():
        for ex in features:
            yield (
                {
                    "input_ids": ex.input_ids,
                    "attention_mask": ex.attention_mask,
                    "token_type_ids": ex.token_type_ids,
                },
                ex.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32,
          "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


def _my_glue_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    processor = MyProcessor()

    label_list = processor.get_labels()
    output_mode = "classification"

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    return features


class MyProcessor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_labels(self):
        """See base class."""
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = None if set_type == "test" else line[1]
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def train_and_test_splitter_to_transformer(dataset, size, rate):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    TRAIN_SIZE = int(size * rate)
    TEST_SIZE = size - TRAIN_SIZE
    STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE
    VALIDATION_STEPS = TEST_SIZE // BATCH_SIZE
    Xtrain = dataset.take(TEST_SIZE)
    Xtest = dataset.skip(TEST_SIZE)

    Xtrain = Xtrain.map(parse_example)
    Xtrain = glue_convert_examples_to_features(examples=Xtrain,
                                               tokenizer=tokenizer,
                                               max_length=MAX_SEQUENCE_LENGTH,
                                               task='sst-2',
                                               label_list=['0', '1'])
    Xtrain = Xtrain.batch(BATCH_SIZE).repeat(-1)

    Xtest = Xtest.map(parse_example)
    Xtest = glue_convert_examples_to_features(examples=Xtest,
                                              tokenizer=tokenizer,
                                              max_length=MAX_SEQUENCE_LENGTH,
                                              task='sst-2',
                                              label_list=['0', '1'])
    Xtest = Xtest.batch(BATCH_SIZE).repeat(-1)

    return (Xtrain, Xtest)


def create_corpus(output_file_name, data):
    size = len(data)
    output_corpus = open(output_file_name, "w", encoding="utf-8")
    my_filters = '"#$&()*+/:;<=>?@[\\]^_`{|}~\t\n'

    with click.progressbar(length=len_data, label="CREATE CORPUS: ", fill_char=click.style('=', fg='white')) as bar:
        for i in range(0, len_data):
            tmp = str(data[i]).lower()
            tmp = text.text_to_word_sequence(text=tmp,
                                             filters=my_filters)
            tmp = " ".join(map(str, tmp))
            output_corpus.write(tmp + "\n")

            bar.update(1)

    output_corpus.close()


def create_train_to_pattern(output_file_name, data, templates):
    len_data = len(data)
    output_corpus = open(output_file_name, "w", encoding="utf-8")
    min_prev_tokens_count = min_token(templates, "prev_tokens_count")
    min_next_tokent_count = min_token(templates, "next_tokens_count")
    max_prev_tokens_count = max_token(templates, "prev_tokens_count")
    max_next_tokent_count = max_token(templates, "next_tokens_count")
    position = max_prev_tokens_count + 1
    len_mask = max_prev_tokens_count + max_next_tokent_count

    lenv = len(data["obsval"])
    with click.progressbar(length=lenv, label="CREATE TRAINING TO PATTERN (STANDARD EXAMPELS): ", fill_char=click.style('=', fg='white')) as bar:
        for i in range(0, lenv):
            try:
                tmp_split = data["obsval"][i].lower().split(" ")
                sample = ["<unk>"] * len_mask

                len_tmp = len(tmp_split)
                for index in range(0, len_tmp):

                    tmp_min_next_tokent_count = index + 1 + min_next_tokent_count
                    tmp_min_prev_tokens_count = index - min_prev_tokens_count
                    if tmp_min_prev_tokens_count >= 0 and tmp_min_next_tokent_count < len_tmp and len(tmp_split[index]) > 0:

                        tmp_max_next_tokent_count = index + 1 + max_next_tokent_count
                        if(tmp_max_next_tokent_count > len_tmp):
                            tmp_max_next_tokent_count = len_tmp

                        tmp_max_prev_tokens_count = index - max_prev_tokens_count
                        if(tmp_max_prev_tokens_count < 0):
                            tmp_max_prev_tokens_count = 0

                        prev_env = tmp_split[tmp_max_prev_tokens_count:index]
                        next_env = tmp_split[(
                            index+1):tmp_max_next_tokent_count]

                        start_position = position - len(prev_env) - 1
                        for item in prev_env:
                            sample[start_position] = item
                            start_position = start_position + 1

                        start_position = position - 1
                        for item in next_env:
                            sample[start_position] = item
                            start_position = start_position + 1

                        output_corpus.write(
                            tmp_split[index] + "\t" + " ".join(sample) + "\n")

                bar.update(1)
            except:
                bar.update(1)

    for template in templates:
        sample = ["<unk>"] * len_mask
        start_position = position - template["prev_tokens_count"] - 1
        tmp_sample = template["sample"].lower().split(" ")
        len_env = len(tmp_sample) + start_position

        for i in range(start_position, len_env):
            sample[i] = tmp_sample[i-start_position]

        with click.progressbar(length=template["sample_count"], label="CREATE TRAINING TO PATTERN (POSITIVE EXAMPELS): ", fill_char=click.style('=', fg='white')) as bar:
            for j in range(0, template["sample_count"]):
                output_corpus.write("SAMPLE\t" + " ".join(sample) + "\n")
                bar.update(1)


def create_similarity_matrix(data, weights, samples):

    lenv = len(data)
    lens = len(samples[0])
    lenss = len(samples)

    weights_sim = cosine_similarity(weights)
    weights_sim = 1 + weights_sim
    weights_sim = weights_sim / 2
    weights_sim = weights_sim / lens
    similarities = []
    with click.progressbar(length=lenv, label="CREATING SIMILARITY MATRIX: ", fill_char=click.style('=', fg='white')) as bar:
        for i in range(0, lenv):
            calc_buffer = []
            for n in range(0, lenss):
                calc_buffer.append(0.0)

            for j in range(0, lens):
                for n in range(0, lenss):
                    # K = data[i][j]
                    # T = samples[n][j]
                    # F = weights_sim[K][F]
                    calc_buffer[n] = calc_buffer[n] + \
                        weights_sim[data[i][j]][samples[n][j]]

            similarities.append(calc_buffer)

            if i % 1000 == 0:
                bar.update(1000)

    similarities = numpy.asarray(similarities)

    return similarities


def load_config(path):
    f_json = open(path)
    result = json.load(f_json)
    f_json.close()

    print("CONFIG IS LOADED: " + path)

    return result


def load_feauters(input_file_name, random_state=-1, ):
    df = pd.read_excel(input_file_name)

    if random_state >= 0:
        df = df.sample(frac=1,
                       random_state=random_state)

    labels = df.iloc[:, 0]
    feauters = df.iloc[:, 1, ]

    print("FEATURES IS LOADED: " + input_file_name)

    return (labels, feauters)


def load_tf_dataset(path, shuffle=True, reshuffle_each_iteration=False):
    dataset = tf.data.TFRecordDataset(path)
    size = sum([1 for item in dataset])
    if shuffle:
        dataset = dataset.shuffle(size,
                                  reshuffle_each_iteration=reshuffle_each_iteration)

    print("DATA IS LOADED: " + path)

    return (dataset, size)


def load_train_to_pattern(input_file_name):
    results = []
    input_file = open(input_file_name, "r", encoding="utf-8")

    with click.progressbar(input_file, label="LOAD PATTERNS TO TRAINING: ", fill_char=click.style('=', fg='white')) as input_file_bar:
        for row in input_file_bar:
            if(len(row) == 0):
                continue

            tmp_row = row.split("\t")
            results.append(tmp_row[1])

    return results


def create_tfrecord(input_file_name, output_file_name, random_state=-1):

    (labels, feauters) = load_feauters(input_file_name=input_file_name,
                                       random_state=random_state)

    len_feauters = len(feauters.values)
    writer = TFRecordWriter(output_file_name)
    for i in range(0, len_feauters):
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'idx': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
            # Sentence is the yelp review which is stored in UTF-8 bytes
            'sentence': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feauters.values[i].encode('utf-8')])),
            # Label is the sentiment value we are trying to predict
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels.values[i]]))
        }))
        writer.write(tf_example.SerializeToString())
    writer.close()


def split_test_and_training(input_file_name, labels, test=0.2, random_state=-1):
    # Load featuers and labels
    (labels, feauters) = load_feauters(input_file_name)

    # Create temporary dictionary
    docs = {}
    for label in labels:
        docs[label] = []

    # Group Feauters
    len_feutures = len(feauters)
    with click.progressbar(length=len_feutures, label="GROUP FEAUTERS: ", fill_char=click.style('=', fg='white')) as bar:
        for i in range(0, len_feutures):
            if labels[i] in labels:
                docs[labels[i]].append(feauters[i])
            else:
                continue

            bar.update(1)

    # Random down sampling
    tmp_test_labels = []
    tmp_test_feauters = []
    tmp_train_labels = []
    tmp_train_feauters = []
    with click.progressbar(length=len(labels), label="DOWN SAMPLING: ", fill_char=click.style('=', fg='white')) as bar:
        for key in docs.keys():
            test_size = int(round(len(docs[key]) * test, 0))
            train_size = len(docs[key]) - test_size

            tmp = random.shuffle(docs[key])
            tmp_train_feauters = tmp_train_feauters + docs[key][:train_size]
            tmp_train_labels = tmp_train_labels + ([key] * train_size)
            tmp_test_feauters = tmp_test_feauters + docs[key][train_size:]
            tmp_test_labels = tmp_test_labels + ([key] * test_size)

            bar.update(1)

    new_train_df = pd.DataFrame()
    new_train_df["labels"] = tmp_train_labels
    new_train_df["feauters"] = tmp_train_feauters
    new_test_df = pd.DataFrame()
    new_test_df["labels"] = tmp_test_labels
    new_test_df["feauters"] = tmp_test_feauters

    if random_state >= 0:
        # new_df = shuffle(new_df)
        new_train_df = new_train_df.sample(frac=1,
                                           random_state=random_state)
        new_test_df = new_test_df.sample(frac=1,
                                         random_state=random_state)
        print("DATA: SHUFFLED")

    # write excell
    file_name = input_file_name.split(".")[0]
    new_train_df.to_excel(excel_writer=file_name + "_train" + ".xlsx",
                          index=False)
    new_test_df.to_excel(excel_writer=file_name + "_test" + ".xlsx",
                         index=False)

    print("TEST AND TRAIN SPLITTING: COMPLETE")


def create_sampling(input_file_name, output_file_name, labels, number=0, random_state=-1):
    # Load featuers and labels
    (labels, feauters) = load_feauters(input_file_name)

    # Create temporary dictionary
    docs = {}
    for label in labels:
        docs[label] = []

    # Group Feauters
    len_feutures = len(feauters)
    with click.progressbar(length=len_feutures, label="GROUP FEAUTERS: ", fill_char=click.style('=', fg='white')) as bar:
        for i in range(0, len_feutures):
            if labels[i] in labels:
                docs[labels[i]].append(feauters[i])
            else:
                continue

            bar.update(1)

    # Random down sampling
    tmp_labels = []
    tmp_feauters = []
    with click.progressbar(length=len(labels), label="DOWN SAMPLING: ", fill_char=click.style('=', fg='white')) as bar:
        for key in docs.keys():
            if number == 0:
                tmp_feauters = tmp_feauters + \
                    random.sample(docs[key], len(docs[key]))
                tmp_labels = tmp_labels + ([key] * len(docs[key]))
            else:
                tmp_feauters = tmp_feauters + random.sample(docs[key], number)
                tmp_labels = tmp_labels + ([key] * number)

            bar.update(1)

    new_df = pd.DataFrame()
    new_df["labels"] = tmp_labels
    new_df["feauters"] = tmp_feauters
    if random_state >= 0:
        # new_df = shuffle(new_df)
        new_df = new_df.sample(frac=1,
                               random_state=random_state)
        print("DATA: SHUFFLED")

    # write excell
    new_df.to_excel(excel_writer=output_file_name,
                    index=False)

    print("DOWN SAMPLING: COMPLETE")
