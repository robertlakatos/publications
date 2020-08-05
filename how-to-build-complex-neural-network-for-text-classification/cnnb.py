import os
import etl
import json
import click
import random
import warnings
import embendder
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import StratifiedKFold
from transformers import *
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

warnings.simplefilter(action='ignore', category=UserWarning)

layer_global = {"type": "GLOBAL"}
layer_flatten = {"type": "FLATTEN"}
layer_dense = {}

# region METHOD(s)


def load_model(path):

    if os.path.isfile(path):
        with open(path, "r", errors="ignore") as file:
            best = json.load(file)
    else:
        best = {
            "best_score": 0.0,
            "layers": []
        }

    print("MODEL IS LOADED")

    return (best["layers"], best["best_score"])


def create_functional_model(model_layers, config, compile=True):
    input_layer = layers.Input(shape=(config["input_dim"],))
    func_layers = []
    func_layers.append(input_layer)

    for item in model_layers:
        if item["type"] == "EMBEDDING":
            input_layer = layers.Input(shape=(item["input_length"],))
            func_layers = []
            func_layers.append(input_layer)

            embedding_output_dim = item["output_dim"]
            embedding_layer = layers.Embedding(input_dim=item["input_dim"],
                                               output_dim=item["output_dim"],
                                               input_length=item["input_length"])(func_layers[-1])
            func_layers.append(embedding_layer)
        elif(item["type"] == "POSITION_EMBEDDING"):
            input_layer = layers.Input(shape=(item["input_length"],))
            func_layers = []
            func_layers.append(input_layer)

            embedding_output_dim = item["output_dim"]
            position_embedding_layer = PositionEmbedding(vocab_size=item["input_dim"],
                                                         output_dim=item["output_dim"],
                                                         maxlen=item["input_length"])(func_layers[-1])
            func_layers.append(position_embedding_layer)
        elif(item["type"] == "CNNTransformer"):
            cnn_Transformer_layer = CNNTransformer(embed_dim=embedding_output_dim,
                                                   num_heads=item["num_heads"],
                                                   ff_dim=item["ff_dim"])(func_layers[-1])
            func_layers.append(cnn_Transformer_layer)
        elif (item["type"] == "FLATTEN"):
            flatten_layer = layers.Flatten()(func_layers[-1])
            func_layers.append(flatten_layer)
        elif item["type"] == "CNN_1D":
            conv1d_layer = layers.Conv1D(filters=item["num_filters"],
                                         kernel_size=item["kernel_size"],
                                         activation=item["activation"])(func_layers[-1])
            func_layers.append(conv1d_layer)
        elif item["type"] == "MAX_POOLING":
            max_pooling = layers.MaxPooling1D(pool_size=item["max_pooling_size"],
                                              strides=item["max_pooling_strides"])(func_layers[-1])
            func_layers.append(max_pooling)
        elif item["type"] == "BATCH_NORMALIZATION":
            batch_normalization = layers.BatchNormalization()(func_layers[-1])
            func_layers.append(batch_normalization)
        elif item["type"] == "GLOBAL":
            global_layer = layers.GlobalMaxPooling1D()(func_layers[-1])
            func_layers.append(global_layer)
        elif (item["type"] == "DENSE"):
            dense_layer = layers.Dense(units=item["neurons"],
                                       activation=item["activation"])(func_layers[-1])
            func_layers.append(dense_layer)
        elif(item["type"] == "DROPOUT"):
            if(item["dropout"] > 0 and item["dropout"] < 1):
                dropout_layer = layers.Dropout(
                    rate=item["dropout"])(func_layers[-1])
                func_layers.append(dropout_layer)
        else:
            continue

    model = models.Model(inputs=input_layer, outputs=func_layers[-1])

    # optimalizáló és veszteség függvények beállítása
    if compile == True:
        model.compile(optimizer=config["optimizer"],
                      loss=config["loss"],
                      metrics=config["metrics"])

    # modell összegzése
    if(config["verbose"] >= 1):
        model.summary()

    return model


def create_model(model_layers, config, compile=True):

    model = models.Sequential()

    for item in model_layers:
        if(item["type"] == "EMBEDDING"):
            model.add(layers.Embedding(input_dim=item["input_dim"],
                                       output_dim=item["output_dim"],
                                       input_length=item["input_length"]))
        elif(item["type"] == "MAX_POOLING"):
            model.add(layers.MaxPooling1D(pool_size=item["max_pooling_size"],
                                          strides=item["max_pooling_strides"]))
        elif(item["type"] == "CNN_1D"):
            model.add(layers.Conv1D(item["num_filters"],
                                    item["kernel_size"],
                                    activation=item["activation"]))
        elif(item["type"] == "LSTM"):
            if(item["recurrent_dropout"] > 0):
                model.add(layers.LSTM(
                    units=item["unit"], recurrent_dropout=item["recurrent_dropout"]))
            else:
                model.add(layers.LSTM(units=item["units"]))
        elif(item["type"] == "BiLSTM"):
            if(item["recurrent_dropout"] > 0 and item["recurrent_dropout"] < 1):
                model.add(layers.Bidirectional(
                    layers.LSTM(units=item["units"], recurrent_dropout=item["recurrent_dropout"])))
            else:
                model.add(layers.Bidirectional(
                    layers.LSTM(units=item["units"])))
        elif (item["type"] == "GRU"):
            if(item["recurrent_dropout"] > 0 and item["recurrent_dropout"] < 1):
                model.add(layers.GRU(
                    units=item["units"], recurrent_dropout=item["recurrent_dropout"]))
            else:
                model.add(layers.GRU(units=item["units"]))
        elif (item["type"] == "BiGRU"):
            if(item["recurrent_dropout"] > 0):
                model.add(layers.Bidirectional(
                    layers.GRU(units=item["units"], recurrent_dropout=item["recurrent_dropout"])))
            else:
                model.add(layers.Bidirectional(
                    layers.GRU(units=item["units"])))
        elif (item["type"] == "GLOBAL"):
            model.add(layers.GlobalMaxPooling1D())
        elif (item["type"] == "FLATTEN"):
            model.add(layers.Flatten())
        elif (item["type"] == "DENSE"):
            model.add(layers.Dense(item["neurons"],
                                   activation=item["activation"]))
        elif(item["type"] == "DROPOUT"):
            if(item["dropout"] > 0 and item["dropout"] < 1):
                model.add(layers.Dropout(rate=item["dropout"]))
        elif(item["type"] == "SPATIAL_DROPOUT_2D"):
            if(item["dropout"] > 0 and item["dropout"] < 1):
                model.add(layers.SpatialDropout2D(rate=item["dropout"]))
        else:
            continue

    # optimalizáló és veszteség függvények beállítása
    if compile == True:
        model.compile(optimizer=config["optimizer"],
                      loss=config["loss"],
                      metrics=config["metrics"])

    # modell összegzése
    if(config["verbose"] >= 1):
        model.summary()

    return model


def create_model_with_transformer(model_layers=[], pretrained="bert-base-multilingual-cased", metrics=["accuracy"]):

    MyTFBertForSequenceClassification.model_layers = model_layers
    model = MyTFBertForSequenceClassification.from_pretrained(pretrained)

    opt = tf.keras.optimizers.Adam(learning_rate=3e-5,
                                   epsilon=1e-08)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=metrics)

    model.summary()

    return model


def save_model(best_score, layers, config):
    best_training = {
        "best_score": best_score,
        "layers": layers
    }

    with open(config["autosave"]["layers"], 'w', encoding='utf-8') as f:
        f.write(json.dumps(best_training, ensure_ascii=False))

    print("SAVED")
    print(best_training)


def evaluate(train_x, train_y, test_x, test_y, config, model_layers):
    fit_verbose = 0
    if(config["verbose"] >= 3):
        fit_verbose = 1

    model = create_functional_model(model_layers=model_layers,
                                    config=config)

    stoppoint = EarlyStopping(monitor=config["early_stopping_monitor"],
                              patience=config["early_stopping_patience"])

    history = model.fit(x=train_x,
                        y=train_y,
                        batch_size=config["batch_size"],
                        epochs=config["epochs"],
                        validation_data=(test_x, test_y),
                        callbacks=[stoppoint],
                        verbose=fit_verbose)


def cross_validation(k_folds, model_layers, nn_layers, config):
    cvscores = []
    histories = []

    stop_verbose = 0
    if(config["verbose"] >= 2):
        stop_verbose = 1

    fit_verbose = 0
    if(config["verbose"] >= 3):
        fit_verbose = 1

    evaluate_verbose = 0
    if(config["verbose"] == 4):
        evaluate_verbose = 1

    # try:
    for fold in k_folds:
        scores = 0.0
        for i in range(0, config["cross_validation_repeat"]):
            print("CV STEP: " + str(i))
            if "bert" in nn_layers.keys():
                model = create_model_with_transformer(model_layers=model_layers[: -1],
                                                      nn_layers=nn_layers,
                                                      config=config)

                history = model.fit(fold["train"],
                                    epochs=config["epochs"],
                                    steps_per_epoch=fold["train_steps_per_epoch"],
                                    validation_data=fold["validation"],
                                    validation_steps=fold["validation_steps_per_epoch"],
                                    verbose=fit_verbose)
            else:
                model = create_functional_model(model_layers=model_layers,
                                                config=config)

                stoppoint = EarlyStopping(monitor=config["early_stopping_monitor"],
                                          patience=config["early_stopping_patience"])

                history = model.fit(x=fold["train"]["feauters"],
                                    y=fold["train"]["labels"],
                                    batch_size=config["batch_size"],
                                    epochs=config["epochs"],
                                    validation_data=(fold["validation"]["feauters"],
                                                     fold["validation"]["labels"]),
                                    callbacks=[stoppoint],
                                    verbose=fit_verbose)

            scores = scores + \
                max([item for item in history.history["val_accuracy"]]) * 100

        scores = scores / config["cross_validation_repeat"]
        print("Fold Cross Validation: " + str(scores))
        cvscores.append(scores)
        histories.append(history)

    mean_cvscore = np.mean(cvscores)
    print("Fold Cross Validation Total: " + str(mean_cvscore))

    return mean_cvscore

    # except:
    #     print("Model Error")
    #     return 0.0


def calc_permutation_with_reputation(set_element, repeat_count):
    tmp = list(product(set_element, repeat=repeat_count))
    result = []

    for items in tmp:
        tmp_result = []
        for item in items:
            tmp_result.append(item)

        result.append(tmp_result)

    return result


def cnn_abstraction(level_cnn, layer_base, layer_cnn, layer_dropout, layer_maxpool, layer_global,
                    layer_dense, nn_layers, k_folds, config):

    tmp_layers = layer_base.copy()

    for i in range(0, level_cnn):
        tmp_layers.append(layer_cnn)
        if layer_dropout[i]["dropout"] < 1.0:
            tmp_layers.append(layer_dropout[i])

    tmp_layers.append(layer_global)
    tmp_layers.append(layer_dense)

    print("Layers: " + str(tmp_layers))
    mean_cvscore = cross_validation(k_folds,
                                    tmp_layers,
                                    nn_layers,
                                    config)

    # mean_cvscore = random.random()

    act_result = {
        "layers":  tmp_layers,
        "mean_cvscore": mean_cvscore
    }

    return act_result


def cnn_dropout(best_nn, kfold, config, train_data, train_labels):

    print("CNN 1D SEARCH DROPOUT")
    best_results = []
    for dropout in config["cnn_1d"]["dropouts"]:
        tmp_layers = []
        layer_dropout = {
            "type": "DROPOUT",
            "dropout": dropout
        }
        for layer in best_nn:
            tmp_layers.append(layer)

        if(dropout > 0):
            tmp_layers.append(layer_dropout)

        tmp_layers.append(layer_global)
        tmp_layers.append(layer_dense)

        print("Layers: " + str(tmp_layers))
        mean_cvscore = cross_validation(kfold,
                                        tmp_layers,
                                        train_data,
                                        train_labels,
                                        config)

        act_result = {
            "layers":  tmp_layers,
            "mean_cvscore": mean_cvscore
        }
        best_results.append(act_result)

    (fix_layers, best_score) = search_best(best_results)
    save_model(best_score, fix_layers, config)
    fix_layers = fix_layers[:-2]


def search_best(data, save_path=""):

    max_val = 0.0
    best_layer = 0
    i = 0
    for item in data:
        if item["mean_cvscore"] > max_val:
            max_val = item["mean_cvscore"]
            best_layer = item["layers"]

    if len(save_path) > 0:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False))

    return (best_layer, max_val)


def build_embedding(nn_base, nn_layers, config, k_folds, tokenization):

    results = []
    layer_dense = {
        "type": "DENSE",
        "neurons": config["output_dim"],
        "activation": config["activation"]
    }

    # search basic embedding parameters
    start_output_dim = nn_layers["embedding"]["start_output_dim"]
    while start_output_dim <= nn_layers["embedding"]["end_output_dim"]:
        layer_embedding = {
            "type": "EMBEDDING",
            "input_dim": tokenization["vocab_size"],
            "output_dim": start_output_dim,
            "input_length": tokenization["maxlen"]
        }
        layers = nn_base.copy()
        layers[0] = layer_embedding

        print("Layers: " + str(layers))

        mean_cvscore = cross_validation(k_folds,
                                        layers,
                                        nn_layers,
                                        config)

        # mean_cvscore = random.random()

        act_result = {
            "layers":  layers,
            "mean_cvscore": mean_cvscore
        }
        results.append(act_result)
        with open(config["autosave"]["results"] + "_embedding_ck.json", "w", encoding="utf8") as chekpoint_file:
            chekpoint_file.write(json.dumps(results, ensure_ascii=False))

        start_output_dim = start_output_dim + \
            nn_layers["embedding"]["increase_embedding_dim"]

    (best_nn, best_score) = search_best(results,
                                        config["autosave"]["results"] + "_embedding.json")
    save_model(best_score, best_nn, config)

    return (best_nn[:-1], results, best_score)


def build_cnn(nn_base, nn_layers, config, k_folds, transformer=False):

    layer_dense = {
        "type": "DENSE",
        "neurons": config["output_dim"],
        "activation": config["activation"]
    }
    activation = nn_layers["cnn_1d"]["activation"]
    print("SEARCH BEST CNN 1D")

    # region Filterek optimalizálása és kernel méret optimalizálás

    print("CNN 1D FILTER AND KERNEL OPTIMALIZATION")
    results = []
    kernel_size = nn_layers["cnn_1d"]["min_kernel_size"]
    num_filters = nn_layers["cnn_1d"]["min_num_filters"]
    while num_filters <= nn_layers["cnn_1d"]["max_num_filters"]:
        while kernel_size <= nn_layers["cnn_1d"]["max_kernel_size"]:
            layer_cnn = {
                "type": "CNN_1D",
                "activation": activation,
                "kernel_size": kernel_size,
                "num_filters": num_filters
            }
            tmp_base = nn_base.copy()
            tmp_base.append(layer_cnn)
            tmp_base.append(layer_global)
            tmp_base.append(layer_dense)

            print("Layers: " + str(tmp_base))

            mean_cvscore = cross_validation(k_folds,
                                            tmp_base,
                                            nn_layers,
                                            config)

            # mean_cvscore = random.random()

            act_result = {
                "layers":  tmp_base,
                "mean_cvscore": mean_cvscore
            }
            results.append(act_result)
            with open(config["autosave"]["results"] + "_cd_1_ck.json", "w", encoding="utf8") as chekpoint_file:
                chekpoint_file.write(json.dumps(results, ensure_ascii=False))

            kernel_size = kernel_size + \
                nn_layers["cnn_1d"]["increase_kernel_size"]

        kernel_size = nn_layers["cnn_1d"]["min_kernel_size"]
        num_filters = num_filters + \
            nn_layers["cnn_1d"]["increase_num_filters"]

    (nn_base, best_score) = search_best(results)
    save_model(best_score, nn_base, config)
    nn_base = nn_base[:-2]

    # endregion

    # region  Absztrakció mélyítése

    print("CNN 1D INCREASEING ABSTRACTION")
    ab_results = []
    best_kernel_size = nn_base[1]["kernel_size"]
    best_num_filters = nn_base[1]["num_filters"]
    nn_base = nn_base[:-1]
    layer_cnn = {
        "type": "CNN_1D",
        "activation": activation,
        "kernel_size": best_kernel_size,
        "num_filters": best_num_filters
    }
    layer_maxpool = {
        "type": "MAX_POOLING",
        "max_pooling_size": nn_layers["cnn_1d"]["max_pooling_size"],
        "max_pooling_strides": nn_layers["cnn_1d"]["max_pooling_strides"],
    }
    for j in range(1, nn_layers["cnn_1d"]["level_max_pool"]):
        for i in range(1, nn_layers["cnn_1d"]["level_cnn"]):
            dropouts_perm = calc_permutation_with_reputation(nn_layers["cnn_1d"]["dropouts"],
                                                             i)
            layer_dropouts = []
            for dropouts in dropouts_perm:
                tmp_layer_dropout = []
                for dropout in dropouts:
                    layer_dropout = {
                        "type": "DROPOUT",
                        "dropout": dropout
                    }
                    tmp_layer_dropout.append(layer_dropout)
                layer_dropouts.append(tmp_layer_dropout)

            for layer_dropout in layer_dropouts:
                act_result = cnn_abstraction(level_cnn=i,
                                             layer_base=nn_base,
                                             layer_cnn=layer_cnn,
                                             layer_dropout=layer_dropout,
                                             layer_maxpool=layer_maxpool,
                                             layer_global=layer_global,
                                             layer_dense=layer_dense,
                                             nn_layers=nn_layers,
                                             k_folds=k_folds,
                                             config=config)

                ab_results.append(act_result)
                with open(config["autosave"]["results"] + "_cd_1_ck.json", "w", encoding="utf8") as chekpoint_file:
                    chekpoint_file.write(json.dumps(
                        (results + ab_results), ensure_ascii=False))

        (nn_base, best_score) = search_best(ab_results)
        save_model(best_score, nn_base, config)
        nn_base = nn_base[:-2]

        if (j + 1) != nn_layers["cnn_1d"]["level_max_pool"]:
            nn_base.append(layer_maxpool)
        else:
            nn_base.append(layer_global)

    # endregion

    nn_base[:-1]
    return (nn_base, results, best_score)


def build_dense(nn_base,score, nn_layers, config, k_folds):

    results = []
    tmp_score = score

    layer_dense = {
        "type": "DENSE",
        "neurons": config["output_dim"],
        "activation": config["activation"]
    }

    units = [0]
    for item in reversed(nn_base):
        if item["type"] == "CNN_1D":
            units[0] = item["num_filters"]
            break
 
    unit = int(units[-1] / 2)
    while unit > layer_dense["neurons"]:
        units.append(unit)
        unit = int(units[-1] / 2)
        
    while True:
        ab_results = []
        for unit in units:        
            layer_sub_dense = {
                "type": "DENSE",
                "neurons": unit,
                "activation" : "relu"
            }
            for dropout in nn_layers["dense"]["dropouts"]:
                tmp_base = nn_base.copy()
                tmp_base.append(layer_sub_dense)
                if dropout > 0 and dropout < 1:            
                    layer_dropout = {
                        "type": "DROPOUT",
                        "dropout": dropout
                    }
                    tmp_base.append(layer_dropout)                
                tmp_base.append(layer_dense)

                # mean_cvscore = cross_validation(k_folds,
                #                                 tmp_base,
                #                                 nn_layers,
                #                                 config)

                mean_cvscore = random.random() * 100

                act_result = {
                    "layers":  tmp_base,
                    "mean_cvscore": mean_cvscore
                }
                ab_results.append(act_result)

        results = results + ab_results    
        with open(config["autosave"]["results"] + "_dense_ck.json", "w", encoding="utf8") as chekpoint_file:
            chekpoint_file.write(json.dumps(results, ensure_ascii=False))

        (nn_base, best_score) = search_best(ab_results)
        
        if best_score > tmp_score:
            tmp_score = best_score
            for item in reversed(nn_base[:-1]):
                if item["type"] == "DENSE":
                    max_unit = item["neurons"]
                    new_units = []
                    for item in units:
                        if item <= max_unit:
                            new_units.append(item)
                    units = new_units
                    break
        else:
            break

    (best_nn, best_score) = search_best(results,
                                        config["autosave"]["results"] + "_dense.json")
    if best_score >= score:
        save_model(best_score, best_nn, config)

    return (best_nn, results, best_score)


def build(nn_base, score, nn_layers, config, train, test):
    results = []
    best_score = score
    best_nn = nn_base
    nn_base_types = [item["type"].lower() for item in nn_base]

    (k_folds, tokenization) = etl.k_fold_split(k_fold=config["k_fold_n_splits"],
                                               train=train,
                                               test=test,
                                               batch_size=config["batch_size"])

    # Embedding layer building
    if "embedding" in nn_layers and "position_embedding" not in nn_base_types:
        layer_embedding = {
            "type": "POSITION_EMBEDDING",
            "input_dim": tokenization["vocab_size"],
            "output_dim": nn_layers["embedding"]["start_output_dim"],
            "input_length": tokenization["maxlen"]
        }
        best_nn = [layer_embedding]

    # Convolution layer building
    if "cnn_1d" in nn_layers and "cnn_1d" not in nn_base_types:
    # if "cnn_1d" in nn_layers:
        (best_nn, sub_results, best_score) = build_cnn(nn_base=best_nn,
                                                       nn_layers=nn_layers,
                                                       config=config,
                                                       k_folds=k_folds)
        results = results + sub_results    
    else:
        best_nn = nn_base[:-1]

    # if "dense" in nn_layers:        
    #     (best_nn, sub_results, best_score) = build_dense(nn_base=best_nn,
    #                                                      score=best_score,
    #                                                      nn_layers=nn_layers,
    #                                                      config=config,
    #                                                      k_folds=k_folds)

    #     results = results + sub_results  

    if "embedding" in nn_layers:
        (best_nn, sub_results, best_score) = build_embedding(nn_base=nn_base,
                                                             nn_layers=nn_layers,
                                                             config=config,
                                                             k_folds=k_folds,
                                                             tokenization=tokenization)
        results = results + sub_results  

    print(best_nn)
    print(results)

    with open("history.json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(results, ensure_ascii=False))


# endregion

# region CLASSE(S)

class MyTFBertForSequenceClassification(TFBertPreTrainedModel):
    model_layers = []

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # self.num_labels = config.num_labels
        self.num_labels = 11

        # init base layers
        self.bert = TFBertMainLayer(config, name="bert")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(self.num_labels,
                                                kernel_initializer=modeling_tf_utils.get_initializer(
                                                    config.initializer_range),
                                                name="classifier")

        self.output_0_layers = []
        for item in self.model_layers:
            if(item["type"].lower() == "cnn_1d"):
                self.output_0_layers.append(layers.Conv1D(filters=item["num_filters"],
                                                          kernel_size=item["kernel_size"],
                                                          activation=item["activation"]))
            elif(item["type"].lower() == "max_pooling"):
                self.output_0_layers.append(layers.MaxPooling1D(pool_size=item["max_pooling_size"],
                                                                strides=item["max_pooling_strides"]))
            elif (item["type"].lower() == "global"):
                self.output_0_layers.append(layers.GlobalMaxPooling1D())
            elif (item["type"].lower() == "flatten"):
                self.output_0_layers.append(layers.Flatten())
            elif (item["type"].lower() == "dense"):
                self.output_0_layers.append(layers.Dense(item["neurons"],
                                                         activation=item["activation"]))
            elif(item["type"].lower() == "dropout"):
                if(item["dropout"] > 0 and item["dropout"] < 1):
                    self.output_0_layers.append(
                        layers.Dropout(rate=item["dropout"]))
            else:
                continue

    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)

        if len(self.output_0_layers) > 0:
            layers = []
            layers.append(self.output_0_layers[0](outputs[0]))

            len_output_0_layers = len(self.output_0_layers)
            for i in range(1, len_output_0_layers):
                layers.append(self.output_0_layers[i](layers[i - 1]))

            concat = tf.keras.layers.Concatenate()([layers[-1], outputs[1]])

            concat_dropout = self.dropout(concat,
                                          training=kwargs.get("training", False))

            logits = self.classifier(concat_dropout)
        else:
            pooled_output = outputs[1]
            pooled_output = self.dropout(
                pooled_output, training=kwargs.get("training", False))
            logits = self.classifier(pooled_output)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]

        return outputs  # logits, (hidden_states), (attentions)


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        # (batch_size, num_heads, seq_len, projection_dim)
        query = self.separate_heads(query, batch_size)
        # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(key, batch_size)
        # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        # (batch_size, seq_len, num_heads, projection_dim)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len, embed_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim))
        # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(concat_attention)

        return output


class CNNTransformer(layers.Layer):
    """
    https://keras.io/examples/nlp/text_classification_with_transformer/
    https://machinetalk.org/2019/04/29/create-the-transformer-with-tensorflow-2-0/
    https://www.kaggle.com/fareise/multi-head-self-attention-for-text-classification
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(CNNTransformer, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"),
             layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        # return attn_output
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, output_dim):
        super(PositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size,
                                          output_dim=output_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen,
                                        output_dim=output_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# endregion
