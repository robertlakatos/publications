import etl
import cnnb
from ac_bilstm import ACBiLSTM
from keras.callbacks import EarlyStopping


def main():
    config_build = etl.load_config("config/build_otrs.json")

    tokenization = etl.load_test_and_train_data(train="otrs_labelled_shuffled_downsampled_train.xlsx",
                                                test="otrs_labelled_shuffled_downsampled_test.xlsx",
                                                pad_maxlen_cover=0.99)
    train_x = tokenization["train_feauters"]
    train_y = tokenization["train_labels"]
    validation_x = tokenization["test_feauters"]
    validation_y = tokenization["test_labels"]

    acbilstm = ACBiLSTM(input_dim=tokenization["maxlen"],
                        emb_vocab_size=tokenization["vocab_size"],
                        dense=config_build["output_dim"])

    acbilstm.summary()

    acbilstm.compile(optimizer=config_build["optimizer"],
                     loss=config_build["loss"],
                     metrics=config_build["metrics"])

    stoppoint = EarlyStopping(monitor=config_build["early_stopping_monitor"],
                              patience=config_build["early_stopping_patience"])

    history = acbilstm.fit(x=train_x,
                           y=train_y,
                           epochs=20,
                           callbacks=[stoppoint],
                           validation_data=(validation_x, validation_y),
                           verbose=1)


if __name__ == "__main__":
    main()
