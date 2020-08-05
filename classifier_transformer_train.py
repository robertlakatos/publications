import etl
import cnnb
import math
from classifier_transformer import ClassifierTransformer
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

    ct = ClassifierTransformer(input_dim=tokenization["maxlen"],
                               emb_vocab_size=tokenization["vocab_size"],
                               dense=config_build["output_dim"])

    ct.summary()

    batch_size = 50
    step_num = len(train_x) // batch_size
    warmup_steps = 4000
    lrate = (ct.embed_dim ** -0.5) * min([step_num ** -0.5, step_num * (warmup_steps ** -0.5)])

    optimizer = tf.keras.optimizers.Adam(beta_1=0.9,
                                         beta_2=0.998,
                                         learning_rate=lrate,
                                         epsilon=1e-08)

    ct.compile(optimizer=optimizer,
               loss=config_build["loss"],
               metrics=config_build["metrics"])

    history = ct.fit(x=train_x,
                     y=train_y,
                     epochs=50,
                     batch_size=batch_size,
                     callbacks=[stoppoint],
                     validation_data=(validation_x, validation_y),
                     verbose=1)


if __name__ == "__main__":
    main()
