import etl
import cnnb
from keras.callbacks import EarlyStopping

import math


def main():

    e = 500
    n = round(math.log(e/4)/math.log(2), 0)
    f = int(2**n)
    print(f)

    # load configuration
    nn_layers = etl.load_config("config/nn_layers.json")
    config_build = etl.load_config("config/build_otrs.json") 

    # read save if it exists
    (nn_base, score) = cnnb.load_model("save/save_otrs.json")

    # build network
    cnnb.build(nn_base=nn_base,
               score=score,
               nn_layers=nn_layers,
               config=config_build,
               train="otrs_labelled_train.xlsx",
               test="otrs_labelled_test.xlsx")

    # evaluate
    for i in range(0, 3):
        tokenization = etl.load_test_and_train_data(train="otrs_labelled_shuffled_downsampled_train.xlsx",
                                                    test="otrs_labelled_shuffled_downsampled_test.xlsx",
                                                    pad_maxlen_cover=nn_base[0]["input_length"])
        train_x = tokenization["train_feauters"]
        train_y = tokenization["train_labels"]
        validation_x = tokenization["test_feauters"]
        validation_y = tokenization["test_labels"]

        cnnb.evaluate(train_x=train_x,
                    train_y=train_y,
                    test_x=validation_x,
                    test_y=validation_y,
                    config=config_build,
                    model_layers=nn_base)


if __name__ == "__main__":
    main()
