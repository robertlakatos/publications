import keras
from keras_self_attention import SeqSelfAttention

# https://www.sciencedirect.com/science/article/pii/S0925231219301067
class ACBiLSTM():
    def __init__(self, input_dim, emb_vocab_size, dense):
        self.model = keras.models.Sequential(name="AC-BiLSTM")

        self.model.add(keras.layers.Embedding(input_dim=emb_vocab_size,
                                              output_dim=300,
                                              input_length=input_dim))

        self.model.add(keras.layers.Conv1D(filters=100,
                                           kernel_size=3,
                                           strides=1,
                                           activation="relu"))

        self.model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=150,
                                                                    return_sequences=True)))

        # https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
        self.model.add(SeqSelfAttention(attention_activation='sigmoid'))

        self.model.add(keras.layers.Flatten())

        self.model.add(keras.layers.Dense(units=dense,
                                          activation="softmax"))

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)

    def summary(self):
        self.model.summary()

    def fit(self, x=None, y=None, batch_size=50, epochs=1, verbose=1,
            callbacks=None, validation_data=None):
        return self.model.fit(x=x,
                              y=y,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=verbose,
                              callbacks=callbacks,
                              validation_data=validation_data)

    def evaluate(self, x=None, y=None, batch_size=32, verbose=1):
        self.model.evaluate(self,
                            x=x,
                            y=y,
                            batch_size=batch_size,
                            verbose=verbose)

    def predict(self, x):
        return self.model.predict(x)
