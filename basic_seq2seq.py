'''
@author: wanzeyu

@contact: wan.zeyu@outlook.com

@file: basic_seq2seq.py

@time: 2017/12/7 17:16
'''

import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Embedding, GRU
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from helper import get_data_v2, get_data_v2_offset, vocab, id_vocab, manual_one_hot

# file reader
original = pad_sequences(get_data_v2("train_source.txt"), padding="post", value=1.0, maxlen=30)
paraphrase = pad_sequences(get_data_v2("train_target.txt"), padding="post", value=1.0, maxlen=30)
target_paraphrase = manual_one_hot(
    pad_sequences(get_data_v2_offset("train_target.txt"), padding="post", value=1.0, maxlen=30))

# parameters
latent_dim = 256
num_encoder_tokens = 30
num_decoder_tokens = len(vocab)
batch_size = 32
epochs = 200

# Define an input sequence and process it.
# encoder_inputs = Input(shape=(time_steps, input_dim,), )
encoder_inputs = Input(shape=(None,), name="EncoderInput_1")
x = Embedding(num_encoder_tokens, num_decoder_tokens)(encoder_inputs)
encoder = GRU(latent_dim, return_state=True)
x, state_h = encoder(x)
# We discard `encoder_outputs` and only keep the states.
encoder_states = state_h

# Set up the decoder, using `encoder_states` as initial state.
# decoder_inputs = Input(shape=(time_steps, input_dim,), )
decoder_inputs = Input(shape=(None,), name="DecoderInput_1")
embedded_decoder_inputs = Embedding(30, num_decoder_tokens)(decoder_inputs)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use thex
# return states in the training model, but we will use them in inference.
decoder_lstm = GRU(latent_dim, return_sequences=True, return_state=True)
x, _ = decoder_lstm(embedded_decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(x)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.compile(optimizer='sgd', loss='categorical_crossentropy')
# model.compile(optimizer='adam', loss='categorical_crossentropy')
print("begin training")
model.fit([original, paraphrase], target_paraphrase,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[TensorBoard(log_dir="./logs"),
                     ModelCheckpoint(filepath="model.h5", verbose=1, save_weights_only=True, save_best_only=True),
                     EarlyStopping(monitor="val_loss")]
          )

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(None,), name="DecoderStateInput_1")

decoder_outputs, decoder_state_h = decoder_lstm(embedded_decoder_inputs)
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + [decoder_state_input_h],
    [decoder_outputs] + [decoder_state_h])


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 30))
    current_step = 0
    # Populate the first character of target sequence with the start character.
    target_seq[0, current_step] = vocab["<S>"]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h = decoder_model.predict([target_seq] + [states_value])
        current_step += 1
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, current_step, :])
        sampled_char = id_vocab[sampled_token_index]
        decoded_sentence += sampled_char + " "

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == '</S>' or len(decoded_sentence) > 30:
            stop_condition = True

        # Update the target sequence (of length 1).
        # target_seq = np.zeros((1, 30))
        target_seq[0, current_step] = sampled_token_index

        # Update states
        states_value = h

    return decoded_sentence


print(decode_sequence(original[0]))
