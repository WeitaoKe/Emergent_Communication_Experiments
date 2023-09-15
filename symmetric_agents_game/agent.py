import numpy as np
import tensorflow as tf
from keras.layers import Embedding, LSTM, Dense
from keras.initializers import glorot_uniform
from keras.models import Model
from keras import regularizers
from keras.layers import Dropout

# Constants
batch_size = 96
vocab_size = 5
embedding_dim = 128
hidden_units = 256
image_feature_size = 2048
max_sequence_length = 6
start_token = vocab_size
end_token = vocab_size + 1
max_decoding_steps = max_sequence_length - 1
l2_lambda = 0.01  # Regularization strength
DR_Rate = 0.1

# Placeholder
f_t = np.random.rand(batch_size, embedding_dim)


class Agent(Model):
    embedding_layer = Embedding(vocab_size + 2, embedding_dim)
    embedding_layer.build((None,))

    def __init__(self):
        super(Agent, self).__init__()
        # dummy_input = tf.zeros((1, 1))  # assuming input is of shape (batch_size, sequence_length)
        # _ = embedding_layer(dummy_input)

    # def create_lstm_layers(self, num_layers):
    #    lstm_layers = [LSTM(hidden_units, return_sequences=True, return_state=True,
    #                        kernel_regularizer=regularizers.l2(l2_lambda),
    #                        input_shape=(1, 384)) if i == 0 else LSTM(hidden_units, return_sequences=True,
    #                                                                  return_state=True,
    #                                                                  kernel_regularizer=regularizers.l2(l2_lambda)) for
    #                   i in range(num_layers)]
    #    return lstm_layers

    def gumbel_softmax(logits, temperature=1.2, hard=False):
        gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(logits.shape, minval=0, maxval=1)))
        y = tf.nn.softmax((logits + gumbel_noise) / temperature)
        if hard:
            k = tf.shape(logits)[-1]
            y_hard = tf.cast(tf.one_hot(tf.argmax(y, axis=-1), k), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y

class CommunicationPair:
    def __init__(self):
        self.shared_lstm_layers = [LSTM(hidden_units, return_sequences=True, return_state=True,
                                        kernel_regularizer=regularizers.l2(l2_lambda),
                                        input_shape=(1, 384)) for _ in range(5)]
        self.sender = Sender(self.shared_lstm_layers)
        self.receiver = Receiver(self.shared_lstm_layers)


class Sender(Agent):
    def __init__(self, shared_lstm_layers):
        super(Sender, self).__init__()
        self.lstm_layers = shared_lstm_layers
        self.dropout = Dropout(DR_Rate)
        self.token_probs_layer = Dense(vocab_size + 2, activation='softmax',
                                       kernel_regularizer=regularizers.l2(l2_lambda), name='token_probs_layer')
        self.h0 = Dense(hidden_units, activation='tanh', kernel_initializer=glorot_uniform(),
                        kernel_regularizer=regularizers.l2(l2_lambda), name='h0')
        self.hidden_state_transform = Dense(hidden_units, activation='linear',
                                            kernel_regularizer=regularizers.l2(l2_lambda),
                                            name='hidden_state_transform')
        self.cell_state_transform = Dense(hidden_units, activation='linear',
                                          kernel_regularizer=regularizers.l2(l2_lambda), name='cell_state_transform')

    def call(self, f_t, training=True):
        f_t_flattened = f_t
        hidden_state = self.hidden_state_transform(f_t_flattened)
        cell_state = self.cell_state_transform(f_t_flattened)
        sampled_token = tf.fill([batch_size, 1], start_token)
        generated_tokens = []
        lstm_output = self.h0(f_t)
        lstm_output = tf.expand_dims(lstm_output, axis=1)

        for i, lstm_layer in enumerate(self.lstm_layers):
            sampled_token_expanded = Agent.embedding_layer(sampled_token)
            input_sequence = tf.concat([sampled_token_expanded, lstm_output], axis=-1)
            lstm_output, hidden_state, cell_state = lstm_layer(input_sequence, initial_state=[hidden_state, cell_state])
            # print(hidden_state)
            lstm_output = self.dropout(lstm_output, training=training)
            token_probs = self.token_probs_layer(lstm_output)
            token_probs_squeezed = tf.squeeze(token_probs, axis=1)
            gumbel_output = Agent.gumbel_softmax(token_probs_squeezed)
            generated_tokens.append(gumbel_output)

        return tf.stack(generated_tokens, axis=1)


class Receiver(Agent):
    def __init__(self, shared_lstm_layer):
        super(Receiver, self).__init__()
        self.lstm_layers = shared_lstm_layer
        self.dropout = Dropout(DR_Rate)
        self.decision_layer = Dense(image_feature_size, activation='linear',
                                    kernel_regularizer=regularizers.l2(l2_lambda), name='decision_layer')


    def call(self, received_message, training=True):
        lstm_states = [(tf.zeros((batch_size, hidden_units)), tf.zeros((batch_size, hidden_units))) for _ in
                       self.lstm_layers]
        lstm_output = tf.zeros((batch_size, 1, hidden_units))

        last_hidden_state = None
        for i, lstm_layer in enumerate(self.lstm_layers):
            token_probs = received_message[:, i]
            embedding_weights = Agent.embedding_layer.get_weights()[0]
            weighted_embedding = tf.matmul(tf.expand_dims(token_probs, 1), embedding_weights)
            input_sequence = tf.concat([weighted_embedding, lstm_output], axis=-1)
            lstm_output, new_hidden_state, new_cell_state = lstm_layer(input_sequence, initial_state=lstm_states[i])
            lstm_output = self.dropout(lstm_output, training=training)
            lstm_states[i] = (new_hidden_state, new_cell_state)
            last_hidden_state = new_hidden_state
            # print(new_hidden_state)

        decision = self.decision_layer(last_hidden_state)
        return decision
