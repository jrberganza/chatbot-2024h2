import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import re
import time

# Verificar la versión de TensorFlow
print(f"TensorFlow Version: {tf.__version__}")

# Cargar los datos
lines = (
    open("movie_lines.txt", encoding="utf-8", errors="ignore").read().split("\n")
)
conv_lines = (
    open("movie_conversations.txt", encoding="utf-8", errors="ignore")
    .read()
    .split("\n")
)

# Crear un diccionario para mapear cada línea con su texto
id2line = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# Crear una lista de todas las conversaciones
convs = []
for line in conv_lines:
    _line = line.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    convs.append(_line.split(","))

# Ordenar las sentencias en preguntas y respuestas
questions = []
answers = []

for conv in convs:
    for i in range(len(conv) - 1):
        questions.append(id2line.get(conv[i], ""))
        answers.append(id2line.get(conv[i + 1], ""))

# Función para limpiar el texto
def clean_text(text):
    """Limpia el texto eliminando caracteres innecesarios y formateando las palabras."""

    text = text.lower()

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    return text

# Limpiar las preguntas y respuestas
clean_questions = [clean_text(question) for question in questions]
clean_answers = [clean_text(answer) for answer in answers]

# Filtrar las preguntas y respuestas por longitud
min_line_length = 2
max_line_length = 20

short_questions = []
short_answers = []

for question, answer in zip(clean_questions, clean_answers):
    if (
        min_line_length <= len(question.split()) <= max_line_length
        and min_line_length <= len(answer.split()) <= max_line_length
    ):
        short_questions.append(question)
        short_answers.append(answer)

# Crear un diccionario de frecuencia de palabras
vocab = {}
for question in short_questions:
    for word in question.split():
        vocab[word] = vocab.get(word, 0) + 1

for answer in short_answers:
    for word in answer.split():
        vocab[word] = vocab.get(word, 0) + 1

# Eliminar palabras raras del vocabulario
threshold = 10
questions_vocab_to_int = {}
answers_vocab_to_int = {}

word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        questions_vocab_to_int[word] = word_num
        answers_vocab_to_int[word] = word_num
        word_num += 1

# Añadir tokens especiales
codes = ["<PAD>", "<EOS>", "<UNK>", "<GO>"]

for code in codes:
    questions_vocab_to_int[code] = len(questions_vocab_to_int)
    answers_vocab_to_int[code] = len(answers_vocab_to_int)

# Crear diccionarios inversos
questions_int_to_vocab = {v: k for k, v in questions_vocab_to_int.items()}
answers_int_to_vocab = {v: k for k, v in answers_vocab_to_int.items()}

# Añadir token <EOS> al final de cada respuesta
short_answers = [answer + " <EOS>" for answer in short_answers]

# Convertir el texto a enteros y reemplazar las palabras desconocidas con <UNK>
questions_int = []
for question in short_questions:
    ints = []
    for word in question.split():
        ints.append(
            questions_vocab_to_int.get(word, questions_vocab_to_int["<UNK>"])
        )
    questions_int.append(ints)

answers_int = []
for answer in short_answers:
    ints = []
    for word in answer.split():
        ints.append(answers_vocab_to_int.get(word, answers_vocab_to_int["<UNK>"]))
    answers_int.append(ints)

# Ordenar las preguntas y respuestas por la longitud de las preguntas
sorted_questions = []
sorted_answers = []

for length in range(1, max_line_length + 1):
    for question, answer in zip(questions_int, answers_int):
        if len(question) == length:
            sorted_questions.append(question)
            sorted_answers.append(answer)

# Definir el tamaño máximo de las secuencias
max_length_inp = max_line_length
max_length_targ = max_line_length + 1  # +1 por el token <EOS>

# Pad de las secuencias
from keras.preprocessing.sequence import pad_sequences

input_tensor = pad_sequences(
    sorted_questions, maxlen=max_length_inp, padding="post"
)
target_tensor = pad_sequences(
    sorted_answers, maxlen=max_length_targ, padding="post"
)

# Dividir los datos en entrenamiento y validación
from sklearn.model_selection import train_test_split

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = (
    train_test_split(input_tensor, target_tensor, test_size=0.15)
)

# Configuración de hiperparámetros
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE

embedding_dim = 512
units = 512
vocab_inp_size = len(questions_vocab_to_int)
vocab_tar_size = len(answers_vocab_to_int)

dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)
)
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


# Definir el codificador (Encoder)
class Encoder(keras.Model):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        enc_units,
        batch_sz,
        num_layers=1,
        dropout=0.2,
        **kwargs,
    ):
        super(Encoder, self).__init__(**kwargs)
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = keras.layers.LSTM(
            enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
            dropout=dropout,
        )

    def get_config(self):
        return super().get_config() | {
            "vocab_size": vocab_inp_size,
            "embedding_dim": embedding_dim,
            "enc_units": units,
            "batch_sz": BATCH_SIZE,
            "dropout": 0.25,
        }

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x, initial_state=hidden)
        return output, [state_h, state_c]

    def initialize_hidden_state(self):
        return [
            tf.zeros((self.batch_sz, self.enc_units)),
            tf.zeros((self.batch_sz, self.enc_units)),
        ]


# Definir el mecanismo de atención (BahdanauAttention)
class BahdanauAttention(keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = keras.layers.Dense(units)
        self.W2 = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)

    def call(self, query, values):
        # query: [batch_size, hidden size]
        # values: [batch_size, max_len, hidden size]

        query_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# Definir el decodificador (Decoder)
class Decoder(keras.Model):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        dec_units,
        batch_sz,
        num_layers=1,
        dropout=0.2,
        **kwargs,
    ):
        super(Decoder, self).__init__(**kwargs)
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = keras.layers.LSTM(
            dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
            dropout=dropout,
        )
        self.fc = keras.layers.Dense(vocab_size)

        # Implementar el mecanismo de atención
        self.attention = BahdanauAttention(dec_units)

    def get_config(self):
        return super().get_config() | {
            "vocab_size": vocab_tar_size,
            "embedding_dim": embedding_dim,
            "dec_units": units,
            "batch_sz": BATCH_SIZE,
            "dropout": 0.25,
        }

    def call(self, x, hidden, enc_output):
        # x: entrada al decodificador (batch_size, 1)
        # hidden: estado oculto previo (batch_size, hidden size)
        # enc_output: salida del codificador (batch_size, max_length, hidden size)

        context_vector, attention_weights = self.attention(hidden[0], enc_output)

        x = self.embedding(x)

        # Concatenar el vector de contexto con la entrada embebida
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # Pasar por la capa LSTM
        output, state_h, state_c = self.lstm(x)

        # Pasar por la capa densa
        output = tf.reshape(output, (-1, output.shape[2]))  # (batch_size, hidden_size)
        x = self.fc(output)

        return x, [state_h, state_c], attention_weights


if __name__ == "__main__":
    # Inicializar el codificador y el decodificador
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE, dropout=0.25)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE, dropout=0.25)

    # Definir el optimizador y la función de pérdida
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_object = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, questions_vocab_to_int["<PAD>"]))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# Definir el bucle de entrenamiento
@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([answers_vocab_to_int["<GO>"]] * BATCH_SIZE, 1)

        # Teacher forcing
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # Uso de teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = loss / int(targ.shape[1])

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


if __name__ == "__main__":
    # Entrenar el modelo
    EPOCHS = 10

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for batch, (inp, targ) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print(f"Epoch {epoch + 1} Batch {batch} Loss {batch_loss.numpy():.4f}")

        print(f"Epoch {epoch +1} Loss {total_loss / steps_per_epoch:.4f}")
        print(f"Time taken for 1 epoch {time.time() - start:.2f} sec\n")


# Función para preparar una pregunta
def evaluate(sentence, *, encoder, decoder):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = clean_text(sentence)

    inputs = [
        questions_vocab_to_int.get(word, questions_vocab_to_int["<UNK>"])
        for word in sentence.split(" ")
    ]
    inputs = pad_sequences([inputs], maxlen=max_length_inp, padding="post")
    inputs = tf.convert_to_tensor(inputs)

    result = ""

    enc_hidden = [tf.zeros((1, units)), tf.zeros((1, units))]
    enc_output, enc_hidden = encoder(inputs, enc_hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([answers_vocab_to_int["<GO>"]], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(
            dec_input, dec_hidden, enc_output
        )

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += answers_int_to_vocab.get(predicted_id, "<UNK>") + " "

        if answers_int_to_vocab.get(predicted_id) == "<EOS>":
            return result.strip("<EOS>"), sentence

        dec_input = tf.expand_dims([predicted_id], 0)

    return result.strip("<EOS>"), sentence


# Probar el modelo con una pregunta aleatoria
def predict_random_question():
    random_index = np.random.choice(len(short_questions))
    input_sentence = short_questions[random_index]
    print("Pregunta:", input_sentence)
    result, sentence = evaluate(input_sentence, encoder=encoder, decoder=decoder)
    print("Respuesta:", result)


if __name__ == "__main__":
    predict_random_question()
    # Guardar los modelos
    encoder.save("encoder_model.h5")
    decoder.save("decoder_model.h5")

    # Cargar los modelos (si es necesario)
    # encoder = keras.models.load_model('encoder_model.h5', custom_objects={'Encoder': Encoder, 'BahdanauAttention': BahdanauAttention})
    # decoder = keras.models.load_model('decoder_model.h5', custom_objects={'Decoder': Decoder, 'BahdanauAttention': BahdanauAttention})
