import random
import tensorflow as tf
import string
import re

text_file = "spa-eng/spa.txt"
with open(text_file, encoding="utf-8") as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    english, spanish = line.split("\t")

    spanish = "[start] " + spanish + " [end]"
    text_pairs.append((english, spanish))

random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, f"[{re.escape(strip_chars)}]", "")

vacab_size = 15000
sequence_length = 20

source_vectorization = tf.keras.layers.TextVectorization(
    max_tokens=vacab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)

target_vectorization = tf.keras.layers.TextVectorization(
    max_tokens=vacab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)

train_english_texts = [pair[0] for pair in train_pairs]
train_spanish_texts = [pair[1] for pair in train_pairs]
source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_spanish_texts)

batch_size = 64

def format_dataset(eng, spa):
    eng = source_vectorization(eng)
    spa = target_vectorization(spa)
    return ({
            "english": eng,
            "spanish": spa[:, :-1]},
            spa[:, 1:])

def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(2048).prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

embed_dim = 256
latent_dim = 1024

source = tf.keras.Input(shape=(None,), dtype="int64", name="english")
x = tf.keras.layers.Embedding(vacab_size, embed_dim)(source)
encoded_source = tf.keras.layers.Bidirectional(
    tf.keras.layers.GRU(latent_dim), merge_mode="sum")(x)

past_target = tf.keras.Input(shape=(None,), dtype="int64", name="spanish")
x = tf.keras.layers.Embedding(vacab_size, embed_dim,mask_zero=True)(past_target)
decoder_gru = tf.keras.layers.GRU(latent_dim, return_sequences=True)
x = decoder_gru(x, initial_state=encoded_source)
x = tf.keras.layers.Dropout(0.5)(x)
target_next_step = tf.keras.layers.Dense(vacab_size, activation="softmax")(x)
seq2seq = tf.keras.Model([source, past_target], target_next_step)

seq2seq.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

seq2seq.fit(train_ds, epochs=15, validation_data=val_ds)