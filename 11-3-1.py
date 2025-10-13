from turtle import mode
from tensorflow import keras
import tensorflow as tf

class TransformerEncoder(keras.layers.Layer):
    def __init__(self,embed_dim,dense_dim,num_heads,**kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads,key_dim=embed_dim)
        self.dense_proj = keras.Sequential([
            keras.layers.Dense(dense_dim,activation="relu"),
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()
    
    def call(self,inputs,mask=None):
        if mask is not None:
            mask = mask[:,tf.newaxis,:]
        attention_output = self.attention(
            inputs,inputs,attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim":self.embed_dim,
            "dense_dim":self.dense_dim,
            "num_heads":self.num_heads,
        })
        return config

class PositionalEmbedding(keras.layers.Layer):
    def __init__(self,sequence_length,input_dim,output_dim,**kwargs):
        super().__init__(**kwargs)
        self.token_emb = keras.layers.Embedding(
            input_dim=input_dim,output_dim=output_dim)
        self.pos_emb = keras.layers.Embedding(
            input_dim=sequence_length,output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def call(self,inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0,limit=length,delta=1)
        embedded_tokens = self.token_emb(inputs)
        embedded_positions = self.pos_emb(positions)
        return embedded_tokens + embedded_positions
    
    def compute_mask(self,inputs,mask=None):
        return tf.math.not_equal(inputs,0)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim":self.output_dim,
            "sequence_length":self.sequence_length,
            "input_dim":self.input_dim,
        })
        return config

batch_size = 32
max_length = 600
max_tokens = 20000

vocab_size = 20000
embed_dim = 256
num_heads = 2
dense_dim = 32

train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train", batch_size=batch_size)
val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/val", batch_size=batch_size)
test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size)

text_vectorization = keras.layers.TextVectorization(
    ngrams=2,
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length
    )
text_only_train_ds = train_ds.map(lambda x, y: x)

text_vectorization.adapt(text_only_train_ds)

# 查看 text_vectorization 的內容
print("=== Text Vectorization 資訊 ===")
print(f"詞彙表大小: {text_vectorization.vocabulary_size()}")
print(f"最大 token 數量: {text_vectorization.get_config()['max_tokens']}")
print(f"輸出模式: {text_vectorization.get_config()['output_mode']}")
print(f"N-gram 設定: {text_vectorization.get_config()['ngrams']}")
print(f"輸出序列長度: {text_vectorization.get_config()['output_sequence_length']}")
print(f"標準化: {text_vectorization.get_config()['standardize']}")
print(f"分割: {text_vectorization.get_config()['split']}")

# 查看前 20 個詞彙
vocab = text_vectorization.get_vocabulary()
print(f"\n前 20 個詞彙:")
for i, word in enumerate(vocab[:20]):
    print(f"{i}: {word}")

# 測試一個樣本文本的向量化結果
sample_text = "This is a great movie!"
vectorized = text_vectorization([sample_text])
print(f"\n樣本文本: '{sample_text}'")
print(f"向量化結果形狀: {vectorized.shape}")
print(f"向量化結果 (前 20 個值): {vectorized.numpy()[0][:20]}")

int_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)

int_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)

int_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)

inputs = keras.Input(shape = (None,), dtype="int64")
x = PositionalEmbedding(sequence_length=max_length,input_dim=vocab_size,output_dim=embed_dim)(inputs)
x = TransformerEncoder(embed_dim=embed_dim,dense_dim=dense_dim,num_heads=num_heads)(x)
x = keras.layers.GlobalMaxPooling1D()(x)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(1,activation="sigmoid")(x)
model = keras.Model(inputs,outputs)
model.compile(optimizer = "rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("transformer_encoder.keras",
                                    save_best_only=True)
]

model.fit(int_train_ds,
          validation_data = int_val_ds,
          epochs = 10,
          callbacks=callbacks)

# 載入最佳模型進行測試
best_model = keras.models.load_model("transformer_encoder.keras",
                                    custom_objects={
                                        "TransformerEncoder":TransformerEncoder,
                                        "PositionalEmbedding":PositionalEmbedding
                                    })
print(f"Test acc:{best_model.evaluate(int_test_ds)[1]:.3f}")