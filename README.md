# Seq2seq 機器翻譯專案 (英文轉西班牙文)

使用 GRU (Gated Recurrent Unit) 實現的序列到序列 (sequence-to-sequence) 翻譯模型。

## 專案架構圖

```mermaid
graph TD
    A["英文輸入<br/>Hello"] --> B["TextVectorization<br/>詞彙表: 15000<br/>序列長度: 20"]
    C["西班牙文輸出<br/>[start] Hola [end]"] --> D["TextVectorization<br/>詞彙表: 15000<br/>序列長度: 21"]

    B --> E["整數序列<br/>[3, 15, 8, ...]"]
    D --> F["整數序列<br/>[2, 45, 12, ..., 3]"]

    E --> G["Embedding 嵌入層<br/>維度: 256"]
    F --> H["Embedding 嵌入層<br/>維度: 256<br/>mask_zero=True"]

    G --> I["Bidirectional GRU<br/>編碼器<br/>隱藏層: 1024<br/>merge_mode=sum"]
    H --> J["GRU 解碼器<br/>隱藏層: 1024<br/>return_sequences=True"]

    I -->|"初始狀態<br/>initial_state"| J

    J --> K["Dropout<br/>rate: 0.5"]
    K --> L["Dense 輸出層<br/>softmax<br/>15000 個詞彙"]
    L --> M["預測結果<br/>Hola"]

    style I fill:#e1f5ff
    style J fill:#fff4e1
    style L fill:#ffe1e1
```

## 模型組件說明

### Embedding 層 (詞嵌入層)

**作用**: 將整數索引轉換為密集的向量表示

```python
# 編碼器的嵌入層
Embedding(vocab_size, embed_dim)  # (15000, 256)

# 解碼器的嵌入層
Embedding(vocab_size, embed_dim, mask_zero=True)  # (15000, 256)
```

**為什麼需要 Embedding？**
- 神經網絡無法直接處理離散的整數索引
- 將稀疏的 one-hot 編碼轉換為密集向量
- 在訓練中學習詞彙之間的語義關係
- 相似的詞會有相似的向量表示

**輸入輸出轉換：**
```
輸入：整數序列 [3, 15, 8, 42]
     shape: (batch_size, sequence_length)

                    ↓  Embedding 層

輸出：嵌入向量
     shape: (batch_size, sequence_length, 256)
     每個整數變成 256 維的浮點數向量
```

**mask_zero=True 的作用：**
- 將索引 `0` 視為填充符號 (padding)
- 在計算時自動忽略這些位置
- 用於處理不同長度的序列

**範例：**
```python
# 詞彙表: {"hello": 3, "world": 15, "good": 8, ...}
輸入 ID:  [3,    15,    8,    42   ]
         ↓     ↓      ↓     ↓
嵌入向量: [0.2,  [-0.1, [0.5,  [0.3,
          0.5,   0.3,   0.1,  -0.2,
          -0.1,  0.7,   0.4,   0.6,
          ...]   ...]   ...]   ...]
         256維  256維  256維  256維
```

---

### Bidirectional 層 (雙向包裝層)

**作用**: 讓 RNN 同時從兩個方向處理序列

```python
# 編碼器使用雙向 GRU
encoded_source = Bidirectional(
    GRU(latent_dim),  # 1024 維
    merge_mode="sum"
)(x)
```

**為什麼需要 Bidirectional？**
- 單向 RNN 只能看到"過去"的信息
- 雙向可以同時看到前後文，獲得更完整的理解
- 特別適合需要理解整個句子含義的任務（如翻譯編碼器）

**運作方式：**
```
輸入序列: ["I", "love", "you"]

前向 GRU (→):  I  →  love  →  you    (從左到右)
                                ↓
                            h_forward

後向 GRU (←):  I  ←  love  ←  you    (從右到左)
                                ↓
                            h_backward

merge_mode="sum": h_final = h_forward + h_backward
```

**merge_mode 參數比較：**

| merge_mode | 說明 | 輸出維度 | 特點 |
|------------|------|----------|------|
| `"sum"` | 相加 | 1024 | 節省參數，結合兩方向 |
| `"concat"` | 串接 | 2048 | 保留完整信息 |
| `"mul"` | 相乘 | 1024 | 強調共同特徵 |
| `"ave"` | 平均 | 1024 | 平衡兩方向 |
| `None` | 分開 | [1024, 1024] | 分別處理 |

**使用時機：**
- ✅ **適合雙向**: 文本分類、情感分析、**翻譯編碼器**
  - 可以看到整個輸入序列
  - 需要理解完整的上下文

- ❌ **不適合雙向**: 文本生成、**翻譯解碼器**
  - 生成時無法看到未來的詞
  - 必須按順序逐步生成

**在本專案中：**
- **Encoder 用雙向**: 可以完整理解英文輸入句子的含義
- **Decoder 不用雙向**: 逐步生成西班牙文翻譯，不能提前看到未來的詞

---

### Encoder (編碼器 - 雙向 GRU)
- **作用**: 處理英文輸入序列
- **輸出**: 單一上下文向量 (encoded_source)
- **雙向**: 同時從前向後和從後向前讀取序列
- **merge_mode="sum"**: 將兩個方向的輸出相加

### Decoder (解碼器 - GRU)
- **作用**: 生成西班牙文翻譯
- **初始狀態**: 使用編碼器的上下文向量
- **逐步預測**: 每個時間步預測下一個詞
- **Teacher Forcing**: 訓練時使用正確答案作為輸入

### 關鍵參數
- **詞彙表大小**: 15,000 tokens
- **序列長度**: 20 (輸入), 21 (目標)
- **嵌入維度**: 256
- **隱藏層維度**: 1024
- **批次大小**: 64
- **訓練輪數**: 15 epochs

---

## GRU 內部結構圖解

GRU (Gated Recurrent Unit) 是一種改良的 RNN 架構，使用門控機制來控制信息流動。

```mermaid
graph TB
    subgraph "GRU 單元 (時間步 t)"
        Input["輸入 x(t)"]
        HiddenPrev["前一狀態 h(t-1)"]

        Input --> Concat1["串接"]
        HiddenPrev --> Concat1

        Concat1 --> ResetGate["重置門 (Reset Gate)<br/>r = σ(Wr·[h(t-1), x(t)])"]
        Concat1 --> UpdateGate["更新門 (Update Gate)<br/>z = σ(Wz·[h(t-1), x(t)])"]

        ResetGate --> Multiply1["⊙<br/>逐元素相乘"]
        HiddenPrev --> Multiply1

        Multiply1 --> Concat2["串接"]
        Input --> Concat2

        Concat2 --> CandidateState["候選狀態<br/>h̃(t) = tanh(Wh·[r⊙h(t-1), x(t)])"]

        UpdateGate --> Multiply2["⊙<br/>1-z"]
        UpdateGate --> Multiply3["⊙<br/>z"]

        CandidateState --> Multiply2
        HiddenPrev --> Multiply3

        Multiply2 --> Add["⊕<br/>相加"]
        Multiply3 --> Add

        Add --> Output["新狀態 h(t)<br/>h(t) = (1-z)⊙h̃(t) + z⊙h(t-1)"]
    end

    style ResetGate fill:#ffe1e1
    style UpdateGate fill:#e1f5ff
    style CandidateState fill:#e1ffe1
    style Output fill:#fff4e1
```

### GRU 三大組件

#### 1. 重置門 (Reset Gate) - 紅色
```
r(t) = σ(Wr · [h(t-1), x(t)])
```
- **作用**: 決定要忘記多少過去的信息
- **範圍**: 0 到 1 (sigmoid 激活)
- **r ≈ 0**: 忽略過去狀態
- **r ≈ 1**: 保留過去狀態

#### 2. 更新門 (Update Gate) - 藍色
```
z(t) = σ(Wz · [h(t-1), x(t)])
```
- **作用**: 決定要保留多少舊狀態、接受多少新狀態
- **範圍**: 0 到 1 (sigmoid 激活)
- **z ≈ 0**: 更新狀態 (接受新信息)
- **z ≈ 1**: 保持狀態 (忽略新信息)

#### 3. 候選狀態 (Candidate State) - 綠色
```
h̃(t) = tanh(Wh · [r(t) ⊙ h(t-1), x(t)])
```
- **作用**: 計算候選的新狀態
- **範圍**: -1 到 1 (tanh 激活)
- **使用重置門**: 控制過去信息的影響

### 最終輸出
```
h(t) = (1 - z(t)) ⊙ h̃(t) + z(t) ⊙ h(t-1)
```
- **線性插值**: 在新狀態和舊狀態之間取平衡
- **更新門控制**: z 決定新舊狀態的比例

### GRU vs LSTM
| 特性 | GRU | LSTM |
|------|-----|------|
| 門的數量 | 2 個 (重置、更新) | 3 個 (輸入、輸出、遺忘) |
| 參數量 | 較少 | 較多 |
| 訓練速度 | 較快 | 較慢 |
| 記憶能力 | 適中 | 較強 |
| 適用場景 | 中短序列 | 長序列 |

## NLP Text Preprocessing Notes

### LLM 文字前處理步驟

1. **Tokenization (分詞)**
   - 將文本切分成單詞或子詞單元
   - 處理標點符號和特殊字符

2. **Normalization (正規化)**
   - 轉換為小寫
   - 移除或統一標點符號
   - 處理數字和特殊符號

3. **Stop Words Removal (停用詞移除)**
   - 移除常見但無意義的詞彙（如：the, a, is）

4. **Stemming/Lemmatization (詞幹提取/詞形還原)**
   - 將詞彙還原到基本形式

5. **建立詞彙表 (Vocabulary Building)**
   - 統計所有訓練數據中的詞彙
   - 依頻率排序，選取最常見的 N 個詞
   - 為每個詞分配唯一的整數索引 (token ID)
   - 保留特殊 token（如：`[PAD]`, `[UNK]`, `[START]`, `[END]`）

---

### target_vectorization.adapt() 方法

**作用**: 在訓練前分析文本數據，建立詞彙表（執行上述步驟 5）

```python
target_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",  # 可選: "int", "multi_hot", "count", "tf_idf"
    output_sequence_length=sequence_length
)

# 重要：使用 adapt() 學習詞彙表
target_vectorization.adapt(train_text_data)
```

**adapt() 做了什麼:**
1. 掃描所有訓練數據
2. 統計詞彙出現頻率
3. 建立詞彙表（依頻率排序，保留前 N 個最常見的詞）
4. 為每個詞分配唯一的整數索引

**注意事項:**
- 必須在訓練前執行
- 只能用訓練數據進行 adapt（避免數據洩漏）
- 驗證集和測試集使用相同的詞彙表

---

### TextVectorization 的 output_mode 參數

詞彙表建立完成後，**`output_mode` 參數**決定了如何將文本轉換為數值：

#### 1. `output_mode="int"` (整數序列)
- **用途**: 將文本轉換為整數序列，保留詞彙順序
- **輸出**: `[3, 15, 8, 42, ...]` (每個數字代表詞彙表中的索引)
- **適合**: 序列模型 (RNN, LSTM, Transformer)
- **特點**: 保留時間順序信息

#### 2. `output_mode="multi_hot"` (Multi-Hot 編碼)
- **用途**: 只記錄出現過的單字（不統計次數）
- **輸出**: `[0, 1, 0, 1, 1, 0, ...]` (二進制向量，1 表示該詞出現過)
- **適合**: 簡單分類任務
- **特點**: 忽略詞頻和順序

#### 3. `output_mode="count"` (詞頻統計)
- **用途**: 統計每個詞出現的次數
- **輸出**: `[0, 3, 0, 1, 2, 0, ...]` (每個位置的數字表示該詞出現次數)
- **適合**: 傳統機器學習模型 (如 Naive Bayes, SVM)
- **特點**: Bag of Words 的標準實作

#### 4. `output_mode="tf_idf"` (TF-IDF 權重)
- **用途**: 計算詞彙的 TF-IDF 重要程度
- **輸出**: `[0, 0.23, 0, 0.89, 0.15, ...]` (浮點數表示詞的重要性)
- **適合**: 文本分類、信息檢索
- **特點**: 過濾常見詞，強調重要詞彙

---

### Bag of Words (詞袋模型)

**基本概念：**
- 記錄每個詞在文檔中出現的次數
- 忽略詞彙的順序，只關注出現頻率
- 將文本轉換為數值向量
- 對應 `output_mode="count"` 或 `"multi_hot"`

**詞彙重要程度計算 (TF-IDF):**
- **TF (Term Frequency)**: 詞在單篇文檔中的出現次數
- **IDF (Inverse Document Frequency)**: `log(總文檔數 / 包含該詞的文檔數)`
- **TF-IDF**: `TF × IDF`
  - 衡量詞彙在文檔中的重要性
  - 公式: 詞在文章中的重要性 = (詞在該文章的出現次數) × log(總文檔數 / 包含該詞的文檔數)
  - 過濾掉過於常見或過於罕見的詞
- 對應 `output_mode="tf_idf"`

---

### 模型選擇準則：序列模型 vs 詞袋模型

根據數據特性選擇合適的模型：

**使用序列模型 (RNN/LSTM/Transformer) 當:**
- **樣本數量 / 樣本平均長度 > 1500**
- 詞彙順序很重要（如情感分析、翻譯）
- 需要捕捉上下文依賴關係

**使用詞袋模型 (Bag of Words) 當:**
- **樣本數量 / 樣本平均長度 < 1500**
- 數據量較小
- 詞彙出現與否比順序更重要（如垃圾郵件分類）
- 需要快速訓練和推理

**判斷公式:**
```
如果 (總樣本數 / 平均文本長度) > 1500:
    使用序列模型 (output_mode="int")
否則:
    使用詞袋模型 (output_mode="count" 或 "tf_idf")
```

---

## 專案檔案

- `seq2seq_gru_translation.py`: Seq2seq GRU 翻譯模型訓練腳本
- `spa-eng/`: 英文-西班牙文平行語料庫資料夾
- `spa-eng/spa.txt`: 訓練資料 (格式: English\tSpanish)

## 使用方式

```bash
# GRU 版本
python seq2seq_gru_translation.py

# Transformer 版本（待實作）
# python seq2seq_transformer_translation.py
```

## 相依套件

- tensorflow/keras
- Python 3.x

## 資料集格式

```
Hello.\t¡Hola!
How are you?\t¿Cómo estás?
Good morning.\tBuenos días.
```

每行包含英文和西班牙文，以 tab (`\t`) 分隔。
