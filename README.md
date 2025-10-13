# RNN Temperature Forecasting Project

This project uses LSTM recurrent neural networks for time series temperature prediction using the Jena Climate dataset (2009-2016).

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

## Project Files

- `Untitled-1.py`: LSTM model training script
- `jena_climate_2009_2016.csv`: Climate dataset
- `jena_dense.keras`: Trained model checkpoint

## Usage

```bash
python Untitled-1.py
```

## Dependencies

- tensorflow/keras
- numpy
- matplotlib
