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

### Bag of Words (詞袋模型)

**基本概念：**
- 記錄每個詞在文檔中出現的次數
- 忽略詞彙的順序，只關注出現頻率
- 將文本轉換為數值向量

**詞彙重要程度計算：**
- **Term Frequency (TF)**: 詞在單篇文檔中的出現次數
- **Inverse Document Frequency (IDF)**: log(總文檔數 / 包含該詞的文檔數)
- **TF-IDF**: TF × IDF
  - 衡量詞彙在文檔中的重要性
  - 公式: `詞在文章中的出現次數 / 詞在所有文章中的出現次數`
  - 過濾掉過於常見或過於罕見的詞

**時間序列的考量：**
- 加入時間判斷和順序信息
- 考慮詞彙的上下文關係
- 使用 RNN/LSTM 模型捕捉序列依賴性

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
