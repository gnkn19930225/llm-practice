# LSTM 架構詳解

## `layers.LSTM(32, recurrent_dropout=0.25)(input)` 中的 32 是什麼？

**32 是 LSTM 單元的數量（units）**，也就是隱藏狀態（hidden state）的維度。

- 輸入：shape = `(batch_size, 120, 14)` - 120個時間步，每步14個特徵
- LSTM處理後輸出：shape = `(batch_size, 32)` - 32維的特徵向量
- `recurrent_dropout=0.25`：在循環連接上應用25%的dropout防止過擬合

---

## 重要概念：輸入數據的時間區間與預測目標

### `(batch_size, 120, 14)` 的詳細解釋

```mermaid
graph TB
    subgraph "batch_size = 256"
        direction TB
        S1["樣本1"]
        S2["樣本2"]
        S3["..."]
        S256["樣本256"]
    end

    subgraph "每個樣本的結構 (120, 14)"
        direction TB
        T1["時間步1：過去119小時<br/>14個氣象特徵"]
        T2["時間步2：過去118小時<br/>14個氣象特徵"]
        T3["..."]
        T119["時間步119：過去1小時<br/>14個氣象特徵"]
        T120["時間步120：當前時刻<br/>14個氣象特徵"]

        T1 --> T2 --> T3 --> T119 --> T120
    end

    subgraph "對應的預測目標"
        P["未來24小時後<br/>1個溫度值"]
    end

    S1 -.例如.-> T1
    T120 -.預測.-> P

    style T120 fill:#ffcccc
    style P fill:#ccffff
```

### 時間步（Timestep）的定義

**時間步（Timestep）** 是時間序列中的基本時間單位。在這個項目中：

```python
# 原始數據：每10分鐘記錄一次
CSV記錄頻率 = 10分鐘/筆

# 代碼第34行
sampling_rate = 6  # 每6筆原始數據取1筆

# 計算一個時間步的實際時間
1 timestep = 6 × 10分鐘 = 60分鐘 = 1小時
```

**✅ 所以在這個項目中，1個時間步 = 1小時**

但要注意：時間步的長度取決於 `sampling_rate`，不是固定的！

| sampling_rate | 1個時間步的實際時間 |
|---------------|-------------------|
| 1 | 10分鐘 |
| 6 | 1小時（你的代碼） |
| 12 | 2小時 |
| 144 | 24小時（1天） |

### 維度詳解

| 維度 | 數值 | 意義 | 如何得到 |
|------|------|------|----------|
| **batch_size** | 256 | 一次訓練的樣本數量 | 代碼第37行：`batch_size = 256` |
| **120** | **時間步數（Timesteps）** | **過去120個時間步的歷史數據**<br/>= 過去120小時 | 代碼第35行：`sequence_length = 120` |
| **14** | **特徵數（Features）** | 每個時間點的氣象特徵 | 代碼第97行：`raw_data.shape[-1]`<br/>= CSV總列數15 - 時間列1 = 14 |

**14個特徵詳細列表**（從CSV的第2~15列）：
1. p (mbar) - 氣壓
2. T (degC) - 溫度
3. Tpot (K) - 位溫
4. Tdew (degC) - 露點溫度
5. rh (%) - 相對濕度
6. VPmax (mbar) - 飽和水汽壓
7. VPact (mbar) - 實際水汽壓
8. VPdef (mbar) - 水汽壓差
9. sh (g/kg) - 比濕
10. H2OC (mmol/mol) - 水汽濃度
11. rho (g/m³) - 空氣密度
12. wv (m/s) - 風速
13. max. wv (m/s) - 最大風速
14. wd (deg) - 風向

### ⚠️ 常見誤解

❌ **錯誤**：120 代表預測未來120個時間點
✅ **正確**：120 代表輸入**過去**120個時間點的歷史數據

❌ **錯誤**：輸出是未來120小時的溫度序列
✅ **正確**：輸出是**未來24小時後**的**單一溫度值**

### 具體數據示例

假設原始數據每10分鐘記錄一次，`sampling_rate=6` 表示每小時取一個樣本：

```
原始數據採樣頻率：10分鐘/次
sampling_rate = 6  →  每6筆取1筆  →  1小時/次
sequence_length = 120  →  120小時的歷史
```

#### 訓練樣本示例

```mermaid
timeline
    title 單個訓練樣本的時間軸
    section 輸入數據（過去）
        時間 t-119 : 2016/01/01 00:00
                   : 壓力=996.5
                   : 溫度=-8.0°C
                   : 濕度=93%
                   : ...（共14個特徵）
        時間 t-118 : 2016/01/01 01:00
        ... : ...
        時間 t-1 : 2016/01/05 22:00
        時間 t : 2016/01/05 23:00
               : （當前時刻）
    section 預測目標（未來）
        時間 t+24 : 2016/01/06 23:00
                  : 預測溫度=?°C
                  : （24小時後）
```

### 代碼中的關鍵參數

```python
# Untitled-1.py 第16行
raw_data = np.zeros((len(lines), len(header) - 1))
#                                 ^^^^^^^^^^^^^^
#                                 CSV有15列，減去第1列（時間）= 14個特徵

# Untitled-1.py 第34-37行
sampling_rate = 6              # 每6筆原始數據取1筆 = 1小時
sequence_length = 120          # 輸入120個時間步 = 過去120小時
delay = sampling_rate * (sequence_length + 24 - 1)
#       = 6 × (120 + 24 - 1)
#       = 6 × 143 = 858
# 意義：目標值是輸入序列最後一個時間點之後的24小時
batch_size = 256               # 一批訓練256個樣本

# Untitled-1.py 第97行
input = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
#                          ^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^
#                          120個時間步       14個特徵
#                          shape = (120, 14)
```

### 數據配對關係

```mermaid
graph LR
    subgraph "訓練樣本構建"
        direction TB

        subgraph "樣本1"
            I1["輸入：第0~119小時<br/>shape: (120, 14)"]
            O1["目標：第143小時溫度<br/>shape: (1)"]
        end

        subgraph "樣本2"
            I2["輸入：第1~120小時<br/>shape: (120, 14)"]
            O2["目標：第144小時溫度<br/>shape: (1)"]
        end

        subgraph "樣本3"
            I3["輸入：第2~121小時<br/>shape: (120, 14)"]
            O3["目標：第145小時溫度<br/>shape: (1)"]
        end
    end

    I1 -.24小時後.-> O1
    I2 -.24小時後.-> O2
    I3 -.24小時後.-> O3

    style O1 fill:#ffe6e6
    style O2 fill:#ffe6e6
    style O3 fill:#ffe6e6
```

### 預測結果詳解

**問：預測結果是一個時間點的資料嗎？**

**答：是的！** 預測結果是**單一時間點的單一數值**：

```python
# 模型輸出
output = layers.Dense(1)(x)  # 輸出層只有1個神經元

# 輸出shape
(batch_size, 1)  # 每個樣本預測1個溫度值

# 具體例子
樣本1：輸入過去120小時 → 預測 5.2°C（24小時後）
樣本2：輸入過去120小時 → 預測 5.5°C（24小時後）
...
樣本256：輸入過去120小時 → 預測 4.8°C（24小時後）
```

### 為什麼是24小時後？

```python
# Untitled-1.py 第36行
delay = sampling_rate * (sequence_length + 24 - 1)
#                                          ^^^
#                                   這個 24 決定預測多遠的未來

# 如果改成：
delay = sampling_rate * (sequence_length + 48 - 1)  # 預測48小時後
delay = sampling_rate * (sequence_length + 12 - 1)  # 預測12小時後
```

### 完整的輸入輸出流程

```mermaid
flowchart TD
    A["原始CSV數據<br/>420,551筆記錄<br/>每10分鐘一筆"] --> B["採樣<br/>sampling_rate=6<br/>每6筆取1筆"]

    B --> C["70,092筆數據<br/>每小時一筆"]

    C --> D["切分數據集<br/>訓練50% / 驗證25% / 測試25%"]

    D --> E["創建時序數據集<br/>sequence_length=120<br/>delay=24小時"]

    E --> F["單個訓練樣本"]

    F --> G["輸入X<br/>(120, 14)<br/>過去120小時的14個特徵"]

    F --> H["目標y<br/>(1,)<br/>24小時後的溫度"]

    G --> I["LSTM模型"]
    H --> I

    I --> J["訓練後預測<br/>給定過去120小時<br/>輸出未來24小時後溫度"]

    style G fill:#e1f5ff
    style H fill:#ffe1e1
    style J fill:#e1ffe1
```

### 總結：時間區間完整說明

| 概念 | 數值 | 實際意義 |
|------|------|----------|
| **輸入時間跨度** | 120小時 | 過去5天的歷史數據 |
| **輸入時間點** | t-119 到 t | 從119小時前到當前時刻 |
| **預測時間點** | t+24 | 當前時刻的24小時後 |
| **預測數量** | 1個數值 | 只預測1個溫度值 |
| **預測類型** | 點預測 | 不是序列預測 |

**關鍵理解**：
- 模型看的是**過去** → 120個時間步
- 模型預測的是**未來** → 1個時間點
- 時間間隔 → 24小時

如果要預測未來多個時間點（如未來24小時的逐時溫度），需要改用**序列到序列（Seq2Seq）架構**，輸出層改為 `return_sequences=True` 的LSTM或其他架構。

## LSTM 內部結構

LSTM（Long Short-Term Memory）有4個主要門控機制：

```mermaid
graph TB
    subgraph "LSTM Cell at time t"
        Input[輸入 x_t<br/>14個特徵] --> FG[遺忘門<br/>Forget Gate]
        Input --> IG[輸入門<br/>Input Gate]
        Input --> OG[輸出門<br/>Output Gate]
        Input --> CG[候選記憶<br/>Candidate]

        PrevH[前一隱藏狀態<br/>h_t-1<br/>32維] --> FG
        PrevH --> IG
        PrevH --> OG
        PrevH --> CG

        PrevC[前一細胞狀態<br/>C_t-1<br/>32維] --> FG

        FG --> Mult1[⊗ 遺忘]
        IG --> Mult2[⊗ 記住新資訊]
        CG --> Mult2

        Mult1 --> Add[⊕ 更新細胞狀態]
        Mult2 --> Add

        Add --> NewC[新細胞狀態<br/>C_t<br/>32維]
        NewC --> Tanh[tanh]
        Tanh --> Mult3[⊗]
        OG --> Mult3

        Mult3 --> NewH[新隱藏狀態<br/>h_t<br/>32維]

        NewC -.下一時間步.-> PrevC
        NewH -.下一時間步.-> PrevH
    end

    style FG fill:#ffcccc
    style IG fill:#ccffcc
    style OG fill:#ccccff
    style CG fill:#ffffcc
    style NewH fill:#ff9999
    style NewC fill:#99ff99
```

## LSTM 四個門的功能

### 1. 遺忘門（Forget Gate）
```
f_t = σ(W_f · [h_t-1, x_t] + b_f)
```
- 決定從細胞狀態中丨棄多少舊資訊
- 輸出 0~1 之間的值（0=完全遺忘，1=完全保留）

### 2. 輸入門（Input Gate）
```
i_t = σ(W_i · [h_t-1, x_t] + b_i)
C̃_t = tanh(W_C · [h_t-1, x_t] + b_C)
```
- 決定要將多少新資訊存入細胞狀態
- `i_t`：控制門（0~1）
- `C̃_t`：候選值（-1~1）

### 3. 細胞狀態更新
```
C_t = f_t ⊗ C_t-1 + i_t ⊗ C̃_t
```
- 遺忘舊資訊 + 加入新資訊

### 4. 輸出門（Output Gate）
```
o_t = σ(W_o · [h_t-1, x_t] + b_o)
h_t = o_t ⊗ tanh(C_t)
```
- 決定輸出什麼資訊
- `h_t` 就是 LSTM 的輸出（32維）

## 完整模型資料流程

```mermaid
graph LR
    subgraph "輸入資料"
        A[120個時間步<br/>每步14個特徵<br/>shape: 120×14]
    end

    subgraph "LSTM層"
        B[LSTM-1<br/>處理時間步1]
        C[LSTM-2<br/>處理時間步2]
        D[...]
        E[LSTM-120<br/>處理時間步120]

        B -->|h_1, C_1| C
        C -->|h_2, C_2| D
        D -->|...| E
    end

    subgraph "輸出"
        F[最後隱藏狀態<br/>h_120<br/>shape: 32]
    end

    subgraph "後續層"
        G[Dropout 50%<br/>shape: 32]
        H[Dense輸出層<br/>shape: 1]
    end

    A --> B
    E --> F
    F --> G
    G --> H
    H --> I[溫度預測值]

    style A fill:#e1f5ff
    style F fill:#ffe1e1
    style I fill:#e1ffe1
```

## 為什麼選擇 32 個單元？

這是一個**超參數**，需要根據任務調整：

- **太小**（如8）：模型容量不足，無法捕捉複雜的時間模式
- **太大**（如256）：容易過擬合，訓練慢，需要更多數據
- **32**：對於這個氣候預測任務是一個合理的平衡點

## 參數數量計算

LSTM 的參數數量公式：
```
參數總數 = 4 × (units × (units + input_dim + 1))
         = 4 × (32 × (32 + 14 + 1))
         = 4 × (32 × 47)
         = 6,016 個參數
```

其中 4 代表四個門（遺忘、輸入、輸出、候選）

## 關鍵概念總結

```mermaid
mindmap
  root((LSTM))
    32個單元
      隱藏狀態維度
      細胞狀態維度
      模型容量
    輸入
      120時間步
      14個特徵
      批次資料
    輸出
      32維向量
      最後時間步
    四個門
      遺忘門
      輸入門
      輸出門
      候選記憶
    優勢
      記住長期依賴
      避免梯度消失
      選擇性記憶
```

## 與普通RNN的區別

| 特性 | RNN | LSTM |
|------|-----|------|
| 記憶機制 | 只有隱藏狀態 h_t | 細胞狀態 C_t + 隱藏狀態 h_t |
| 長期依賴 | 容易遺忘（梯度消失） | 可以記住長期資訊 |
| 門控機制 | 無 | 3個門控制資訊流動 |
| 參數量 | 少 | 多（4倍） |

LSTM 通過細胞狀態這條"高速公路"，讓資訊可以在很長的時間序列中傳遞而不衰減！
