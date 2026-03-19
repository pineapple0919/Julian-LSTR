# RE-LSTR: Robust & Efficient Lane Shape Prediction with Transformers

本專案基於 **LSTR** 進行改良，導入了 **FasterNet** 輕量化骨幹、**Stable Diffusion** 資料增強（SPDA）以及運算管線（Pipeline）優化，旨在提升台灣複雜道路環境下的偵測精度與硬體運算效率。

## 2) 程式碼結構（重要檔案）

在使用前請確保已安裝相關環境並啟用：
```text
.
├── config.py                    # 全域預設設定類別（system_configs）
├── config/
│   ├── LSTR.json                # TuSimple 設定
│   └── LSTR_CULANE.json         # CULane 設定
├── train.py                     # 訓練主程式
├── test.py                      # 測試與推論主程式
├── evaluate.py / eva.py / eva2.py  # 評估相關腳本
├── nnet/
│   └── py_factory.py            # NetworkFactory: 組模型、loss、optimizer
├── models/
│   ├── LSTR.py                  # LSTR 模型入口（model/loss）
│   ├── LSTR_CULANE.py           # CULane 版本模型入口
│   ├── FasterNet.py             # 骨幹/模組實作（另有 py_utils/FasterNet.py）
│   └── py_utils/                # Transformer、matcher、loss、parallel 等核心元件
├── db/
│   ├── datasets.py              # 資料集名稱到類別映射
│   ├── tusimple.py              # TuSimple 讀取與標註轉換
│   ├── culane.py                # CULane 讀取與標註轉換
│   └── utils/                   # metric/evaluator/可視化工具
├── sample/
│   ├── tusimple.py              # TuSimple 訓練取樣邏輯
│   └── culane.py                # CULane 訓練取樣邏輯
├── test/
│   ├── tusimple.py              # TuSimple 測試流程
│   ├── culane.py                # CULane 測試流程
│   └── images.py                # 單張/資料夾影像推論
└── lane_evaluation/             # C++ + Python 的 lane 評估工具
```

---

## 🚀 環境準備

在使用前請確保已安裝相關環境並啟用：

```bash
# 啟用 Conda 環境
conda activate lstr

# 檢查系統資源負載 (CPU / GPU)
htop
nvidia-smi

```

---

## 🏋️ 訓練 (Training)

啟動模型訓練，預設會讀取 `config/LSTR.json`。

```bash
python train.py LSTR
python train.py LSTR --iter 70000

```

> **注意**：本專案已優化資料處理順序，將擬合運算提前至預處理階段，以解決 GPU 等待 CPU 的效能瓶頸。

---

## 🔍 測試與評估 (Testing & Evaluation)

### 1. 跑數據與性能評估 (TuSimple 格式)

針對測試集進行評估並輸出準確率指標：

```bash
# 跑數據 (指定 iteration 與配置文件)
python test.py LSTR --testiter 70000 --split testing --modality eval
python test.py LSTR --testiter 307500 --split testing --modality eval
```

### 2. 可視化測試結果

可視化預測的車道線並顯示參數（$k, f, m \dots$）：

```bash
# 基本可視化
python test.py LSTR --testiter 190000 --split testing --modality eval

# 儲存偵測結果圖至 ./results/LSTR/k/testing/lane_debug
python test.py LSTR --testiter 190000 --modality eval --split testing --debug

```

### 3. 對自定義影像進行偵測

將您想測試的照片放入 `./images` 資料夾中，結果將輸出至 `./detections`：

```bash
python test.py LSTR --testiter 500000 --modality images --image_root ./ --debug

```

---

## 🛠️ 技術貢獻 (RE-LSTR 亮點)

* **SPDA 資料增強**：整合 Stable Diffusion 生成模擬台灣雨天、大太陽、夜間等 10 種環境，解決資料分佈不均問題。
* **FasterNet Backbone**：使用 PConv 優化特徵提取，顯著降低 GPU 運算負擔並提升推論速度。
* **Pipeline 優化**：重組資料處理順序，改善原本 CPU 滿載而 GPU 利用率低的問題。
* **幾何精度改善**：優化曲線擬合參數調整，提升遠景地平線處的車道擬合精確度。

---

**您是否需要我幫您補充關於 FasterNet 的具體安裝步驟或是環境依賴表（requirements.txt）？**