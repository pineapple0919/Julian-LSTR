# RE-LSTR: Robust & Efficient Lane Shape Prediction with Transformers

本專案基於 **LSTR** 進行改良，導入了 **FasterNet** 輕量化骨幹、**Stable Diffusion** 資料增強（SPDA）以及運算管線（Pipeline）優化，旨在提升台灣複雜道路環境下的偵測精度與硬體運算效率。

## 📂 專案目錄結構

```Plaintext
.
├── cache/              # 訓練快照 (.pkl) 與資料集快取檔案
├── config/             # 系統與資料庫配置檔 (JSON)
│   └── LSTR.json       # 核心參數設定（如權重、層數、學習率）
├── db/                 # 資料集加載與標籤處理邏輯
│   ├── datasets.py     # 資料集註冊入口
│   ├── detection.py    # 基礎偵測類別
│   └── tusimple.py     # TuSimple 資料集專用處理（含標記轉參數邏輯）
├── models/             # 神經網路架構定義
│   ├── LSTR.py         # 模型實例化入口與參數初始化
│   └── py_utils/       # 核心組件
│       ├── kp.py       # 主架構定義（含 Backbone 與 Transformer 交互）
│       ├── fasternet.py# (新增) 改良型輕量化 PConv 骨幹網路
│       └── transformer.py# Transformer 編解碼器實作
├── sample/             # 訓練採樣與資料增強
│   └── tusimple.py     # 訓練時的動態採樣與標籤預處理
├── images/             # 放置自定義測試影像
├── detections/         # 自定義影像的偵測結果輸出
├── results/            # 訓練過程中的可視化除錯圖
├── train.py            # 訓練啟動腳本
└── test.py             # 測試、評估與推論腳本

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

```

> **注意**：本專案已優化資料處理順序，將擬合運算提前至預處理階段，以解決 GPU 等待 CPU 的效能瓶頸。

---

## 🔍 測試與評估 (Testing & Evaluation)

### 1. 跑數據與性能評估 (TuSimple 格式)

針對測試集進行評估並輸出準確率指標：

```bash
# 跑數據 (指定 iteration 與配置文件)
python test.py LSTR --testiter 200000 --split testing --modality eval cfg_file: ./config/LSTR.json

```

### 2. 可視化測試結果

可視化預測的車道線並顯示參數（$k, f, m \dots$）：

```bash
# 基本可視化
python test.py LSTR --testiter 200000 --split testing --modality eval

# 儲存偵測結果圖至 ./results/LSTR/500000/testing/lane_debug
python test.py LSTR --testiter 500000 --modality eval --split testing --debug

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