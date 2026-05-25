import numpy as np
import os
import glob
from scipy.spatial.distance import cdist

# 資料路徑
pred_path = "./results/LSTR/500000/testing/lane_debug"  # 模型預測結果
gt_base_path = "/home/lab602/LSTRproject/CULane"     # 真值標註根目錄

# 工具函式：讀取車道點座標
def read_lane_points(file_path):
    lane_points = []
    with open(file_path, 'r') as f:
        for line in f:
            points = line.strip().split()
            points = [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]
            lane_points.append(points)
    return lane_points

# 計算兩條車道的平均距離
def calculate_average_distance(gt_points, pred_points):
    gt_points = np.array(gt_points)
    pred_points = np.array(pred_points)
    if len(gt_points) == 0 or len(pred_points) == 0:
        return float('inf')
    distances = cdist(gt_points, pred_points, metric='euclidean')
    return np.mean(np.min(distances, axis=1))

# 主流程
def main():
    pred_files = glob.glob(os.path.join(pred_path, "*/*/*.txt"))
    
    if len(pred_files) == 0:
        print(f"❌ 錯誤：在 {pred_path} 找不到任何預測的 .txt 檔案！請檢查路徑。")
        return

    # 💡 教授建議：一次跑完 10, 30, 50 像素三個標準，直接拿去論文畫折線圖
    thresholds = [10, 30, 50] 

    print(f"🔍 開始評估... 找到的預測影像總數: {len(pred_files)}")
    
    for thresh in thresholds:
        correct_count = 0
        total_pred_count = 0
        total_gt_count = 0
        false_positive_count = 0
        false_negative_count = 0
        total_image_count = 0
        full_match_count = 0 

        for pred_file in sorted(pred_files):
            relative_path = os.path.relpath(pred_file, pred_path)
            gt_file = os.path.join(gt_base_path, relative_path)

            if not os.path.exists(gt_file):
                continue

            gt_lanes = read_lane_points(gt_file)
            pred_lanes = read_lane_points(pred_file)

            total_gt_count += len(gt_lanes)
            total_pred_count += len(pred_lanes)
            total_image_count += 1

            gt_matched = [False] * len(gt_lanes)
            pred_matched = [False] * len(pred_lanes)

            # 比對
            for i, pred_points in enumerate(pred_lanes):
                best_match_j = -1
                best_distance = float('inf')
                for j, gt_points in enumerate(gt_lanes):
                    if gt_matched[j]:
                        continue
                    avg_distance = calculate_average_distance(gt_points, pred_points)
                    # 💡 使用動態的閾值 thresh
                    if avg_distance < thresh and avg_distance < best_distance:
                        best_distance = avg_distance
                        best_match_j = j

                if best_match_j != -1:
                    pred_matched[i] = True
                    gt_matched[best_match_j] = True

            correct_count += sum(pred_matched)
            false_positive_count += sum(1 for matched in pred_matched if not matched)
            false_negative_count += sum(1 for matched in gt_matched if not matched)

            if all(gt_matched) and len(gt_matched) > 0:
                full_match_count += 1

        # 指標計算
        precision = (correct_count / total_pred_count) * 100 if total_pred_count > 0 else 0
        culane_accuracy = (full_match_count / total_image_count) * 100 if total_image_count > 0 else 0
        fp_percentage = (false_positive_count / total_pred_count) * 100 if total_pred_count > 0 else 0
        fn_percentage = (false_negative_count / total_gt_count) * 100 if total_gt_count > 0 else 0

        print(f"\n==================== 【像素容忍度門檻：{thresh} px】 ====================")
        print(f"正確配對數量 (TP) : {correct_count}  |  總預測數: {total_pred_count}  |  總真值數: {total_gt_count}")
        print(f"Precision (正確預測/預測總數)       : {precision:.2f}%")
        print(f"False Positives (誤報 FP 數量)      : {false_positive_count} ({fp_percentage:.2f}%)")
        print(f"False Negatives (漏報 FN 數量)      : {false_negative_count} ({fn_percentage:.2f}%)")
        print(f"CULane Accuracy (全圖完美配對率)    : {culane_accuracy:.2f}%")
        print("=========================================================================")


        
if __name__ == "__main__":
    main()
