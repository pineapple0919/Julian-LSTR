import numpy as np
import os
import glob
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist


# 資料路徑
pred_path = "./results/LSTR_CULANE/500000/testing"  # 模型預測結果
gt_base_path = "/home/lab602/LSTRproject/CULane"  # 真值車道標註根目錄

# 工具函式：讀取標註資料
def read_lane_points(file_path):
    """
    讀取車道點座標，CUlane的標註通常以行為單位存儲。
    """
    lane_points = []
    with open(file_path, 'r') as f:
        for line in f:
            points = line.strip().split()
            points = [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]
            lane_points.append(points)
    return lane_points

# 工具函式：計算兩條車道之間的平均距離
def calculate_average_distance(gt_points, pred_points):
    """
    計算真值車道與預測車道之間的平均距離。
    """
    gt_points = np.array(gt_points)
    pred_points = np.array(pred_points)
    
    # 使用 cdist 計算每個點與所有點的歐式距離
    distances = cdist(gt_points, pred_points, metric='euclidean')
    
    # 將每個點的最小距離作為對應點之間的距離
    min_distances = np.min(distances, axis=1)
    
    # 返回平均距離
    return np.mean(min_distances)

# 主流程
def main():
    # 搜尋所有預測檔案
    pred_files = glob.glob(os.path.join(pred_path, "*/*/*.txt"))
    
    correct_count = 0
    total_pred_count = 0
    total_gt_count = 0
    false_positive_count = 0
    false_negative_count = 0
    
    for pred_file in sorted(pred_files):
        # 從預測文件路徑推導對應的真值文件路徑
        relative_path = os.path.relpath(pred_file, pred_path)  # 相對於pred_path的路徑
        gt_file = os.path.join(gt_base_path, relative_path)  # 在gt_base_path中尋找對應路徑
        
        if not os.path.exists(gt_file):
            print(f"找不到對應的真值文件: {gt_file}")
            continue
        
        # 讀取真值與預測的車道點
        gt_lanes = read_lane_points(gt_file)
        pred_lanes = read_lane_points(pred_file)
        
        # 更新總數量
        total_gt_count += len(gt_lanes)
        total_pred_count += len(pred_lanes)
        
        # 追蹤每條預測線和真值線是否匹配
        gt_matched = [False] * len(gt_lanes)
        pred_matched = [False] * len(pred_lanes)
        
        # 比對每條預測線與真值線
        for i, pred_points in enumerate(pred_lanes):
            for j, gt_points in enumerate(gt_lanes):
                avg_distance = calculate_average_distance(gt_points, pred_points)
                if avg_distance < 50:
                    pred_matched[i] = True
                    gt_matched[j] = True
                    break  # 找到匹配後停止檢查其他線
        
        # 計算正確預測數量
        correct_count += sum(pred_matched)
        
        # 計算 False Positives 和 False Negatives
        false_positive_count += sum(1 for matched in pred_matched if not matched)
        false_negative_count += sum(1 for matched in gt_matched if not matched)
    
    # 計算準確率
    accuracy = (correct_count / total_pred_count) * 100 if total_pred_count > 0 else 0
    fp_percentage = (false_positive_count / total_pred_count) * 100 if total_pred_count > 0 else 0
    fn_percentage = (false_negative_count / total_gt_count) * 100 if total_gt_count > 0 else 0
    
    print(f"總預測數量: {total_pred_count}")
    print(f"總真值數量: {total_gt_count}")
    print(f"正確預測數量: {correct_count}")
    print(f"準確率: {accuracy:.2f}%")
    print(f"False Positives (FP): {false_positive_count} ({fp_percentage:.2f}%)")
    print(f"False Negatives (FN): {false_negative_count} ({fn_percentage:.2f}%)")

if __name__ == "__main__":
    main()
