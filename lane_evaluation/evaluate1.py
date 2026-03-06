import numpy as np
import os
import glob
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import interp1d


# 資料路徑
pred_path = "../results/LSTR_CULANE/500000/testing"  # 模型預測結果
gt_base_path = "../../../CULane"  # 真值車道標註根目錄

# 工具函式：讀取標註資料
def read_lane_points(file_path):
    """
    讀取車道點座標，CUlane的標註通常以行為單位存儲。
    """
    lane_points = []
    with open(file_path, 'r') as f:
        for line in f:
            points = line.strip().split()
            # 改為支持浮點數
            points = [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]
            lane_points.append(points)
    return lane_points

# 工具函式：回歸擬合
def fit_lane(points):
    """
    對輸入車道點進行線性回歸擬合。
    返回擬合的線性模型和擬合的y值。
    """
    if len(points) < 2:
        return None, None  # 資料不足無法擬合
    points = np.array(points)
    X = points[:, 0].reshape(-1, 1)  # x座標
    y = points[:, 1]  # y座標
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred

def interpolate_points(points, target_num=100):
    """
    將車道點插值到固定數量。
    :param points: 車道點 [(x1, y1), (x2, y2), ...]
    :param target_num: 目標點數
    :return: 插值後的車道點 [(x', y'), ...]
    """
    if len(points) < 2:
        return None  # 資料不足無法插值
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]
    
    # 使用線性插值將點數固定為 target_num
    interp_func_x = interp1d(np.linspace(0, 1, len(x)), x, kind='linear')
    interp_func_y = interp1d(np.linspace(0, 1, len(y)), y, kind='linear')
    new_x = interp_func_x(np.linspace(0, 1, target_num))
    new_y = interp_func_y(np.linspace(0, 1, target_num))
    return np.column_stack((new_x, new_y))

# 評估函式
def evaluate_lane(gt_points, pred_points):
    """
    評估真值車道與預測車道之間的差異。
    """
    if len(gt_points) < 2 or len(pred_points) < 2:
        return None  # 資料不足無法評估
    
    # 將車道點插值到相同數量
    gt_points = interpolate_points(gt_points)
    pred_points = interpolate_points(pred_points)
    
    if gt_points is None or pred_points is None:
        return None  # 插值失敗
    
    mse = mean_squared_error(gt_points[:, 1], pred_points[:, 1])
    r2 = r2_score(gt_points[:, 1], pred_points[:, 1])
    return mse, r2

# 主流程
def main():
    # 搜尋所有預測檔案
    pred_files = glob.glob(os.path.join(pred_path, "*/*/*.txt"))
    
    total_mse, total_r2 = 0, 0
    count = 0
    
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
        
        for gt_points, pred_points in zip(gt_lanes, pred_lanes):
            # 對每條車道擬合並計算評估指標
            _, gt_y_pred = fit_lane(gt_points)
            _, pred_y_pred = fit_lane(pred_points)
            
            if gt_y_pred is not None and pred_y_pred is not None:
                mse, r2 = evaluate_lane(gt_points, pred_points)
                total_mse += mse
                total_r2 += r2
                count += 1
    
    # 輸出總體評估結果
    print(f"平均 MSE: {total_mse / count if count > 0 else '無法計算'}")
    print(f"平均 R2: {total_r2 / count if count > 0 else '無法計算'}")

if __name__ == "__main__":
    main()