import os
import numpy as np
import pandas as pd
from glob import glob
from shapely.geometry import LineString
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm  
# ====== 基礎函式 ======
def read_lane_points(file_path):
    lane_points = []
    with open(file_path, 'r') as f:
        for line in f:
            points = line.strip().split()
            points = [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]
            if len(points) >= 2:
                lane_points.append(points)
    return lane_points

def interpolate_line(points, num=50):
    if len(points) < 2:
        return []
    points = np.array(points)
    xs, ys = points[:, 0], points[:, 1]
    interp_points = []
    for i in range(len(points) - 1):
        x_vals = np.linspace(xs[i], xs[i + 1], num=num)
        y_vals = np.linspace(ys[i], ys[i + 1], num=num)
        interp_points.extend(list(zip(x_vals, y_vals)))
    return interp_points

def create_lane_polygon(line_points, width=10):
    if len(line_points) < 2:
        return None
    line = LineString(line_points)
    poly = line.buffer(width / 2.0, cap_style=2, join_style=2)
    return poly

def compute_iou(poly_pred, poly_gt):
    if poly_pred is None or poly_gt is None:
        return 0.0
    inter = poly_pred.intersection(poly_gt).area
    union = poly_pred.union(poly_gt).area
    return inter / union if union > 0 else 0.0

# ====== 評估單張圖片 ======
def evaluate_lane_iou(gt_lanes, pred_lanes, iou_threshold=0.4, width=10):
    gt_polys = [create_lane_polygon(interpolate_line(l), width) for l in gt_lanes]
    pred_polys = [create_lane_polygon(interpolate_line(l), width) for l in pred_lanes]

    iou_matrix = np.zeros((len(gt_polys), len(pred_polys)), dtype=np.float32)
    for i, gt in enumerate(gt_polys):
        for j, pred in enumerate(pred_polys):
            iou_matrix[i, j] = compute_iou(pred, gt)

    cost_matrix = 1 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    tp = 0
    matched_gt = set()
    matched_pred = set()
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= iou_threshold:
            tp += 1
            matched_gt.add(i)
            matched_pred.add(j)

    fp = len(pred_lanes) - len(matched_pred)
    fn = len(gt_lanes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ====== 批次評估主程式 ======


def run_batch_iou_evaluation(pred_root, gt_root, iou_threshold=0.4, lane_width=30):
    pred_files = glob(os.path.join(pred_root, "*/*/*.txt"))
    total_tp, total_fp, total_fn = 0, 0, 0
    results_per_image = []

    for pred_file in tqdm(sorted(pred_files), desc="Evaluating lanes"):
        relative_path = os.path.relpath(pred_file, pred_root)
        gt_file = os.path.join(gt_root, relative_path)
        if not os.path.exists(gt_file):
            continue

        try:
            gt_lanes = read_lane_points(gt_file)
            pred_lanes = read_lane_points(pred_file)
        except Exception:
            continue

        result = evaluate_lane_iou(gt_lanes, pred_lanes, iou_threshold=iou_threshold, width=lane_width)
        result["image"] = relative_path
        results_per_image.append(result)

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    summary = {
        "Total TP": total_tp,
        "Total FP": total_fp,
        "Total FN": total_fn,
        "Overall Precision": precision,
        "Overall Recall": recall,
        "Overall F1-score": f1
    }

    summary_df = pd.DataFrame([summary])
    per_image_df = pd.DataFrame(results_per_image)
    return summary_df, per_image_df

# ====== 呼叫主程式 ======
if __name__ == "__main__":
    summary_df, _ = run_batch_iou_evaluation(
        pred_root="./results/LSTR_CULANE/500000/testing",
        gt_root="/home/lab602/LSTRproject/CULane",
        iou_threshold=0.4,
        lane_width=30
    )

    # 抓出數值
    result = summary_df.iloc[0].to_dict()
    total = result["Total TP"] + result["Total FP"] + result["Total FN"]

    # 計算比例（百分比）
    tp_pct = result["Total TP"] / total * 100 if total > 0 else 0
    fp_pct = result["Total FP"] / total * 100 if total > 0 else 0
    fn_pct = result["Total FN"] / total * 100 if total > 0 else 0

    # 換行逐項輸出
    print(f"TP: {tp_pct:.2f}%")
    print(f"FP: {fp_pct:.2f}%")
    print(f"FN: {fn_pct:.2f}%")
    print(f"Overall Precision: {result['Overall Precision']:.4f}")
    print(f"Overall Recall: {result['Overall Recall']:.4f}")
    print(f"Overall F1-score: {result['Overall F1-score']:.4f}")
