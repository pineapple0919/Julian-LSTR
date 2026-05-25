import sys
import cv2
import numpy as np

# from lib.datasets.lane_dataset import LaneDataset

EXPS_DIR = 'experiments'


class Evaluator(object):
    def __init__(self, dataset, exp_dir, poly_degree=3):
        self.dataset = dataset
        # self.predictions = np.zeros((len(dataset.annotations), dataset.max_lanes, 4 + poly_degree))
        self.predictions = None
        self.runtimes = np.zeros(len(dataset))
        self.loss = np.zeros(len(dataset))
        self.exp_dir = exp_dir
        self.new_preds = False

    def add_prediction(self, idx, pred, runtime):
        if self.predictions is None:
            self.predictions = np.zeros((len(self.dataset._annotations), pred.shape[1], pred.shape[2]))
        self.predictions[idx, :pred.shape[1], :] = pred
        self.runtimes[idx] = runtime
        self.new_preds = True

    def eval(self, **kwargs):
        return self.dataset.eval(self.exp_dir, self.predictions, self.runtimes, **kwargs)


# if __name__ == "__main__":
#     evaluator = Evaluator(LaneDataset(split='test'), exp_dir=sys.argv[1])
#     evaluator.tusimple_eval()
def evaluate_culane_pure_python(pred_lanes, gt_lanes, img_shape=(590, 1640), line_width=30):
    """
    使用純 Python (NumPy/OpenCV) 模擬 CULane C++ 的像素級 IoU 評估機制
    pred_lanes: list of np.array, 每個 array 為該車道線的 (x, y) 坐標點集
    gt_lanes: list of np.array, 每個 array 為真實車道線的 (x, y) 坐標點集
    """
    if len(gt_lanes) == 0 and len(pred_lanes) == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0}
    if len(gt_lanes) == 0 and len(pred_lanes) > 0:
        return {'tp': 0, 'fp': len(pred_lanes), 'fn': 0}
    if len(gt_lanes) > 0 and len(pred_lanes) == 0:
        return {'tp': 0, 'fp': 0, 'fn': len(gt_lanes)}

    # 1. 為每條真實車道線生成個別的二值化遮罩
    gt_masks = []
    for gt_lane in gt_lanes:
        mask = np.zeros(img_shape, dtype=np.uint8)
        # 將坐標點連接成線段並加粗
        for i in range(len(gt_lane) - 1):
            p1 = (int(gt_lane[i][0]), int(gt_lane[i][1]))
            p2 = (int(gt_lane[i+1][0]), int(gt_lane[i+1][1]))
            cv2.line(mask, p1, p2, 1, thickness=line_width)
        gt_masks.append(mask)

    # 2. 為每條預測車道線生成個別的二值化遮罩
    pred_masks = []
    for pred_lane in pred_lanes:
        mask = np.zeros(img_shape, dtype=np.uint8)
        for i in range(len(pred_lane) - 1):
            p1 = (int(pred_lane[i][0]), int(pred_lane[i][1]))
            p2 = (int(pred_lane[i+1][0]), int(pred_lane[i+1][1]))
            cv2.line(mask, p1, p2, 1, thickness=line_width)
        pred_masks.append(mask)

    # 3. 計算 匈牙利匹配或貪婪匹配 的 IoU 矩陣
    num_gt = len(gt_masks)
    num_pred = len(pred_masks)
    iou_matrix = np.zeros((num_gt, num_pred))

    for g_idx, g_mask in enumerate(gt_masks):
        for p_idx, p_mask in enumerate(pred_masks):
            intersection = np.logical_and(g_mask, p_mask).sum()
            union = np.logical_or(g_mask, p_mask).sum()
            iou_matrix[g_idx, p_idx] = intersection / (union + 1e-6)

    # 4. 依據 IoU > 0.5 進行配對統計 (採用貪婪匹配模擬官方)
    tp, fp, fn = 0, 0, 0
    matched_gt = set()
    matched_pred = set()

    # 按 IoU 從大到小排序進行配對
    flat_indices = np.argsort(iou_matrix, axis=None)[::-1]
    for idx in flat_indices:
        g_idx, p_idx = np.unravel_index(idx, iou_matrix.shape)
        if iou_matrix[g_idx, p_idx] < 0.5:
            break
        if g_idx not in matched_gt and p_idx not in matched_pred:
            matched_gt.add(g_idx)
            matched_pred.add(p_idx)
            tp += 1

    fp = num_pred - len(matched_pred)
    fn = num_gt - len(matched_gt)

    return {'tp': tp, 'fp': fp, 'fn': fn}