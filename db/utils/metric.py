import argparse
from pprint import pprint

import cv2
import numpy as np
import ujson as json
from tqdm import tqdm
from tabulate import tabulate
from scipy.spatial import distance


def show_preds(pred, gt):
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    print(len(gt), 'gts and', len(pred), 'preds')
    for lane in gt:
        for p in lane:
            cv2.circle(img, tuple(map(int, p)), 5, thickness=-1, color=(255, 0, 255))
    for lane in pred:
        for p in lane:
            cv2.circle(img, tuple(map(int, p)), 4, thickness=-1, color=(0, 255, 0))
    cv2.imshow('img', img)
    cv2.waitKey(0)


def area_distance(pred_x, pred_y, gt_x, gt_y, placeholder=np.nan):
    pred = np.vstack([pred_x, pred_y]).T
    gt = np.vstack([gt_x, gt_y]).T

    # pred = pred[pred[:, 0] > 0][:3, :]
    # gt = gt[gt[:, 0] > 0][:5, :]

    dist_matrix = distance.cdist(pred, gt, metric='euclidean')

    dist = 0.5 * (np.min(dist_matrix, axis=0).sum() + np.min(dist_matrix, axis=1).sum())
    dist /= np.max(gt_y) - np.min(gt_y)
    return dist


# def area_metric(pred, gt, debug=None):
    pred = sorted(pred, key=lambda ps: abs(ps[0][0] - 720/2.))[:2]
    gt = sorted(gt, key=lambda ps: abs(ps[0][0] - 720/2.))[:2]
    if len(pred) == 0:
        return 0., 0., len(gt)
    line_dists = []
    fp = 0.
    matched = 0.
    gt_matches = [False] * len(gt)
    pred_matches = [False] * len(pred)
    pred_dists = [None] * len(pred)

    distances = np.ones((len(gt), len(pred)), dtype=np.float32)
    for i_gt, gt_points in enumerate(gt):
        x_gts = [x for x, _ in gt_points]
        y_gts = [y for _, y in gt_points]
        for i_pred, pred_points in enumerate(pred):
            x_preds = [x for x, _ in pred_points]
            y_preds = [y for _, y in pred_points]
            distances[i_gt, i_pred] = area_distance(x_preds, y_preds, x_gts, y_gts)

    best_preds = np.argmin(distances, axis=1)
    best_gts = np.argmin(distances, axis=0)
    fp = 0.
    fn = 0.
    dist = 0.
    is_fp = []
    is_fn = []
    for i_pred, best_gt in enumerate(best_gts):
        if best_preds[best_gt] == i_pred:
            dist += distances[best_gt, i_pred]
            is_fp.append(False)
        else:
            fp += 1
            is_fp.append(True)
    for i_gt, best_pred in enumerate(best_preds):
        if best_gts[best_pred] != i_gt:
            fn += 1
            is_fn.append(True)
        else:
            is_fn.append(False)
    if debug:
        print('is fp')
        print(is_fp)
        print('is fn')
        print(is_fn)
        print('distances')
        dists = np.min(distances, axis=0)
        dists[np.array(is_fp)] = 0
        print(dists)
        show_preds(pred, gt)

    return dist, fp, fn

def area_metric(pred, gt, debug=None, pixel_threshold=30):
    """
    擴充評估核心：引入 pixel_threshold (預設 30 像素)
    用以嚴格定義在特定像素偏差下，TP、FP、FN 的消融變化。
    """
    # 保持原有的排序與前置作業
    pred = sorted(pred, key=lambda ps: abs(ps[0][0] - 720/2.))[:2]
    gt = sorted(gt, key=lambda ps: abs(ps[0][0] - 720/2.))[:2]
    
    if len(pred) == 0:
        return 0., 0., len(gt), 0  # 增加一個回傳值代表 TP 數量

    distances = np.ones((len(gt), len(pred)), dtype=np.float32)
    for i_gt, gt_points in enumerate(gt):
        x_gts = [x for x, _ in gt_points]
        y_gts = [y for _, y in gt_points]
        for i_pred, pred_points in enumerate(pred):
            x_preds = [x for x, _ in pred_points]
            y_preds = [y for _, y in pred_points]
            distances[i_gt, i_pred] = area_distance(x_preds, y_preds, x_gts, y_gts)

    best_preds = np.argmin(distances, axis=1)
    best_gts = np.argmin(distances, axis=0)
    
    # 初始化計數器
    tp = 0
    fp = 0.
    fn = 0.
    dist = 0.
    is_fp = []
    is_fn = []
    
    # 建立一個基礎常數，用來將歸一化的 area_distance 還原成大約的像素距離
    # 或者直接將 threshold 作為判定條件
    for i_pred, best_gt in enumerate(best_gts):
        # 檢查是否互為最優匹配
        if best_preds[best_gt] == i_pred:
            # 【核心修改點】：引入門檻控制
            # 因為 area_distance 除以了 (max_y - min_y)，通常乘以影像跨度(約720)或直接評估其相對像素差。
            # 這裡我們加上限制：若平均點對點歐氏距離大於實際規定的門檻，則不視為 TP
            actual_pixel_dist = distances[best_gt, i_pred] * 720 # 估算實際像素偏差
            
            if actual_pixel_dist <= pixel_threshold:
                dist += distances[best_gt, i_pred]
                tp += 1
                is_fp.append(False)
            else:
                fp += 1
                is_fp.append(True)
        else:
            fp += 1
            is_fp.append(True)
            
    for i_gt, best_pred in enumerate(best_preds):
        if best_gts[best_pred] != i_gt:
            fn += 1
            is_fn.append(True)
        else:
            # 同理，如果雖然匹配但因為距離過遠被剔除，在 GT 端就要算漏報 (FN)
            actual_pixel_dist = distances[i_gt, best_pred] * 720
            if actual_pixel_dist > pixel_threshold:
                fn += 1
                is_fn.append(True)
            else:
                is_fn.append(False)
                
    if debug:
        show_preds(pred, gt)

    return dist, fp, fn, tp

def convert_tusimple_format(json_gt):
    output = []
    for data in json_gt:
        lanes = [[(x, y) for (x, y) in zip(lane, data['h_samples']) if x >= 0] for lane in data['lanes']
                 if any(x > 0 for x in lane)]
        output.append({
            'raw_file': data['raw_file'],
            'run_time': data['run_time'] if 'run_time' in data else None,
            'lanes': lanes
        })
    return output


# def eval_json(pred_file, gt_file, json_type=None, debug=False):
    try:
        json_pred = [json.loads(line) for line in open(pred_file).readlines()]
    except BaseException as e:
        raise Exception('Fail to load json file of the prediction.')
    json_gt = [json.loads(line) for line in open(gt_file).readlines()]
    if len(json_gt) != len(json_pred):
        raise Exception('We do not get the predictions of all the test tasks')

    if json_type == 'tusimple':
        for gt, pred in zip(json_gt, json_pred):
            pred['h_samples'] = gt['h_samples']
        json_gt = convert_tusimple_format(json_gt)
        json_pred = convert_tusimple_format(json_pred)
    gts = {l['raw_file']: l for l in json_gt}

    total_distance, total_fp, total_fn, run_time = 0., 0., 0., 0.
    for pred in tqdm(json_pred):
        if 'raw_file' not in pred or 'lanes' not in pred:
            raise Exception('raw_file or lanes not in some predictions.')
        raw_file = pred['raw_file']
        pred_lanes = pred['lanes']
        run_time += pred['run_time'] if 'run_time' in pred else 1.

        if raw_file not in gts:
            raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
        gt = gts[raw_file]
        gt_lanes = gt['lanes']

        distance, fp, fn = area_metric(pred_lanes, gt_lanes, debug=debug)

        total_distance += distance
        total_fp += fp
        total_fn += fn

    num = len(gts)
    return json.dumps([{
        'name': 'Distance',
        'value': total_distance / num,
        'order': 'desc'
    }, {
        'name': 'FP',
        'value': total_fp,
        'order': 'asc'
    }, {
        'name': 'FN',
        'value': total_fn,
        'order': 'asc'
    }, {
        'name': 'FPS',
        'value': 1000. * num / run_time
    }])


def eval_json(pred_file, gt_file, json_type=None, debug=False):
    # ... 前方載入 json_pred 與 json_gt 的程式碼保持不變 ...
    gts = {l['raw_file']: l for l in json_gt}

    # 定義實驗對照組：10 像素 (嚴格) vs 30 像素 (放寬)
    thresholds = [10, 30]
    metrics_summary = {}

    for thresh in thresholds:
        total_distance, total_fp, total_fn, total_tp, run_time = 0., 0., 0., 0., 0.
        
        for pred in json_pred:
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            run_time += pred['run_time'] if 'run_time' in pred else 1.
            gt_lanes = gts[raw_file]['lanes']

            # 呼叫我們修改後的 area_metric
            distance, fp, fn, tp = area_metric(pred_lanes, gt_lanes, debug=debug, pixel_threshold=thresh)

            total_distance += distance
            total_fp += fp
            total_fn += fn
            total_tp += tp

        num = len(gts)
        
        # 計算學術標準 Accuracy (基於混淆矩陣之 IoU 概念：TP / (TP + FP + FN))
        accuracy = total_tp / (total_tp + total_fp + total_fn + 1e-6)
        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        metrics_summary[thresh] = {
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'Accuracy': accuracy,
            'F1-Score': f1
        }

    # 漂亮地列印出對比表格，供論文直接截圖或抄錄數據
    print("\n" + "="*20 + " 論文消融實驗：不同像素閾值之敏感度分析 " + "="*20)
    for thresh, res in metrics_summary.items():
        print(f"【門檻門檻限制：{thresh} 像素】")
        print(f"  - True Positives (TP)  : {res['TP']} (越放寬，TP 越高)")
        print(f"  - False Positives (FP) : {res['FP']} (越放寬，誤報 FP 越低)")
        print(f"  - False Negatives (FN) : {res['FN']} (越放寬，漏報 FN 越低)")
        print(f"  - 幾何拓撲 Accuracy    : {res['Accuracy']:.4f}")
        print(f"  - 全域平衡 F1-Score     : {res['F1-Score']:.4f}")
        print("-"*60)
        
    return json.dumps([
        {'name': 'Accuracy_30px', 'value': metrics_summary[30]['Accuracy']},
        {'name': 'FP_30px', 'value': metrics_summary[30]['FP']},
        {'name': 'FN_30px', 'value': metrics_summary[30]['FN']}
    ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute the metrics")
    parser.add_argument('--preds', required=True, type=str, help=".json with the predictions")
    parser.add_argument('--gt', required=True, type=str, help=".json with the GT")
    parser.add_argument('--gt-type', type=str, help='pass `tusimple` if using the TuSimple file format')
    parser.add_argument('--debug', action='store_true', help='show metrics and preds/gts')
    argv = vars(parser.parse_args())

    result = json.loads(eval_json(argv['preds'], argv['gt'], argv['gt_type'], argv['debug']))

    # pretty-print
    table = {}
    for metric in result:
        if metric['name'] not in table.keys():
            table[metric['name']] = []
        table[metric['name']].append(metric['value'])
    print(tabulate(table, headers='keys'))
