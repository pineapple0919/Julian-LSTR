import sys
import numpy as np

# 1. 引入系統設定
from config import system_configs

EXPS_DIR = 'experiments'

class Evaluator(object):
    def __init__(self, dataset, exp_dir, poly_degree=3):
        self.dataset = dataset
        self.predictions = None
        self.runtimes = np.zeros(len(dataset))
        self.loss = np.zeros(len(dataset))
        self.exp_dir = exp_dir
        self.new_preds = False

    def add_prediction(self, idx, pred, runtime):
        if self.predictions is None:
            # 2. 【關鍵修正】：不使用 pred.shape[1]，改用固定的最大查詢數量
            max_possible_lanes = system_configs.num_queries
            self.predictions = np.zeros((len(self.dataset._annotations), max_possible_lanes, pred.shape[2]))
            
        self.predictions[idx, :pred.shape[1], :] = pred
        self.runtimes[idx] = runtime
        self.new_preds = True

    def eval(self, **kwargs):
        return self.dataset.eval(self.exp_dir, self.predictions, self.runtimes, **kwargs)