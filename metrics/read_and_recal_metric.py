import numpy as np
import json
import os
import argparse
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute P-R metrics from per-sample metric JSON files')
    parser.add_argument('--file_dir', type=str, required=True,
                        help='Directory containing per-sample metric JSON files (output of metrics.py)')
    parser.add_argument('--bin_number', type=int, default=30,
                        help='Number of bins to aggregate results into (default: 30)')
    parser.add_argument('--max_bin', type=int, default=30,
                        help='Total number of bins in the metric files (default: 30)')
    args = parser.parse_args()

    file_dir = args.file_dir
    file_list = os.listdir(file_dir)

    key_list = ['0.0', '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1.0']

    bin_number_list = [args.bin_number]
    max_bin = args.max_bin
    for bin_number in bin_number_list:
        
        avg_p_all = []
        avg_p_50 = []
        avg_p_75 = []
        avg_p_95 = []

        avg_r_all = []
        avg_r_50 = []
        avg_r_75 = []
        avg_r_95 = []

        output_list = []

        for file in tqdm(file_list):

            file_path = os.path.join(file_dir, file)
            
            with open(file_path, 'r') as f:
                data = json.load(f)

            all_p = []
            all_r = []

            for key in key_list:
                TP_thershold = data['TP'][key]
                FP_thershold = data['FP'][key]
                FN_thershold = data['FN'][key]

                p_key = []
                r_key = []

                reorder_TP_thershold = [0] * bin_number
                reorder_FP_thershold = [0] * bin_number
                reorder_FN_thershold = [0] * bin_number
                
                for i, tp in enumerate(TP_thershold):
                    loc = i // (max_bin // bin_number)
                    
                    if tp == np.nan:
                        reorder_TP_thershold[loc] += 0
                    else:
                        reorder_TP_thershold[loc] += tp
                        
                for i, fp in enumerate(FP_thershold):
                    loc = i // (max_bin // bin_number)
                    
                    if fp == np.nan:
                        reorder_FP_thershold[loc] += 0
                    else:
                        reorder_FP_thershold[loc] += fp
                        
                for i, fn in enumerate(FN_thershold):
                    loc = i // (max_bin // bin_number)
                    
                    if fn == np.nan:
                        reorder_FN_thershold[loc] += 0
                    else:
                        reorder_FN_thershold[loc] += fn
                        
                
                for tp, fp, fn in zip(reorder_TP_thershold, reorder_FP_thershold, reorder_FN_thershold):
                    # 计算 precision 和 recall
                    if tp + fp + fn == 0:
                        continue
                    else:
                        # 计算 precision
                        if tp + fp == 0:
                            p = 1.0  # 没有预测为正的样本，precision 为 1
                        else:
                            p = tp / (tp + fp)
                        
                        # 计算 recall
                        if tp + fn == 0:
                            r = 1.0  # 没有真正的正样本，recall 为 1
                        else:
                            r = tp / (tp + fn)
                    
                    p_key.append(p)
                    r_key.append(r)

                p_key = np.mean(np.array(p_key))
                r_key = np.mean(np.array(r_key))

                all_p.append(p_key)
                all_r.append(r_key)

                if key == '0.5':
                    avg_p_50.append(p_key)
                    avg_r_50.append(r_key)

                elif key == '0.75':
                    avg_p_75.append(p_key)
                    avg_r_75.append(r_key)
                elif key == '0.95':
                    avg_p_95.append(p_key)
                    avg_r_95.append(r_key)
            
            avg_p_all.append(np.mean(np.array(all_p)))
            avg_r_all.append(np.mean(np.array(all_r)))

            output_list.append([file, float(np.mean(np.array(all_p)))])

        avg_p_all = np.mean(np.array(avg_p_all))
        avg_p_50 = np.mean(np.array(avg_p_50))
        avg_p_75 = np.mean(np.array(avg_p_75))
        avg_p_95 = np.mean(np.array(avg_p_95))

        avg_r_all = np.mean(np.array(avg_r_all))
        avg_r_50 = np.mean(np.array(avg_r_50))
        avg_r_75 = np.mean(np.array(avg_r_75))
        avg_r_95 = np.mean(np.array(avg_r_95))

        avg_f1_all = 2 * avg_p_all * avg_r_all / (avg_p_all + avg_r_all)
        avg_f1_50 = 2 * avg_p_50 * avg_r_50 / (avg_p_50 + avg_r_50)
        avg_f1_75 = 2 * avg_p_75 * avg_r_75 / (avg_p_75 + avg_r_75)
        avg_f1_95 = 2 * avg_p_95 * avg_r_95 / (avg_p_95 + avg_r_95)

        print('p', avg_p_50, avg_p_75, avg_p_95, avg_p_all)
        print('r', avg_r_50, avg_r_75, avg_r_95, avg_r_all)
        print('f1', avg_f1_50, avg_f1_75, avg_f1_95, avg_f1_all)

    # json.dump(output_list, open('result.json', 'w'))

