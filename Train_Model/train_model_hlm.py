

import numpy as np
import pandas as pd
from help_code_demo import ToTensor, IEGM_DataSET
import time
import argparse
import torchvision.transforms as transforms
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree 
from sklearn.metrics import fbeta_score 
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn.datasets import load_iris
from graphviz import Source
from sklearn.ensemble import AdaBoostClassifier

def extract_features_for_single_peak_optimized_v1(x_peaks,  X_data):
# def extract_features_for_single_peak_optimized_v1(x_peaks):
    peak_features_count = len(x_peaks)   
    # print("x_peaks: ", x_peaks)    
    diff_intervals = [x - x_peaks[i - 1] for i, x in enumerate(x_peaks)][1:]

    if len(diff_intervals) >1:
        peak_features_min_int = np.min(diff_intervals)
        peak_features_max_int = np.max(diff_intervals)
        peak_features_avg_int = np.ceil(np.average(diff_intervals))
        
    else:
        peak_features_min_int = 1
        peak_features_max_int = 1
        peak_features_avg_int = 1

    length = peak_features_avg_int * peak_features_count
    
    #  HugeRabbit_Amplitude fluctuation characteristics around Peaks
    std_val = 1.8
    small_peak_counts = []
    X_data_new = np.array(X_data)
    std_arr = np.abs(np.std(X_data_new)*std_val)
    threshold = std_arr
    for peakIndex in x_peaks:
        start = max(0, peakIndex - int(peak_features_min_int / 2))
        end = min(len(X_data) - 1, peakIndex + int(peak_features_min_int / 2))
    peak_range = X_data[start : end + 1]
    small_peaks = [x for x in peak_range if x > threshold]
    small_peak_counts.append(len(small_peaks))
    num_small_peak_counts = small_peak_counts[0]
    # print("small_peak_counts: ", small_peak_counts)
    # if num_small_peak_counts > 4:
    #     flag = 1
    # else: 
    #     flag = 0
    # print("num_small_peak_counts: ", num_small_peak_counts)


    # ### HugeRabbit_Add feature of Approximate entropy
    # def ApEn(U, m, r) -> float:
    # ### """Approximate_entropy."""
    #     def _maxdist(x_i, x_j):
    #         return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    #     def _phi(m):
    #         x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
    #         C = [
    #             len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
    #             for x_i in x
    #         ]
    #         return (N - m + 1.0) ** (-1) * sum(np.log(C))
    #     N = len(U)            
    #     return abs(_phi(m + 1) - _phi(m))
    
    # # print("Size of X_data: ", X_data.shape)
    # print("Processing_X_data_ApEn_time: ", time.time())
    # # X_data_ApEn = ApEn(X_data, int(peak_features_avg_int) , 0.05)
    # X_data_ApEn = ApEn(X_data, 2 , 0.05)
    # print("X_data_ApEn: ", X_data_ApEn)
        
    # U = np.array([85, 80, 89] * 17)
    # U_ApEn = ApEn(U, 2, 10 )
    # print("ApEn(U, 2, 3): ", U_ApEn)
    # randU = np.random.choice([85, 80, 89], size=17*3)
    # randU_ApEn = ApEn(randU, 2, 10)
    # print("ApEn(randU, 2, 3)", randU_ApEn)    
    
    # feature_list=[peak_features_count, peak_features_min_int, peak_features_max_int,peak_features_avg_int, X_data_ApEn]    
    # feature_list=[peak_features_count, peak_features_min_int, peak_features_max_int,peak_features_avg_int, num_small_peak_counts]
    feature_list= [peak_features_count, peak_features_avg_int] 
    # print("feature_list: ", feature_list)  
    return feature_list



def supress_non_maximum(peak_indices, X_data, window = 30):
    
    if len(peak_indices)<1:
        return []

    new_peak_indices=[]
    last_peak=peak_indices[0]
    for i in range(1, len(peak_indices)):
        curr_diff = peak_indices[i] - last_peak
        if curr_diff > window:
            new_peak_indices.append(last_peak)
            last_peak = peak_indices[i]
        else:
            if X_data[peak_indices[i]] > X_data[last_peak]:
                last_peak = peak_indices[i]
    if len(new_peak_indices)==0:
        return new_peak_indices
    if new_peak_indices[-1] != last_peak :
        new_peak_indices.append(last_peak)
        
    return new_peak_indices



def extract_peaks_features_optimized_v1(X_data, std_val=1.8, window=38):
    X_data_new = np.array(X_data)
    std_arr = np.abs(np.std(X_data_new)*std_val)   
    # print("std_arr: ", std_arr)
    
    peak_indices =np.where(np.abs(X_data_new) > std_arr)[0]   
    peak_indices = supress_non_maximum(peak_indices, X_data, window)
    peaks_features = extract_features_for_single_peak_optimized_v1(peak_indices, X_data)

    return peaks_features


def extract_features_extended(X_data, sigma=1.8, window=38):
    X_data_peak_features = []
   
    # print("X_data: ", X_data)
    
    for i in range(len(X_data)):
        X_data_features = extract_peaks_features_optimized_v1(X_data[i], sigma, window)
        X_data_peak_features.append(X_data_features)
  
    X_data_peaks_feat_df = pd.DataFrame.from_dict(X_data_peak_features)

    return X_data_peaks_feat_df


def extract_labels(dataset):
    dataset_labels = []
    for itm in dataset.names_list:
        dataset_labels.append(itm.split(' ')[0].split('-')[1])
    return dataset_labels

def load_dataset(dataset):
    X_data = []
    y_data = []
    for i in dataset:
        iegm_seg = i['IEGM_seg'].flatten()
        label = i['label']
        X_data.append(iegm_seg)
        y_data.append(label)
    return np.array(X_data), np.array(y_data).reshape(-1,1), np.array(extract_labels(dataset))

def show_validation_results(C, total_time):
        #C = C_board
        print(C)

        #total_time = 0#sum(timeList)
        avg_time = total_time#np.mean(timeList)
        acc = (C[0][0] + C[1][1]) / (C[0][0] + C[0][1] + C[1][0] + C[1][1])
        precision = C[1][1] / (C[1][1] + C[0][1])
        sensitivity = C[1][1] / (C[1][1] + C[1][0])
        FP_rate = C[0][1] / (C[0][1] + C[0][0])
        PPV = C[1][1] / (C[1][1] + C[1][0])
        NPV = C[0][0] / (C[0][0] + C[0][1])
        F1_score = (2 * precision * sensitivity) / (precision + sensitivity)
        F_beta_score = (1+2**2) * (precision * sensitivity) / ((2**2)*precision + sensitivity)

        print("\nacc: {},\nprecision: {},\nsensitivity: {},\nFP_rate: {},\nPPV: {},\nNPV: {},\nF1_score: {}, "
                "\ntotal_time: {},\n average_time: {}".format(acc, precision, sensitivity, FP_rate, PPV, NPV, F1_score,
                                                        total_time, avg_time))

        print("F_beta_score : ", F_beta_score)

def main(args):
    # Hyperparameters
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices

    hyper_param = {}
    hyper_param["sigma"] = args.sigma
    hyper_param["window"] = args.window
    hyper_param["depth"] = args.depth
    hyper_param["n_estimators"] = args.n_estimators
    hyper_param["learning_rate"] = args.learning_rate
    #sigma = args.sigma
    #window = args.window


    print("Hyperparameters: ")
    print(hyper_param)

    
    # Start dataset loading
    trainset = IEGM_DataSET(root_dir=path_data,
                            indice_dir=path_indices,
                            mode='train',
                            size=SIZE,
                            transform=transforms.Compose([ToTensor()]))

    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode='test',
                           size=SIZE,
                           transform=transforms.Compose([ToTensor()]))


    X_test, y_test, _ = load_dataset(testset)
    X_train, y_train, _ = load_dataset(trainset)

    start_time = time.time()

    X_train_peaks_feat_extend_df = extract_features_extended(X_train, hyper_param["sigma"], hyper_param["window"])
    
    feature_process_time = (time.time() - start_time)
    print("--- feature_process_time --- %s seconds ---" % feature_process_time)
    
    X_test_peaks_feat_extend_df = extract_features_extended(X_test, hyper_param["sigma"], hyper_param["window"])    

    # Decision Tree
    DTC_1 = DecisionTreeClassifier(criterion='entropy', max_depth=hyper_param["depth"], random_state=0)
    DTC_1.fit(X_train_peaks_feat_extend_df, y_train)
    

    # ### HugeRabbit_Random Forest
    #     # from sklearn.ensemble import RandomForestClassifier
    #     # 创建随机森林分类器
    #     DTC_1 = RandomForestClassifier(n_estimators=hyper_param["n_estimators"], criterion='entropy', max_depth=hyper_param["depth"], random_state=0)
    #     # 训练随机森林分类器
    #     DTC_1.fit(X_train_peaks_feat_extend_df, y_train)
    # from sklearn.tree import export_graphviz
    # import graphviz
    # # 可视化随机森林中的第一棵决策树
    # estimator = DTC_1.estimators_[0]
    # dot_data = export_graphviz(estimator, out_file=None, feature_names=X_train_peaks_feat_extend_df.columns, class_names=['0', '1'], filled=True, rounded=True, special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph.render('RandomForest_Tree_Visualization')
    # # 可视化随机森林中的所有决策树
    # for i, estimator in enumerate(DTC_1.estimators_):
    #     dot_data = export_graphviz(estimator, out_file=None, feature_names=X_train_peaks_feat_extend_df.columns, class_names=['0', '1'], filled=True, rounded=True, special_characters=True)
    #     graph = graphviz.Source(dot_data)
    #     graph.render('RandomForest_Tree_Visualization_{}'.format(i))


    ### HugeRabbit_AdaBoost
    # 创建决策树分类器
    # DTC_1_origin = DecisionTreeClassifier(criterion='entropy', max_depth=hyper_param["depth"], random_state=0)
    # 创建 AdaBoost 分类器，将决策树分类器作为基分类器
    # print("AdaBoostClassifier: ")
    # DTC_1 = AdaBoostClassifier(
    #                            # base_estimator=DTC_1_origin, 
    #                            base_estimator=None,
    #                            # algorithm='SAMME.R',
    #                            algorithm='SAMME',
    #                            n_estimators=hyper_param["n_estimators"], 
    #                            learning_rate=hyper_param["learning_rate"], 
    #                            random_state=0)
    # # 训练 AdaBoost 分类器
    # DTC_1.fit(X_train_peaks_feat_extend_df, y_train)

    #LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(X_train_peaks_feat_extend_df, y_train)

    print("Train:")
    # 22
    
    dataset = IEGM_DataSET(root_dir=args.path_data,
                            indice_dir=args.path_indices,
                            mode='total',
                            size=args.size,
                            transform=transforms.Compose([ToTensor()]))
    X_dataset, Y_dataset, _ = load_dataset(dataset)
    X_dataset_peaks_feat_extend_df = extract_features_extended(X_dataset, hyper_param["sigma"], hyper_param["window"])
    y_dataset_pred = DTC_1.predict(X_dataset_peaks_feat_extend_df)
    
 
    counts = np.bincount(y_dataset_pred)
    print("Number of zeros: ", counts[0])
    print("Number of ones: ", counts[1])
    print("Distribution of zeros and ones: ", counts)
    print("Size of y_dataset_pred: ", y_dataset_pred.shape)
    print("y_dataset_pred: ", y_dataset_pred)
    print("Size of Y_dataset: ", Y_dataset.shape)
    print("Y_dataset: ", Y_dataset)
    
    
    
    # ### HugeRabbit_Finding the wrong predicted labels
    # # 首先将 arr2 调整为与 arr1 形状相同, shape: (30213,)
    # Y_dataset_reshaped = Y_dataset.reshape(-1) 
    # print("Y_dataset_reshaped: ", Y_dataset_reshaped)
    # # 使用 np.where 找出两个数组中不同的元素的索引
    # a1= np.array([1,1,1,1,0,1])
    # a2= np.array([0,1,0,1,0,1])
    # demo_diff_indices = np.where(a1 != a2)
    # print("demo_diff_indices: ",demo_diff_indices[0])
    # diff_indices = np.where(y_dataset_pred != Y_dataset_reshaped)
    # print("diff_indices: ",diff_indices)
    # diff_indices_array = diff_indices[0]
    # print("diff_indices_array: ",diff_indices_array)
    # print("Size of diff_indices_array: ",diff_indices_array.shape)
    
    # import pandas as pd
    # # 加载 .csv 文件
    # df = pd.read_csv('./data_indices/total_indice.csv')
    # # 使用 .iloc 对 DataFrame 进行切片，选取 diff_indices_array 中指定的行，生成新的 DataFrame
    # df_new = df.iloc[diff_indices_array]
    # # 保存新的 DataFrame 为 .csv 文件
    # df_new.to_csv('diff_indices.csv', index=False)
    
    # import pandas as pd
    # import shutil
    # import os
    # # 读取 .csv 文件
    # df = pd.read_csv('diff_indices.csv', header=None)  
    # # header=None 表示 .csv 文件没有列名，如果有列名则不需要该参数
    # # 获取你要复制的文件列表
    # file_list = df[1].tolist()
    # # 检查并创建目标文件夹
    # dst_dir = "diff_data_set"  # 你的目标文件夹
    # if not os.path.exists(dst_dir):
    #     os.makedirs(dst_dir)
    # # 从原始文件夹复制文件到目标文件夹
    # src_dir = "data_set"  # 你的源文件夹
    # for file_name in file_list:
    #     src_file = os.path.join(src_dir, file_name)
    #     dst_file = os.path.join(dst_dir, file_name)      
    #     # 如果源文件存在，那么复制文件
    #     if os.path.exists(src_file):  
    #         shutil.copy2(src_file, dst_file)  # copy2 将尽可能地保留元数据信息
         
    
    C_DT_train = confusion_matrix(Y_dataset,  y_dataset_pred)
    print(C_DT_train)
    print("False Positives % : " ,100.0*(C_DT_train[0][1]/(C_DT_train[1][1]+C_DT_train[0][1])))
    print("False Negative % : " ,100.0*(C_DT_train[1][0]/(C_DT_train[1][0]+C_DT_train[0][0])))    
    cur_f_beta = fbeta_score(Y_dataset,  y_dataset_pred, beta=2) 
    print(f'F_beta score: {cur_f_beta}') 

    print("test:")
    test_pred = DTC_1.predict(X_test_peaks_feat_extend_df)
    C_DT_test = confusion_matrix(y_test, test_pred)
    print(C_DT_test)
    print("False Positives % : " ,100.0*(C_DT_test[0][1]/(C_DT_test[1][1]+C_DT_test[0][1])))
    print("False Negative % : " ,100.0*(C_DT_test[1][0]/(C_DT_test[1][0]+C_DT_test[0][0])))     
    cur_f_beta = fbeta_score(y_test, DTC_1.predict(X_test_peaks_feat_extend_df), beta=2) 
    print(f'F_beta score: {cur_f_beta}') 

    print("Decision Tree")
    
    #print("Best Intercept" , LR.intercept_, "Best coeff", LR.coef_)
    # https://mljar.com/blog/extract-rules-decision-tree/ 
    # get the text representation
    
    ### 决策树可视化(Decision Tree)
    text_representation = tree.export_text(DTC_1)
    print(text_representation)


    ### HugeRabbit_Random Forest
    # 可视化随机森林中的第一棵决策树
    # for i in range(len(DTC_1.estimators_)):
    #     tree = DTC_1.estimators_[i]
    #     export_graphviz(tree, out_file='tree.dot', feature_names=X_train_peaks_feat_extend_df.columns, class_names=['0', '1'], filled=True)
    #     Source.from_file('tree.dot').view()
    
        # estimator = DTC_1.estimators_[0]
        # dot_data = export_graphviz(estimator, out_file=None, feature_names=X_train_peaks_feat_extend_df.columns, class_names=['0', '1'], filled=True, rounded=True, special_characters=True)
        # graph = graphviz.Source(dot_data)
        # graph.render('RandomForest_Tree_Visualization')
        # print(graph)
        # # 可视化随机森林中的所有决策树
        # for i, estimator in enumerate(DTC_1.estimators_):
        #     dot_data = export_graphviz(estimator, out_file=None, feature_names=X_train_peaks_feat_extend_df.columns, class_names=['0', '1'], filled=True, rounded=True, special_characters=True)
        #     graph = graphviz.Source(dot_data)
        #     graph.render('RandomForest_Tree_Visualization_{}'.format(i))
        #     print(graph)    
    # from sklearn.datasets import load_iris
    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.tree import export_graphviz
    # from graphviz import Source
    
    
    # print("========================================")
    
    # total_time = (time.time() - start_time)
    # print("Train score", round(DTC_1.score(X_train_peaks_feat_extend_df, y_train), 4))
    # print("Test score", round(DTC_1.score(X_test_peaks_feat_extend_df, y_test), 4))

    # y_test_pred = DTC_1.predict(X_test_peaks_feat_extend_df)
    # print()
    # C_DT = confusion_matrix(y_test, y_test_pred)
    # show_validation_results(C_DT, total_time)    


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='./data_set/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')
    argparser.add_argument('--sigma', type=float, default=1.8)
    # argparser.add_argument('--window', type=int, default=40)
    argparser.add_argument('--window', type=int, default=40)
    argparser.add_argument('--depth', type=int, default=2)
    argparser.add_argument('--n_estimators', type=int, default=200)
    argparser.add_argument('--learning_rate', type=float, default=0.01)
    
    argparser.add_argument('--mode', type=str, default='total', choices=['total','train','eval','test'], help="evaluation mode.")
    
    args = argparser.parse_args()

    start_time = time.time()
    
    main(args)

    total_time = (time.time() - start_time)
    
    print("--- %s seconds ---" % total_time)
