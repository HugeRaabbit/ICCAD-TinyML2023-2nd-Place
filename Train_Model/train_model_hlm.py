import numpy as np
import pandas as pd
import shutil
import os
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

def extract_features(X_data, sigma=1.8, window=38):  
    
    def extract_refined_peak_features(x_peaks,  X_data):
        
        # # Approximate Entropy
        # def ApEn(U, m, r) -> float:
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
        
        # Amplitude Fluctuation
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
  
        
        # feature_list=[peak_features_count, peak_features_min_int, peak_features_max_int,peak_features_avg_int, X_data_ApEn]    
        # feature_list=[peak_features_count, peak_features_min_int, peak_features_max_int,peak_features_avg_int, num_small_peak_counts]
        feature_list= [peak_features_count, peak_features_avg_int] 
        # print("feature_list: ", feature_list)  
        return feature_list


    def filter_peaks(X_data, std_val=1.8, window=38):
        X_data_new = np.array(X_data)
        std_arr = np.abs(np.std(X_data_new)*std_val)
        # print("std_arr: ", std_arr)
        
        peak_indices =np.where(np.abs(X_data_new) > std_arr)[0]

        
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
        
        #plot_all_peak(peak_indices, X_data, color='green')
        peak_indices = new_peak_indices
                       
        #plot_all_peak(peak_indices, X_data, color='red')

        peaks_features = extract_refined_peak_features(peak_indices, X_data)


        return peaks_features

    X_data_peak_features = []
    
    for i in range(len(X_data)):
        X_data_features = filter_peaks(X_data[i], sigma, window)
        X_data_peak_features.append(X_data_features)
  
    X_data_peaks_feat_df = pd.DataFrame.from_dict(X_data_peak_features)

    return X_data_peaks_feat_df

def dataloader(dataset):
    X_data = []
    y_data = []
    
    dataset_labels = []
    
    for itm in dataset.names_list:
        dataset_labels.append(itm.split(' ')[0].split('-')[1])

    for i in dataset:
        iegm_seg = i['IEGM_seg'].flatten()
        label = i['label']
        X_data.append(iegm_seg)
        y_data.append(label)
        
    return np.array(X_data), np.array(y_data).reshape(-1,1), np.array(dataset_labels)


def main(args):
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    
    hyper_param = {}
    hyper_param["sigma"] = args.sigma
    hyper_param["window"] = args.window
    hyper_param["depth"] = args.depth
    hyper_param["n_estimators"] = args.n_estimators
    hyper_param["learning_rate"] = args.learning_rate
    print("Hyperparameters: ", hyper_param)
    
    trainset = IEGM_DataSET(root_dir=path_data,
                            indice_dir=path_indices,
                            mode='train',
                            size=SIZE,
                            transform=transforms.Compose([ToTensor()]))
    X_train, y_train, _ = dataloader(trainset)
    start_time_train = time.time() 
    X_train_peaks_feat_extend_df = extract_features(X_train, hyper_param["sigma"], hyper_param["window"])
    feature_process_time_train = (time.time() - start_time_train)
    print("Feature Processing Time of train_indice(seconds): ", feature_process_time_train)
    
    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode='test',
                           size=SIZE,
                           transform=transforms.Compose([ToTensor()]))
    X_test, y_test, _ = dataloader(testset)
    start_time_test = time.time() 
    X_test_peaks_feat_extend_df = extract_features(X_test, hyper_param["sigma"], hyper_param["window"])
    feature_process_time_test = (time.time() - start_time_test)
    print("Feature Processing Time of test_indice(seconds): ", feature_process_time_test)
    
    dataset = IEGM_DataSET(root_dir=args.path_data,
                            indice_dir=args.path_indices,
                            mode='total',
                            size=args.size,
                            transform=transforms.Compose([ToTensor()]))
    X_dataset, Y_dataset, _ = dataloader(dataset)
    start_time_total = time.time() 
    X_dataset_peaks_feat_extend_df = extract_features(X_dataset, hyper_param["sigma"], hyper_param["window"])
    feature_process_time_total = (time.time() - start_time_total)
    print("Feature Processing Time of total_indice(seconds): ", feature_process_time_total)
    
    print("\n------ Training Process ------") 
    
    print("Classifier: Decision Tree") 
    DT = DecisionTreeClassifier(criterion='entropy', max_depth=hyper_param["depth"], random_state=0)
    DT.fit(X_train_peaks_feat_extend_df, y_train)   
    text_representation = tree.export_text(DT)
    print(text_representation)

    # print("Classifier: Random Forest") 
    # RF = RandomForestClassifier(n_estimators=hyper_param["n_estimators"], criterion='entropy', max_depth=hyper_param["depth"], random_state=0)
    # RF.fit(X_train_peaks_feat_extend_df, y_train)
    # estimator = RF.estimators_[0]
    # dot_data = export_graphviz(estimator, out_file=None, feature_names=X_train_peaks_feat_extend_df.columns, class_names=['0', '1'], filled=True, rounded=True, special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph.render('RandomForest_Tree_Visualization')
    # print(graph)
    # for i, estimator in enumerate(RF.estimators_):
    #     dot_data = export_graphviz(estimator, out_file=None, feature_names=X_train_peaks_feat_extend_df.columns, class_names=['0', '1'], filled=True, rounded=True, special_characters=True)
    #     graph = graphviz.Source(dot_data)
    #     graph.render('RandomForest_Tree_Visualization_{}'.format(i))
    #     print(graph)   

    # print("Classifier: AdaBoost") 
    # Ada_origin = DecisionTreeClassifier(criterion='entropy', max_depth=hyper_param["depth"], random_state=0)
    # print("AdaBoostClassifier: ")
    # Ada = AdaBoostClassifier(
    #                            # base_estimator=Ada_origin, 
    #                            base_estimator=None,
    #                            # algorithm='SAMME.R',
    #                            algorithm='SAMME',
    #                            n_estimators=hyper_param["n_estimators"], 
    #                            learning_rate=hyper_param["learning_rate"], 
    #                            random_state=0)
    # Ada.fit(X_train_peaks_feat_extend_df, y_train)

    print("\n------ Analyzing Process ------")
    
    y_dataset_pred = DT.predict(X_dataset_peaks_feat_extend_df)
    counts = np.bincount(y_dataset_pred)
    print("Distribution of zeros and ones: ", counts)
    print("Size of y_dataset_pred: ", y_dataset_pred.shape)
    print("Size of Y_dataset: ", Y_dataset.shape)
    Y_dataset_reshaped = Y_dataset.reshape(-1) 
    diff_indices = np.where(y_dataset_pred != Y_dataset_reshaped)
    diff_indices_array = diff_indices[0]
    print("Size of diff_indices_array: ",diff_indices_array.shape)
    df = pd.read_csv('./data_indices/total_indice.csv')
    df_new = df.iloc[diff_indices_array]
    df_new.to_csv('diff_indices.csv', index=False)
    df = pd.read_csv('diff_indices.csv', header=None)  
    file_list = df[1].tolist()
    dst_dir = "diff_data_set"  
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    src_dir = "data_set"  
    for file_name in file_list:
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(dst_dir, file_name)      
        if os.path.exists(src_file):  
            shutil.copy2(src_file, dst_file)  
         
    print("\n------ Testing Process ------")
    
    # DT_dataset = confusion_matrix(Y_dataset,  y_dataset_pred)
    # print("False Positives on train_indice % : " ,100.0*(DT_dataset[0][1]/(DT_dataset[1][1]+DT_dataset[0][1])))
    # print("False Negative on train_indice % : " ,100.0*(DT_dataset[1][0]/(DT_dataset[1][0]+DT_dataset[0][0])))    
    cur_f_beta = fbeta_score(Y_dataset,  y_dataset_pred, beta=2) 
    print(f'F_beta score on train_indice: {cur_f_beta}') 
    
    test_pred = DT.predict(X_test_peaks_feat_extend_df)
    # DT_testset = confusion_matrix(y_test, test_pred)
    # print(DT_testset)
    # print("False Positives % : " ,100.0*(DT_testset[0][1]/(DT_testset[1][1]+DT_testset[0][1])))
    # print("False Negative % : " ,100.0*(DT_testset[1][0]/(DT_testset[1][0]+DT_testset[0][0])))     
    cur_f_beta = fbeta_score(y_test, test_pred, beta=2) 
    print(f'F_beta score on train_indice: {cur_f_beta}') 



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='./data_set/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')
    argparser.add_argument('--sigma', type=float, default=1.8)
    argparser.add_argument('--window', type=int, default=40)
    argparser.add_argument('--depth', type=int, default=1)
    argparser.add_argument('--n_estimators', type=int, default=200)
    argparser.add_argument('--learning_rate', type=float, default=0.01)    
    argparser.add_argument('--mode', type=str, default='total', choices=['total','train','eval','test'], help="evaluation mode.") 
    args = argparser.parse_args()

    start_time = time.time() 
    main(args)
    total_time = (time.time() - start_time)
    print("\nTotal Processing Time: %s seconds " % total_time)
