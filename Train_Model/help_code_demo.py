import csv, torch, os
import numpy as np

def ACC(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    total = sum(mylist)
    acc = (tp + tn) / total
    return acc


def PPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then ppv should be 1
    if tp + fn == 0:
        ppv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tp + fp == 0 and tp + fn != 0:
        ppv = 0
    else:
        ppv = tp / (tp + fp)
    return ppv


def NPV(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then npv should be 1
    if tn + fp == 0:
        npv = 1
    # for the case: there is some VA segs, but the predictions are wrong
    elif tn + fn == 0 and tn + fp != 0:
        npv = 0
    else:
        npv = tn / (tn + fn)
    return npv


def Sensitivity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no VA segs for the patient, then sen should be 1
    if tp + fn == 0:
        sensitivity = 1
    else:
        sensitivity = tp / (tp + fn)
    return sensitivity


def Specificity(mylist):
    tp, fn, fp, tn = mylist[0], mylist[1], mylist[2], mylist[3]
    # for the case: there is no non-VA segs for the patient, then spe should be 1
    if tn + fp == 0:
        specificity = 1
    else:
        specificity = tn / (tn + fp)
    return specificity


def BAC(mylist):
    sensitivity = Sensitivity(mylist)
    specificity = Specificity(mylist)
    b_acc = (sensitivity + specificity) / 2
    return b_acc


def F1(mylist):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def FB(mylist, beta=2):
    precision = PPV(mylist)
    recall = Sensitivity(mylist)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (1+beta**2) * (precision * recall) / ((beta**2)*precision + recall)
    return f1

def stats_report(mylist):
    f1 = round(F1(mylist), 5)
    fb = round(FB(mylist), 5)
    se = round(Sensitivity(mylist), 5)
    sp = round(Specificity(mylist), 5)
    bac = round(BAC(mylist), 5)
    acc = round(ACC(mylist), 5)
    ppv = round(PPV(mylist), 5)
    npv = round(NPV(mylist), 5)

    output = str(mylist) + '\n' + \
             "F-1 = " + str(f1) + '\n' + \
             "F-B = " + str(fb) + '\n' + \
             "SEN = " + str(se) + '\n' + \
             "SPE = " + str(sp) + '\n' + \
             "BAC = " + str(bac) + '\n' + \
             "ACC = " + str(acc) + '\n' + \
             "PPV = " + str(ppv) + '\n' + \
             "NPV = " + str(npv) + '\n'

    print("F-1 = ", F1(mylist))
    print("F-B = ", FB(mylist))
    print("SEN = ", Sensitivity(mylist))
    print("SPE = ", Specificity(mylist))
    print("BAC = ", BAC(mylist))
    print("ACC = ", ACC(mylist))
    print("PPV = ", PPV(mylist))
    print("NPV = ", NPV(mylist))

    return output

def loadCSV(csvf):
    """
    return a dict saving the information of csv
    :param splitFile: csv file name
    :return: {label:[file1, file2 ...]}
    """
    dictLabels = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]

            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels


def txt_to_numpy(filename, row):
    file = open(filename)
    lines = file.readlines()
    # datamat = np.arange(row, dtype=np.float)
    datamat = np.arange(row, dtype=np.float64)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1

    return datamat


class ToTensor(object):
    def __call__(self, sample):
        text = sample['IEGM_seg']
        return {
            'IEGM_seg': torch.from_numpy(text),
            'label': sample['label']
        }


class IEGM_DataSET1():
    def __init__(self, root_dir, indice_dir, mode, size, transform=None):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.transform = transform

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))

        for i, (k, v) in enumerate(csvdata_all.items()):
            self.names_list.append(str(k) + ' ' + str(v[0]))

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        text_path = self.root_dir + self.names_list[idx].split(' ')[0]

        if not os.path.isfile(text_path):
            print(text_path + 'does not exist')
            return None

        IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
        label = int(self.names_list[idx].split(' ')[1])
        sample = {'IEGM_seg': IEGM_seg[:,2:], 'label': label}

        return sample


class IEGM_DataSET():
    def __init__(self, root_dir, indice_dir, mode, size, transform=None):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.transform = transform

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))

        for i, (k, v) in enumerate(csvdata_all.items()):
            self.names_list.append(str(k) + ' ' + str(v[0]))

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        text_path = self.root_dir + self.names_list[idx].split(' ')[0]

        if not os.path.isfile(text_path):
            print(text_path + 'does not exist')
            return None

        IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
        label = int(self.names_list[idx].split(' ')[1])
        sample = {'IEGM_seg': IEGM_seg, 'label': label}

        return sample




#######################################

def extract_features_for_single_peak_optimized(x_peaks):
    peak_features_count = len(x_peaks)
    diff_intervals = [x - x_peaks[i - 1] for i, x in enumerate(x_peaks)][1:]
    if len(diff_intervals) >1:
        peak_features_min_int = np.min(diff_intervals)
        peak_features_max_int = np.max(diff_intervals)
        peak_features_avg_int = np.ceil(np.average(diff_intervals))
        peak_features_min_cnt = len(np.where(diff_intervals < peak_features_avg_int)[0])
        peak_features_max_cnt = len(np.where(diff_intervals > peak_features_avg_int)[0])
        peak_features_bpm = peak_features_count*12
    else:
        peak_features_min_int = 1
        peak_features_max_int = 1
        peak_features_avg_int = 1
        peak_features_min_cnt = 1
        peak_features_max_cnt = 1
        peak_features_bpm = 12
    
    feature_list=[peak_features_count, peak_features_min_int, peak_features_max_int,peak_features_avg_int,  peak_features_min_cnt, peak_features_max_cnt, peak_features_bpm]
   
    return feature_list

def supress_non_maximum(peak_indices, X_data, window = 30):
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
    if new_peak_indices[-1] != last_peak :
        new_peak_indices.append(last_peak)
        
    return new_peak_indices




def extract_peaks_features_optimized(X_data, std_val=1.8, window=30):
    X_data_new = np.array(X_data)
    std_arr = np.abs(np.std(X_data_new)*std_val)
    peak_indices =np.where(np.abs(X_data_new) > std_arr)[0]

    #plot_all_peak(peak_indices, X_data, color='green')
    peak_indices = supress_non_maximum(peak_indices, X_data, window)
    #plot_all_peak(peak_indices, X_data, color='red')

    peaks_features = extract_features_for_single_peak_optimized(peak_indices)


    return peaks_features

 ###################################   

class UBPercept_peak_DataSET():
    def __init__(self, root_dir, indice_dir, mode, size, transform=None):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.transform = transform

        self.features = {}

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))

        for i, (k, v) in enumerate(csvdata_all.items()):
            self.names_list.append(str(k) + ' ' + str(v[0]))

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        text_path = self.root_dir + self.names_list[idx].split(' ')[0]

        if not os.path.isfile(text_path):
            print(text_path + 'does not exist')
            return None

        IEGM_seg = txt_to_numpy(text_path, self.size)

        if idx not in self.features:
            peaks_features = np.array(extract_peaks_features_optimized(IEGM_seg))
            peaks_features = peaks_features.reshape(1, len(peaks_features), 1)
            self.features[idx]= peaks_features
        
        label = int(self.names_list[idx].split(' ')[1])
        sample = {'IEGM_seg': self.features[idx], 'label': label}

        return sample




def pytorch2onnx(net_path, net_name, size):
    net = torch.load(net_path, map_location=torch.device('cpu'))

    dummy_input = torch.randn(1, 1, size, 1)

    optName = str(net_name)+'.onnx'
    torch.onnx.export(net, dummy_input, optName, verbose=True)
