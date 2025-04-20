""" Training and testing of PTMFusionNet
"""
import csv
from models import init_model_dict, init_optim
from sklearn.model_selection import StratifiedKFold
from utils import *
import pandas as pd
import decimal
import torch
from sklearn.preprocessing import RobustScaler
import numpy as np
from sklearn.metrics import accuracy_score

def prepare_trte_data(data_folder, view_list, protein_self_test_weighted_tr123, protein_self_test_weighted_te123):
        num_view = len(view_list)

        data_tr_list = []
        data_te_list = []
        for i in view_list:
            if i == 1 or i == 2:
                data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_tr.csv"), delimiter=','))
                data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_te.csv"), delimiter=','))
            if i == 3:
                # data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_tr.csv"), delimiter=','))
                # data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_te.csv"), delimiter=','))
                data_tr_list.append(protein_self_test_weighted_tr123)
                data_te_list.append(protein_self_test_weighted_te123)

        num_tr = data_tr_list[0].shape[0]
        num_te = data_te_list[0].shape[0]
        data_mat_list = []
        for i in range(num_view):
            data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
        data_tensor_list = []
        for i in range(len(data_mat_list)):
            data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
            if cuda:
                data_tensor_list[i] = data_tensor_list[i].cuda()
        idx_dict = {}
        idx_dict["tr"] = list(range(num_tr))
        idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))
        data_train_list = []
        data_all_list = []
        for i in range(len(data_tensor_list)):
            data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
            data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                            data_tensor_list[i][idx_dict["te"]].clone()), 0))


        return data_train_list, data_all_list, idx_dict

def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
        adj_metric = "cosine"  # cosine distance
        adj_train_list = []
        adj_test_list = []
        for i in range(len(data_tr_list)):
            adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
            adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
            adj_test_list.append(
                gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))

        return adj_train_list, adj_test_list

def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, train_VCDN=True):
        loss_dict = {}
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        for m in model_dict:
            model_dict[m].train()
        num_view = len(data_list)
        for i in range(num_view):
            optim_dict["C{:}".format(i + 1)].zero_grad()
            ci_loss = 0
            ci = model_dict["C{:}".format(i + 1)](model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i]))

            label = label.view(-1)

            ci_loss = torch.mean(torch.mul(criterion(ci, label), sample_weight))
            ci_loss.backward()
            optim_dict["C{:}".format(i + 1)].step()
            loss_dict["C{:}".format(i + 1)] = ci_loss.detach().cpu().numpy().item()
        if train_VCDN and num_view >= 2:
            optim_dict["C"].zero_grad()
            c_loss = 0
            ci_list = []
            for i in range(num_view):
                ci_list.append(
                    model_dict["C{:}".format(i + 1)](model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])))
            c = model_dict["C"](ci_list)
            c_loss = torch.mean(torch.mul(criterion(c, label), sample_weight))
            c_loss.backward()
            optim_dict["C"].step()
            loss_dict["C"] = c_loss.detach().cpu().numpy().item()

        return loss_dict

def test_epoch(data_list, adj_list, te_idx, model_dict):
        for m in model_dict:
            model_dict[m].eval()
        num_view = len(data_list)
        ci_list = []
        for i in range(num_view):
            ci_list.append(
                model_dict["C{:}".format(i + 1)](model_dict["E{:}".format(i + 1)](data_list[i], adj_list[i])))
        if num_view >= 2:
            c = model_dict["C"](ci_list)
        else:
            c = ci_list[0]
        c = c[te_idx, :]
        prob = F.softmax(c, dim=1).data.cpu().numpy()

        return prob

def train_identification(score_matrix, rank_of_disease, samples_protein, labels):

    positive_biomarker = []

    # read protein name in expression data
    protein_self_test_name1 = pd.read_excel('./Expression Data/THCA/feature_name.xlsx', engine='openpyxl')
    protein_self_test_name = protein_self_test_name1.values


    # Divide the dataset into training and test sets
    X_train_fold, X_val_fold = samples_protein[train_index], samples_protein[test_index]
    y_train_fold, y_val_fold = labels[train_index], labels[test_index]


    # Calculate oringinal ACC
    data_folder = '2'
    view_list = [3]
    num_epoch_pretrain = 500
    num_epoch = 2500
    lr_e_pretrain = 1e-3
    lr_e = 5e-4
    lr_c = 1e-3
    if data_folder == '2':
        num_class = 2

    if data_folder == '3':
        num_class = 3

    if data_folder == '5':
        num_class = 5
    test_inverval = 50
    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)
    if data_folder == '2':
        adj_parameter = 2
        dim_he_list = [200, 200, 100]
    if data_folder == '3':
        adj_parameter = 2
        dim_he_list = [200, 200, 100]
    if data_folder == '5':
        adj_parameter = 10
        dim_he_list = [400, 400, 200]
    data_tr_list, data_trte_list, trte_idx = prepare_trte_data(data_folder, view_list, X_train_fold, X_val_fold)

    labels_trval = np.concatenate([y_train_fold, y_val_fold])


    labels_tr_tensor = torch.LongTensor(labels_trval[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trval[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()

    # print("\nPretrain GCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_VCDN=False)
    # print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)

    ACC_ALL = 0

    for epoch in range(num_epoch + 1):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
        if epoch % test_inverval == 0:
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            ACC_ALL += accuracy_score(labels_trval[trte_idx["te"]], te_prob.argmax(1))
    ACC_ORIN = ACC_ALL/51



    weight_parameter = 2
    weight_biomarker_number=[]



    # Compare the protein names in the association data with the protein names in the expression data to find the shared protein subset.
    for i in range(protein_self_test_name.shape[0]):
        for j in range(protein_bulk_name.shape[0]):
            if protein_self_test_name[i, 0] == protein_bulk_name[j, 1]:
                weight_biomarker_number.append([i, j])

    weight_biomarker_number_double = np.array(weight_biomarker_number)


    for [i,j] in weight_biomarker_number_double :

        protein_self_test_origin_tr = X_train_fold
        protein_self_test_origin_va = X_val_fold
        protein_self_test_weighted_tr = protein_self_test_origin_tr.copy()

        print('--------------------------------------')

        #Training and tuning the model with training and validation sets
        protein_self_test_weighted_val = protein_self_test_origin_va.copy()
        protein_self_test_origin_val = protein_self_test_origin_va.copy()

        # The process of feature weighting
        print("The", i + 1, "th biomarker")
        protein_self_test_weighted_tr[:,i] = protein_self_test_origin_tr [:,i] * weight_parameter * score_matrix[j,rank_of_disease ]
        protein_self_test_weighted_val[:, i] = protein_self_test_origin_val[:, i] * weight_parameter * score_matrix[j, rank_of_disease]
        l1_lambda = 1 # Set to the corresponding best_l1_lambda
        protein_self_test_weighted_tr[:, i] *= np.exp(-l1_lambda * np.abs(protein_self_test_origin_tr[:, i]))
        protein_self_test_weighted_val[:, i] *= np.exp(-l1_lambda * np.abs(protein_self_test_origin_val[:, i]))


        data_folder = '3' # The number of subtypes
        view_list = [3]
        num_epoch_pretrain = 500
        num_epoch = 2500
        lr_e_pretrain = 1e-3
        lr_e = 5e-4
        lr_c = 1e-3
        if data_folder == '2':
            num_class = 2

        if data_folder == '3':
            num_class = 3

        if data_folder == '5':
            num_class = 5
        test_inverval = 50
        num_view = len(view_list)
        dim_hvcdn = pow(num_class, num_view)
        if data_folder == '2':
            adj_parameter = 2
            dim_he_list = [200, 200, 100]

        if data_folder == '3':
            adj_parameter = 2
            dim_he_list = [200, 200, 100]

        if data_folder == '5':
            adj_parameter = 10
            dim_he_list = [400, 400, 200]
        data_tr_list, data_trte_list, trte_idx = prepare_trte_data(data_folder, view_list, protein_self_test_weighted_tr, protein_self_test_weighted_val)

        labels_trval = np.concatenate([y_train_fold , y_val_fold ])


        labels_tr_tensor = torch.LongTensor(labels_trval[trte_idx["tr"]])
        onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
        sample_weight_tr = cal_sample_weight(labels_trval[trte_idx["tr"]], num_class)
        sample_weight_tr = torch.FloatTensor(sample_weight_tr)
        if cuda:
            labels_tr_tensor = labels_tr_tensor.cuda()
            onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
            sample_weight_tr = sample_weight_tr.cuda()
        adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
        dim_list = [x.shape[1] for x in data_tr_list]
        model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
        for m in model_dict:
            if cuda:
                model_dict[m].cuda()

        # print("\nPretrain GCNs...")
        optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
        for epoch in range(num_epoch_pretrain):
            train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                        onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_VCDN=False)
        # print("\nTraining...")
        optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)

        ACC_ALL = 0

        for epoch in range(num_epoch + 1):
            train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                        onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
            if epoch % test_inverval == 0:
                te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
                ACC_ALL += accuracy_score(labels_trval[trte_idx["te"]], te_prob.argmax(1))

        accuracy = ACC_ALL/51

        # If the empowered ACC is higher than the original ACC, then the marker is considered to be a significant marker
        if accuracy  > ACC_ORIN :
            positive_biomarker .append([i , j])

    positive_biomarker_number = np.array(positive_biomarker)

    print("--------------------------------------------------")
    return (positive_biomarker_number)




# read association degree score matrix


file_name = "Association_degree_score_matrix.xlsx"
df = pd.read_excel(file_name, header=None, index_col=None)


S = df.to_numpy()

rank_of_disease=42#the rank of disease in association data
weight_parameter=2 # parameter of feature weighting

# read name of protein in association data
protein_bulk_name1 = pd.read_excel('./Association Data/protein_numbers.xlsx', engine='openpyxl')
protein_bulk_name = protein_bulk_name1.values

# read protein expression data
samples_protein = data_array = pd.read_excel('./Expression Data/KIPAN/samples_protein.xlsx').to_numpy()
samples_protein = samples_protein.astype(float)

# Scale using RobustScaler
scaler = RobustScaler()
samples_protein = scaler.fit_transform(samples_protein)

# read labels of disease
labels1 = pd.read_excel('./Expression Data/KIPAN/labels.xlsx').to_numpy()
labels1 = np.array(labels1, dtype=decimal.Decimal)
labels = np.array(labels1, dtype=float)

# read protein name in expression data
protein_self_test_name1 = pd.read_excel('./Expression Data/KIPAN/feature_name.xlsx', engine='openpyxl')
protein_self_test_name = protein_self_test_name1.values


stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# 5-fold
for fold, (train_index, test_index) in enumerate(stratified_kfold.split(samples_protein, labels)):
    if fold >= 0:
        # Divide the dataset into training and test sets, with the training set accounting for 80% and the test set accounting for 20%.
        X_train, X_temp = samples_protein[train_index], samples_protein[test_index]
        y_train, y_temp = labels[train_index], labels[test_index]

        positive_biomarker_number = train_identification(S, rank_of_disease, samples_protein, labels)

        # Write the screening results to a file, the result is 5 excel files, for the 5 fold cross-screening results, the higher the number of occurrences represents a higher degree of importance
        file_name = f"positive_biomarker_number_fold_{fold}.csv"
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(positive_biomarker_number)





