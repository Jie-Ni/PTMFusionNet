""" Training and testing of PTMFusionNet
"""
from models import init_model_dict, init_optim
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from utils import *
import pandas as pd
import decimal
import torch
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


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

def train_and_test(data_folder, view_list, protein_self_test_weighted_tr, protein_self_test_weighted_te,y_train,y_test, num_epoch_pretrain, num_epoch, lr_e_pretrain, lr_e, lr_c, num_class, adj_parameter, dim_he_list, num_view):

    data_tr_list, data_trte_list, trte_idx= prepare_trte_data(data_folder, view_list, protein_self_test_weighted_tr, protein_self_test_weighted_te)
    labels_trte = np.concatenate([y_train, y_test])

    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
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

    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_VCDN=False)

    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)

    ACC_ALL = 0
    F1_ALL = 0
    AUC_ALL = 0
    F1_weighted=0
    F1_macro=0
    for epoch in range(num_epoch + 1):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
        if epoch % test_inverval == 0:
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            # print("\nTest: Epoch {:d}".format(epoch))
            if num_class == 2:

                ACC_ALL += accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                F1_ALL += f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                AUC_ALL += roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])
            else:

                ACC_ALL += accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                F1_weighted += f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')
                F1_macro += f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')

    if num_class == 2:
        return ACC_ALL / 51, F1_ALL / 51, AUC_ALL / 51
    else:
        return ACC_ALL / 51, F1_weighted / 51, F1_macro / 51

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
# scaler = RobustScaler()
# samples_protein = scaler.fit_transform(samples_protein)

# read labels of disease
labels1 = pd.read_excel('./Expression Data/KIPAN/labels.xlsx').to_numpy()
labels1 = np.array(labels1, dtype=decimal.Decimal)
labels = np.array(labels1, dtype=float)

# read protein name in expression data
protein_self_test_name1 = pd.read_excel('./Expression Data/KIPAN/feature_name.xlsx', engine='openpyxl')
protein_self_test_name = protein_self_test_name1.values

# initial StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Before performing disease subtype classification, conduct five-fold cross-validation to select the l1_lambda parameter corresponding to the highest ACC, which will be used for feature weighting.
# for fold, (train_index, test_index) in enumerate(stratified_kfold.split(samples_protein, labels)):
#     if fold >= 0:
#         # Divide the dataset into training and test sets, with the training set accounting for 80% and the test set accounting for 20%.
#         X_train, X_temp = samples_protein[train_index], samples_protein[test_index]
#         y_train, y_temp = labels[train_index], labels[test_index]
#
#         # Then divide the remaining data into validation and test sets, each accounting for 15%.
#         X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
#
#         best_l1_lambda = None
#         best_ACC = 0
#
#         # Perform grid search on l1_lambda.
#         for l1_lambda in (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3):
#
#             protein_self_test_weighted_tr = X_train.copy()
#             protein_self_test_weighted_val = X_val.copy()
#
#             weight_biomarker_number=[]
#
#             # Compare the protein names in the association data with the protein names in the expression data to find the shared protein subset.
#             for i in range(protein_self_test_name.shape[0]):
#                 for j in range(protein_bulk_name.shape[0]):
#                     if protein_self_test_name[i, 0] == protein_bulk_name[j, 1]:
#                         weight_biomarker_number.append([i, j])
#
#             # weight_biomarker_number_double[i,j] stores the rankings of the shared protein in the association data and expression data, respectively.
#             weight_biomarker_number_double = np.array(weight_biomarker_number)
#
#
#             # The process of feature weighting
#             for [i, j] in weight_biomarker_number_double:
#                 protein_self_test_weighted_tr[:, i] = X_train[:, i] * weight_parameter * S[
#                     j, rank_of_disease]
#                 protein_self_test_weighted_val[:, i] = X_val[:, i] * weight_parameter * S[
#                     j, rank_of_disease]
#                 protein_self_test_weighted_tr[:, i] *= np.exp(-l1_lambda * np.abs(X_train[:, i]))
#                 protein_self_test_weighted_val[:, i] *= np.exp(-l1_lambda * np.abs(X_val[:, i]))
#
#             # Determine the best_l1_lambda using the results of disease classification.
#             data_folder = '2' # Write the corresponding Arabic numeral for the number of subtypes.
#             view_list = [3]
#             num_epoch_pretrain = 500
#             num_epoch = 2500
#             lr_e_pretrain = 1e-3
#             lr_e = 5e-4
#             lr_c = 1e-3
#             if data_folder == '2':
#                 num_class = 2
#             if data_folder == '3':
#                 num_class = 3
#             if data_folder == '5':
#                 num_class = 5
#
#             if data_folder == '2':
#                 adj_parameter = 2
#                 dim_he_list = [200, 200, 100]
#             if data_folder == '3':
#                 adj_parameter = 2
#                 dim_he_list = [200, 200, 100]
#             if data_folder == '5':
#                 adj_parameter = 10
#                 dim_he_list = [400, 400, 200]
#
#             num_view = len(view_list)
#             dim_hvcdn = pow(num_class, num_view)
#             test_inverval = 50
#             # For binary classification tasks, the results are ACC, F1, AUC. For multi-class classification tasks, the results are ACC, F1_weighted, F1_macro.
#             ACC, F1, AUC = train_and_test(data_folder, view_list, protein_self_test_weighted_tr, protein_self_test_weighted_val,
#                                           y_train, y_test,
#                                           num_epoch_pretrain, num_epoch, lr_e_pretrain, lr_e, lr_c, num_class,
#                                           adj_parameter,
#                                           dim_he_list, num_view)
#
#             # Record the ACC value corresponding to each l1_lambda.
#             print(f"l1_lambda: {l1_lambda}, ACC: {ACC}, F1: {F1}, AUC: {AUC}")
#
#             # Update the best ACC value and the corresponding l1_lambda.
#             if ACC > best_ACC:
#                 best_ACC = ACC
#                 best_l1_lambda = l1_lambda
#
#         # Set l1_lambda to the value corresponding to the highest ACC.
#         print(f"Best l1_lambda: {best_l1_lambda}, Best ACC: {best_ACC}")


#disease subtypes classificaiton

ACC_list = []
F1_list = []
AUC_list = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 5-fold
for train_index, test_index in kf.split(samples_protein):
    X_train_combined, X_test_combined = samples_protein[train_index], samples_protein[test_index]
    y_train_combined, y_test_combined = labels[train_index], labels[test_index]

    print("-------------------------")

    l1_lambda = 0.3 # Set to the corresponding best_l1_lambda

    protein_self_test_weighted_tr = X_train_combined.copy()
    protein_self_test_weighted_te = X_test_combined.copy()

    weight_biomarker_number = []

    for i in range(protein_self_test_name.shape[0]):
        for j in range(protein_bulk_name.shape[0]):
            if protein_self_test_name[i, 0] == protein_bulk_name[j, 1]:
                weight_biomarker_number.append([i, j])

    weight_biomarker_number_double = np.array(weight_biomarker_number)


    # The process of feature weighting
    for [i, j] in weight_biomarker_number_double:
        protein_self_test_weighted_tr[:, i] = X_train_combined[:, i] * weight_parameter * S[j, rank_of_disease]
        protein_self_test_weighted_te[:, i] = X_test_combined[:, i] * weight_parameter * S[j, rank_of_disease]
        protein_self_test_weighted_tr[:, i] *= np.exp(-l1_lambda * np.abs(X_train_combined[:, i]))
        protein_self_test_weighted_te[:, i] *= np.exp(-l1_lambda * np.abs(X_test_combined[:, i]))


    data_folder = '3' #The number of subtypes, '2' represents the disease has two subtypes.
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

    if data_folder == '2':
        adj_parameter = 2
        dim_he_list = [200, 200, 100]
    if data_folder == '3':
        adj_parameter = 2
        dim_he_list = [200, 200, 100]
    if data_folder == '5':
        adj_parameter = 10
        dim_he_list = [400, 400, 200]

    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)
    test_inverval = 50

    ACC, F1, AUC = train_and_test(data_folder, view_list, protein_self_test_weighted_tr, protein_self_test_weighted_te, y_train_combined, y_test_combined,
                                  num_epoch_pretrain, num_epoch, lr_e_pretrain, lr_e, lr_c, num_class, adj_parameter,
                                  dim_he_list, num_view)


    print(f"ACC: {ACC}, F1: {F1}, AUC: {AUC}")
    ACC_list.append(ACC)
    F1_list.append(F1)
    AUC_list.append(AUC)

# Calculate the mean and standard deviation.
ACC_mean = np.mean(ACC_list)
F1_mean = np.mean(F1_list)
AUC_mean = np.mean(AUC_list)
ACC_std = np.std(ACC_list)
F1_std = np.std(F1_list)
AUC_std = np.std(AUC_list)

# Print the mean and standard deviation
print(f"Average ACC: {ACC_mean}, Standard Deviation ACC: {ACC_std}")
print(f"Average F1: {F1_mean}, Standard Deviation F1: {F1_std}")
print(f"Average AUC: {AUC_mean}, Standard Deviation AUC: {AUC_std}")

