import scanpy as sc
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse

# Set random seed.
np.random.seed(4)
# Set torch seed.
torch.manual_seed(4)

def get_celltype2int_dict():
    mapping_dict = {
        'Naive B cells': 0, 'Non-classical monocytes': 1, 'Classical Monocytes': 2, 'Natural killer  cells': 3,
        'CD8+ NKT-like cells': 4, 'Memory CD4+ T cells': 5, 'Naive CD8+ T cells': 6, 'Platelets': 7, 'Pre-B cells': 8,
        'Plasmacytoid Dendritic cells': 9, 'Effector CD4+ T cells': 10, 'Macrophages': 11, 'Myeloid Dendritic cells': 12,
        'Effector CD8+ T cells': 13, 'Plasma B cells': 14, 'Memory B cells': 15, "Naive CD4+ T cells": 16,
        'Progenitor cells': 17, 'γδ-T cells': 18, 'Eosinophils': 19, 'Neutrophils': 20, 'Basophils': 21, 'Mast cells': 22,
        'Intermediate monocytes': 23, 'Megakaryocyte': 24, 'Endothelial': 25, 'Erythroid-like and erythroid precursor cells': 26,
        'HSC/MPP cells': 27, 'Granulocytes': 28, 'ISG expressing immune cells': 29, 'Cancer cells': 30, "Memory CD8+ T cells": 31,
        "Pro-B cells": 32, "Immature B cells": 33
    }
    return mapping_dict

def get_celltype2strint_dict():
    mapping_dict = {
        'Naive B cells': '0', 'Non-classical monocytes': '1', 'Classical Monocytes': '2', 'Natural killer  cells': '3',
        'CD8+ NKT-like cells': '4', 'Memory CD4+ T cells': '5', 'Naive CD8+ T cells': '6', 'Platelets': '7', 'Pre-B cells': '8',
        'Plasmacytoid Dendritic cells': '9', 'Effector CD4+ T cells': '10', 'Macrophages': '11', 'Myeloid Dendritic cells': '12',
        'Effector CD8+ T cells': '13', 'Plasma B cells': '14', 'Memory B cells': '15', "Naive CD4+ T cells": "16",
        'Progenitor cells': '17', 'γδ-T cells': '18', 'Eosinophils': '19', 'Neutrophils': '20', 'Basophils': '21', 'Mast cells': '22',
        'Intermediate monocytes': '23', 'Megakaryocyte': '24', 'Endothelial': '25', 'Erythroid-like and erythroid precursor cells': '26',
        'HSC/MPP cells': '27', 'Granulocytes': '28', 'ISG expressing immune cells': '29', 'Cancer cells': '30', "Memory CD8+ T cells": "31",
        "Pro-B cells": "32", "Immature B cells": "33"
    }
    return mapping_dict

def get_colormap():
    color_map = {
        'Naive B cells': 'red', 'Non-classical monocytes': 'black', 'Classical Monocytes': 'orange', 'Natural killer  cells': 'cyan',
        'CD8+ NKT-like cells': 'pink', 'Memory CD4+ T cells': 'magenta', 'Naive CD8+ T cells': 'blue', 'Platelets': 'yellow', 'Pre-B cells': 'cornflowerblue',
        'Plasmacytoid Dendritic cells': 'lime', 'Effector CD4+ T cells': 'grey', 'Macrophages': 'tan', 'Myeloid Dendritic cells': 'green',
        'Effector CD8+ T cells': 'brown', 'Plasma B cells': 'purple', "Memory B cells": "darkred", "Naive CD4+ T cells": "darkblue",
        'Progenitor cells': 'darkgreen', 'γδ-T cells': 'darkcyan', 'Eosinophils': 'darkolivegreen', 'Neutrophils': 'darkorchid', 'Basophils': 'darkred',
        'Mast cells': 'darkseagreen', 'Intermediate monocytes': 'darkslateblue', 'Megakaryocyte': 'darkslategrey', 'Endothelial': 'darkturquoise',
        'Erythroid-like and erythroid precursor cells': 'darkviolet', 'HSC/MPP cells': 'deeppink', 'Granulocytes': 'deepskyblue',
        'ISG expressing immune cells': 'dimgray', 'Cancer cells': 'dodgerblue', 'Memory CD8+ T cells': 'darkkhaki', 'Pro-B cells': 'darkorange',
        'Immature B cells': 'darkgoldenrod'
    }
    return color_map

def load_data(data_dir, barcode_path):
    adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)

    # 讀取 barcode 與標籤檔案
    barcodes_with_labels = pd.read_csv(barcode_path, sep=',', header=None).iloc[1:]
    barcodes_with_labels.columns = ['barcodes', 'labels']

    # 移除標籤為 'Unknown' 或 NaN 的資料
    barcodes_with_labels = barcodes_with_labels[barcodes_with_labels['labels'] != 'Unknown']

    filtered_barcodes = barcodes_with_labels['barcodes'].values
    adata = adata[adata.obs.index.isin(filtered_barcodes)]

    adata.obs['barcodes'] = adata.obs.index
    adata.obs = adata.obs.reset_index(drop=True)
    adata.obs = adata.obs.merge(barcodes_with_labels, on='barcodes', how='left')
    adata.X = adata.X.toarray()
    adata.obs.index = adata.obs.index.astype(str)

    # 利用 mapping 將文字標籤轉為字串數值型態
    mapping_dict = get_celltype2strint_dict()
    adata.obs['labels'] = adata.obs['labels'].replace(mapping_dict)
    adata.obs['labels'] = adata.obs['labels'].astype('category')

    # 原始標籤 Series
    labels = adata.obs['labels']

    # 建立原始 mapping（可能非連續，例如可能有 0, 5, 6, 12）
    label_to_int = {}
    max_int_label = -1
    for label in labels.unique():
        if label.isdigit():
            int_val = int(label)
            label_to_int[label] = int_val
            max_int_label = max(max_int_label, int_val)
        else:
            if label not in label_to_int:
                max_int_label += 1
                label_to_int[label] = max_int_label

    # 將文字標籤轉成對應整數
    int_labels = labels.map(label_to_int)

    # 重新映射，讓標籤連續從 0 到 (num_unique-1)
    unique_labels = sorted(int_labels.unique())
    new_mapping = {old_label: new_idx for new_idx, old_label in enumerate(unique_labels)}
    int_labels = int_labels.map(new_mapping)

    # 建立反向映射字典，將重新映射的標籤轉回原始標籤
    reverse_mapping = {new_idx: old_label for old_label, new_idx in new_mapping.items()}

    # 轉換為 LongTensor
    labels = torch.LongTensor(int_labels.values)
    print("Re-mapped labels tensor:", labels)

    X_tensor = torch.Tensor(adata.X)
    # 隨機抽取80%資料
    rand_index = np.random.choice(X_tensor.shape[0], int(0.8 * X_tensor.shape[0]), replace=False)
    X_tensor = X_tensor[rand_index]
    labels = labels[rand_index]

    print(f"Shape of X_tensor: {X_tensor.shape}")
    print(f"Shape of labels: {labels.shape}")

    dataset = TensorDataset(X_tensor, labels)

    # 計算每個細胞類型的比例（這裡採用原始 adata.obs 中的標籤，但注意其仍是原始映射字串）
    cell_type_fractions = np.unique(adata.obs['labels'].values, return_counts=True)[1] / len(adata.obs['labels'].values)
    
    # 傳回 mapping_dict 為原始 mapping，不過後續可用 labels 的重新映射結果決定 K
    return dataset, X_tensor, labels, cell_type_fractions, mapping_dict, reverse_mapping

def get_saved_GMM_params(mus_path, vars_path):
    gmm_mus_celltypes = torch.load(mus_path).squeeze().T
    gmm_vars_celltypes = torch.load(vars_path).squeeze().T
    return gmm_mus_celltypes, gmm_vars_celltypes

def configure(data_dir, barcode_path):
    dataset, X_tensor, cell_types_tensor, cell_type_fractions, mapping_dict, reverse_mapping = load_data(
        data_dir=data_dir,
        barcode_path=barcode_path,
    )
    num_cells = X_tensor.shape[0]
    num_genes = X_tensor.shape[1]

    parser = argparse.ArgumentParser(description='Process neural network parameters.')
    args = parser.parse_args()
    args.num_cells = num_cells
    args.hidden_dim = 600
    args.latent_dim = 300
    args.train_GMVAE_epochs = 200
    args.bulk_encoder_epochs = 1000
    args.batch_size = num_cells
    args.input_dim = num_genes

    dataloader = DataLoader(dataset, batch_size=num_cells // 5, shuffle=True, drop_last=True)
    args.dataloader = dataloader
    args.cell_types_tensor = cell_types_tensor

    args.mapping_dict = mapping_dict
    args.color_map = get_colormap()

    # 以重新映射後的標籤決定 K，即唯一標籤的數量
    args.K = len(torch.unique(cell_types_tensor))
    
    # 根據每個類型建立 fraction 向量
    unique_cell_types = np.unique(cell_types_tensor)
    cell_type_fractions_ = []
    cell_type_to_fraction = {cell_type: fraction for cell_type, fraction in zip(unique_cell_types, cell_type_fractions)}
    for i in range(args.K):
        cell_type_fractions_.append(cell_type_to_fraction.get(i, 0))
    args.cell_type_fractions = torch.FloatTensor(np.array(cell_type_fractions_))
    
    print("Cell type fractions:", args.cell_type_fractions)
    print("@@")
    
    args.X_tensor = X_tensor
    # 建立一個 label_map，用於後續可視化（保留原始 mapping）
    label_map = {str(v): k for k, v in mapping_dict.items()}
    args.label_map = label_map
    args.reverse_mapping = reverse_mapping
    print('Configuration is complete.')
    return args
