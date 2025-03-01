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

# =============================================================================
# 修改開始：新增擴展 mapping dict（針對不同組織） 
# =============================================================================
# 原有的 mapping dict 為免疫系統 (Immune)，現在擴展增加 Liver 與 Leiden 兩個 mapping。
EXTENDED_MAPPING_DICTS = {
    "Immune": {
        'Naive B cells': 0, 'Non-classical monocytes': 1, 'Classical Monocytes': 2, 'Natural killer  cells': 3,
        'CD8+ NKT-like cells': 4, 'Memory CD4+ T cells': 5, 'Naive CD8+ T cells': 6, 'Platelets': 7, 'Pre-B cells': 8,
        'Plasmacytoid Dendritic cells': 9, 'Effector CD4+ T cells': 10, 'Macrophages': 11, 'Myeloid Dendritic cells': 12,
        'Effector CD8+ T cells': 13, 'Plasma B cells': 14, 'Memory B cells': 15, "Naive CD4+ T cells": 16,
        'Progenitor cells': 17, 'γδ-T cells': 18, 'Eosinophils': 19, 'Neutrophils': 20, 'Basophils': 21, 'Mast cells': 22,
        'Intermediate monocytes': 23, 'Megakaryocyte': 24, 'Endothelial': 25, 'Erythroid-like and erythroid precursor cells': 26,
        'HSC/MPP cells': 27, 'Granulocytes': 28, 'ISG expressing immune cells': 29, 'Cancer cells': 30, "Memory CD8+ T cells": 31,
        "Pro-B cells": 32, "Immature B cells": 33
    },
    "Liver": {
        'Hepatocytes': 0, 
        'Cholangiocytes': 1, 
        'Hepatic Stellate Cells': 2, 
        'Kupffer Cells': 3,
        'Liver Sinusoidal Endothelial Cells': 4, 
        'Portal Fibroblasts': 5, 
        'Central Vein Endothelial Cells': 6,
        'Periportal Hepatocytes': 7, 
        'Pericentral Hepatocytes': 8, 
        'Biliary Epithelial Cells': 9,
        'Liver Progenitor Cells': 10
    },
    "Leiden": {str(i): i for i in range(31)}  # 自動生成 '0':0, '1':1, …, '30':30
}

def get_tissue_mapping_dict(tissue="Immune", as_string=False):
    """
    傳回指定組織的 mapping dict。
    :param tissue: 組織名稱，可選 "Immune", "Liver", "Leiden"
    :param as_string: 若為 True，將 mapping 值轉成字串（除了 Leiden，本身鍵已為字串）
    """
    mapping = EXTENDED_MAPPING_DICTS.get(tissue)
    if mapping is None:
        raise ValueError(f"找不到組織 {tissue} 的 mapping dict")
    if as_string and tissue != "Leiden":
        return {k: str(v) for k, v in mapping.items()}
    return mapping
# =============================================================================
# 修改結束：新增擴展 mapping dict
# =============================================================================

def get_celltype2int_dict():
    # [修改] 使用擴展 mapping dict 中的 Immune mapping
    mapping_dict = EXTENDED_MAPPING_DICTS["Immune"]
    return mapping_dict

def get_celltype2strint_dict():
    # [修改] 使用擴展 mapping dict 中的 Immune mapping，並轉成字串
    mapping_dict = {k: str(v) for k, v in EXTENDED_MAPPING_DICTS["Immune"].items()}
    return mapping_dict

# =============================================================================
# 修改開始：新增擴展 color_map（針對不同組織）
# =============================================================================
EXTENDED_COLOR_MAPS = {
    "Immune": {
        'Naive B cells': 'red', 'Non-classical monocytes': 'black', 'Classical Monocytes': 'orange', 'Natural killer  cells': 'cyan',
        'CD8+ NKT-like cells': 'pink', 'Memory CD4+ T cells': 'magenta', 'Naive CD8+ T cells': 'blue', 'Platelets': 'yellow', 'Pre-B cells': 'cornflowerblue',
        'Plasmacytoid Dendritic cells': 'lime', 'Effector CD4+ T cells': 'grey', 'Macrophages': 'tan', 'Myeloid Dendritic cells': 'green',
        'Effector CD8+ T cells': 'brown', 'Plasma B cells': 'purple', "Memory B cells": "darkred", "Naive CD4+ T cells": "darkblue",
        'Progenitor cells': 'darkgreen', 'γδ-T cells': 'darkcyan', 'Eosinophils': 'darkolivegreen', 'Neutrophils': 'darkorchid',
        'Basophils': 'darkred', 'Mast cells': 'darkseagreen', 'Intermediate monocytes': 'darkslateblue', 'Megakaryocyte': 'darkslategrey',
        'Endothelial': 'darkturquoise', 'Erythroid-like and erythroid precursor cells': 'darkviolet', 'HSC/MPP cells': 'deeppink',
        'Granulocytes': 'deepskyblue', 'ISG expressing immune cells': 'dimgray', 'Cancer cells': 'dodgerblue', 'Memory CD8+ T cells': 'darkkhaki',
        'Pro-B cells': 'darkorange', 'Immature B cells': 'darkgoldenrod'
    },
    "Liver": {
        'Hepatocytes': 'red',
        'Cholangiocytes': 'orange',
        'Hepatic Stellate Cells': 'yellow',
        'Kupffer Cells': 'green',
        'Portal Fibroblasts': 'darkblue',
        'Liver Sinusoidal Endothelial Cells': 'blue',
        'Central Vein Endothelial Cells': 'purple',
        'Periportal Hepatocytes': 'sienna',
        'Pericentral Hepatocytes': 'chocolate',
        'Biliary Epithelial Cells': 'peru',
        'Liver Progenitor Cells': 'pink',
        'Unknown': 'black'
    },
    "Leiden": {
        str(i): color for i, color in zip(range(31), [
            "#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231", "#911eb4", "#46f0f0",
            "#f032e6", "#d2f53c", "#fabebe", "#008080", "#e6beff", "#aa6e28", "#fffac8", 
            "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000080", "#808080", "#FFFFFF", 
            "#000000", "#1f78b4", "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00", 
            "#cab2d6", "#6a3d9a", "#ffff99"
        ])
    }
}

def get_colormap(tissue="Immune"):
    """
    傳回指定 tissue 的 color map。
    :param tissue: 組織名稱，可選 "Immune", "Liver", "Leiden"
    """
    cmap = EXTENDED_COLOR_MAPS.get(tissue)
    if cmap is None:
        raise ValueError(f"找不到組織 {tissue} 的 color map")
    return cmap
# =============================================================================
# 修改結束：新增擴展 color_map
# =============================================================================

# [MODIFIED] 修改 load_data 函數以接受 tissue 參數
def load_data(data_dir, barcode_path, tissue="Immune"):
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
    # [MODIFIED] 使用 tissue 參數來獲取對應 mapping
    mapping_dict = get_tissue_mapping_dict(tissue, as_string=True)
    adata.obs['labels'] = adata.obs['labels'].replace(mapping_dict)
    adata.obs['labels'] = adata.obs['labels'].astype('category')

    # 原始標籤 Series
    labels = adata.obs['labels']

    # 建立原始 mapping（可能非連續，例如可能有 0, 5, 6, 12）
    label_to_int = {}
    max_int_label = -1

    for label in labels.unique():
        # 先轉成字串
        label_str = str(label)
        
        if label_str.isdigit():
            # 若可以轉成整數，就取其 int 值
            int_val = int(label_str)
            label_to_int[label_str] = int_val
            max_int_label = max(max_int_label, int_val)
        else:
            # 否則就視為新的字串標籤
            if label_str not in label_to_int:
                max_int_label += 1
                label_to_int[label_str] = max_int_label

    # 然後在把整個 labels 轉成 int_labels
    int_labels = labels.map(lambda x: label_to_int[str(x)])


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
    
    return dataset, X_tensor, labels, cell_type_fractions, mapping_dict, reverse_mapping

def get_saved_GMM_params(mus_path, vars_path):
    gmm_mus_celltypes = torch.load(mus_path).squeeze().T
    gmm_vars_celltypes = torch.load(vars_path).squeeze().T
    return gmm_mus_celltypes, gmm_vars_celltypes

# [MODIFIED] 修改 configure 函數以解析 tissue 參數
def configure(data_dir, barcode_path):
    parser = argparse.ArgumentParser(description='Process neural network parameters.')
    # 新增 tissue 作為 positional 參數，預設為 "Immune"
    parser.add_argument("tissue", type=str, nargs="?", default="Immune", help="Specify tissue type: Immune, Liver, Leiden")  # [MODIFIED]
    args = parser.parse_args()
    
    # 取得 tissue 參數
    tissue = args.tissue  # [MODIFIED]
    print(f"Using tissue type: {tissue}")  # [MODIFIED]
    
    dataset, X_tensor, cell_types_tensor, cell_type_fractions, mapping_dict, reverse_mapping = load_data(
        data_dir=data_dir,
        barcode_path=barcode_path,
        tissue=tissue  # [MODIFIED]
    )
    num_cells = X_tensor.shape[0]
    num_genes = X_tensor.shape[1]

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
    # [MODIFIED] 根據 tissue 傳入對應的 color map
    args.color_map = get_colormap(tissue)
    
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
