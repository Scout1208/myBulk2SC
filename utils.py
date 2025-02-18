import scanpy as sc
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse  # 這裡僅用於建立 Namespace 物件，不會解析 CLI 參數

# Set random seed.
np.random.seed(4)
torch.manual_seed(4)

# Cell type mappings for different tissues
CELL_TYPE_MAPPINGS = {
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
        'Hepatocytes': 0, 'Cholangiocytes': 1, 'Hepatic Stellate Cells': 2, 'Kupffer Cells': 3,
        'Liver Sinusoidal Endothelial Cells': 4, 'Portal Fibroblasts': 5, 'Central Vein Endothelial Cells': 6,
        'Periportal Hepatocytes': 7, 'Pericentral Hepatocytes': 8, 'Biliary Epithelial Cells': 9,
        'Liver Progenitor Cells': 10
    },
    "Self": {
        # 自訂 cell type marker
        "Germ cells": 0,
        "Keratinocytes": 1,
        "Erythroid-like": 2,
        "Enterocytes": 3,
        "Fibroblasts": 4,
        "Basal cells": 5,
        "Dendritic cells": 6
    }
}

def get_celltype2int_dict(tissue):
    # 轉換為 cell type -> int 的對應關係 (僅針對已定義的 tissue)
    mapping = CELL_TYPE_MAPPINGS.get(tissue, {})
    # 若 mapping 的值為 int，則直接回傳；否則試著將 string 轉換為 int
    return {k: int(v) for k, v in mapping.items()}

def get_celltype2strint_dict():
    # 針對 "Immune" tissue 的對應（字串形式）
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

def get_liver_celltype2strint_dict():
    mapping_dict = {
        'Hepatocytes': '0', 'Cholangiocytes': '1', 'Hepatic Stellate Cells': '2', 'Kupffer Cells': '3',
        'Liver Sinusoidal Endothelial Cells': '4', 'Portal Fibroblasts': '5', 'Central Vein Endothelial Cells': '6',
        'Periportal Hepatocytes': '7', 'Pericentral Hepatocytes': '8', 'Biliary Epithelial Cells': '9',
        'Liver Progenitor Cells': '10'
    }
    return mapping_dict

def get_self_celltype2strint_dict():
    # 自訂 "Self" 的 mapping, 與 CELL_TYPE_MAPPINGS["Self"] 內容一致
    mapping_dict = {
        "Germ cells": "0",
        "Keratinocytes": "1",
        "Erythroid-like": "2",
        "Enterocytes": "3",
        "Fibroblasts": "4",
        "Basal cells": "5",
        "Dendritic cells": "6"
    }
    return mapping_dict

def get_colormap(tissue):
    color_maps = {
        "Immune": {
            'Naive B cells': 'red', 'Non-classical monocytes': 'black', 'Classical Monocytes': 'orange', 'Natural killer  cells': 'cyan',
            'CD8+ NKT-like cells': 'pink', 'Memory CD4+ T cells': 'magenta', 'Naive CD8+ T cells': 'blue', 'Platelets': 'yellow',
            'Plasmacytoid Dendritic cells': 'lime', 'Macrophages': 'tan', 'Myeloid Dendritic cells': 'green',
            'Effector CD8+ T cells': 'brown', 'Plasma B cells': 'purple', 'Memory B cells': 'darkred', "Naive CD4+ T cells": "darkblue",
            'Neutrophils': 'darkorchid', 'Mast cells': 'darkseagreen'
        },
        "Liver": {
            'Hepatocytes': 'red', 'Cholangiocytes': 'blue', 'Hepatic Stellate Cells': 'green', 'Kupffer Cells': 'purple',
            'Liver Sinusoidal Endothelial Cells': 'cyan', 'Portal Fibroblasts': 'orange', 'Central Vein Endothelial Cells': 'pink',
            'Periportal Hepatocytes': 'brown', 'Pericentral Hepatocytes': 'yellow', 'Biliary Epithelial Cells': 'gray',
            'Liver Progenitor Cells': 'magenta'
        },
        "Self": {
            # 自訂 "Self" 的顏色對應，可依需求調整
            "Germ cells": "blue",
            "Keratinocytes": "red",
            "Erythroid-like": "green",
            "Enterocytes": "purple",
            "Fibroblasts": "orange",
            "Basal cells": "brown",
            "Dendritic cells": "cyan"
        }
    }
    return color_maps.get(tissue, {})

def load_data(data_dir, barcode_path, tissue):
    adata = sc.read_10x_mtx(data_dir, var_names='gene_symbols', cache=True)

    # 讀取 barcode-to-celltype 檔案（假設檔案為 CSV 格式，第一行為標題）
    barcodes_with_labels = pd.read_csv(barcode_path, sep=',', header=None).iloc[1:]
    barcodes_with_labels.columns = ['barcodes', 'labels']
    barcodes_with_labels = barcodes_with_labels[barcodes_with_labels['labels'] != 'Unknown']

    filtered_barcodes = barcodes_with_labels['barcodes'].values
    adata = adata[adata.obs.index.isin(filtered_barcodes)]

    adata.obs['barcodes'] = adata.obs.index
    adata.obs = adata.obs.reset_index(drop=True)
    adata.obs = adata.obs.merge(barcodes_with_labels, left_on='barcodes', right_on='barcodes', how='left')
    adata.X = adata.X.toarray()
    adata.obs.index = adata.obs.index.astype(str)

    # 根據 tissue 選擇對應 mapping
    if tissue.lower() == "immune":
        mapping_dict = get_celltype2strint_dict()
    elif tissue.lower() == "liver":
        mapping_dict = get_liver_celltype2strint_dict()
    elif tissue.lower() == "self":
        mapping_dict = get_self_celltype2strint_dict()
    else:
        raise ValueError(f"Unknown tissue type: {tissue}")

    # 將 labels 替換為 mapping_dict 中定義的字串數字
    adata.obs['labels'] = adata.obs['labels'].replace(mapping_dict)
    adata.obs['labels'] = adata.obs['labels'].astype('category')

    # 使用 .cat.codes 將 categorical 轉為 int
    labels = adata.obs['labels'].cat.codes
    labels = torch.LongTensor(labels.values)

    print("Unique cell types after mapping:", adata.obs['labels'].unique())

    X_tensor = torch.Tensor(adata.X)
    dataset = TensorDataset(X_tensor, labels)
    
    cell_type_fractions = np.unique(labels.numpy(), return_counts=True)[1] / len(labels.numpy())

    return dataset, X_tensor, labels, cell_type_fractions, mapping_dict

def configure(data_dir, barcode_path, tissue):
    dataset, X_tensor, cell_types_tensor, cell_type_fractions, mapping_dict = load_data(
        data_dir=data_dir,
        barcode_path=barcode_path,
        tissue=tissue
    )

    num_cells = X_tensor.shape[0]
    num_genes = X_tensor.shape[1]

    args = argparse.Namespace()  # 建立一個 args 物件
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
    args.color_map = get_colormap(tissue)
    args.K = len(mapping_dict)

    args.cell_type_fractions = torch.FloatTensor(cell_type_fractions)
    args.X_tensor = X_tensor

    print(f"Configuration is complete for tissue: {tissue}")
    return args
