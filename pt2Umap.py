import argparse  # [MODIFIED] 新增命令列參數解析
import torch
import pandas as pd
import scanpy as sc
from utils import get_tissue_mapping_dict, get_colormap  # [MODIFIED] 改用 get_tissue_mapping_dict
import numpy as np
import louvain
import igraph
import random

# 固定隨機種子
seed_value = 0
np.random.seed(seed_value)
random.seed(seed_value)
sc.settings.seed = seed_value  # 一併設定 scanpy 的隨機種子

sc.settings.figdir = "/Group16T/common/lcy/dslab_lcy/GitRepo/myBulk2SC/saved_files/PBMC_real_Leiden/"

# [MODIFIED] 解析命令列參數，指定 tissue type，預設為 "Immune"
parser = argparse.ArgumentParser(description="Specify tissue type for UMAP generation")
parser.add_argument("tissue", type=str, nargs="?", default="Immune", help="Tissue type: Immune, Liver, or Leiden")
args = parser.parse_args()
tissue = args.tissue
print(f"Using tissue type: {tissue}")

# 1. 讀取儲存的 cell type 標籤
sampled_celltypes = torch.load("/Group16T/common/lcy/dslab_lcy/GitRepo/myBulk2SC/saved_files/PBMC_real_Leiden/sampled_celltypes.pt")

# [MODIFIED] 取得對應 tissue 的 mapping dict
mapping_dict = get_tissue_mapping_dict(tissue, as_string=False)
int_to_celltype = {v: k for k, v in mapping_dict.items()}  # 建立反向對應

# 2. 將數字標籤轉換成細胞類型名稱
predicted_cell_types = [int_to_celltype[cell.item()] for cell in sampled_celltypes]

# 3. 讀取生成的 scRNA-seq 數據
generated_data = torch.load("/Group16T/common/lcy/dslab_lcy/GitRepo/myBulk2SC/saved_files/PBMC_real_Leiden/generated_aggregate_tensor.pt")

data_array = generated_data.detach().cpu().numpy()
print("最小值:", np.min(data_array))
print("最大值:", np.max(data_array))
print("均值:", np.mean(data_array))
print("標準差:", np.std(data_array))

# 轉換為 AnnData 格式
adata_generated = sc.AnnData(generated_data.detach().cpu().numpy())

# 4. 將細胞類型標籤加入 AnnData 的 obs 中
adata_generated.obs['cell_type'] = predicted_cell_types

# 5. 過濾掉總 counts 為零的細胞
adata_generated = adata_generated[adata_generated.X.sum(axis=1) > 0, :]

# 6. 執行數據前處理與降維 (Normalization, log1p, PCA, UMAP)
sc.pp.normalize_total(adata_generated, target_sum=1e4)
sc.pp.log1p(adata_generated)
sc.pp.highly_variable_genes(adata_generated, n_top_genes=2000, subset=True)
sc.pp.scale(adata_generated, max_value=10)
sc.pp.pca(adata_generated, svd_solver='arpack')
sc.pp.neighbors(adata_generated, n_neighbors=10, n_pcs=40,random_state=0)
sc.tl.leiden(adata_generated, resolution=0.8,random_state=0)
# sc.tl.louvain(adata_generated, resolution=0.8) #scType
sc.tl.umap(adata_generated,random_state=0)

# 7. 設定調色盤，使得 UMAP 圖符合對應 tissue 的 color map
present_types = adata_generated.obs['cell_type'].unique().tolist()

# [MODIFIED] 根據 tissue 取得對應 color map，並過濾只保留出現的細胞類型顏色
cmap = get_colormap(tissue)
filtered_palette = [cmap[ct] for ct in present_types if ct in cmap]

adata_generated.obs['cell_type'] = pd.Categorical(adata_generated.obs['cell_type'], categories=present_types)

sc.pl.umap(adata_generated, color='cell_type', title="Generated scRNA-seq Data", palette=filtered_palette, save="_generated.png")

print("✅ 已生成 `/Group16T/common/lcy/dslab_lcy/GitRepo/myBulk2SC/saved_files/PBMC_real_Leiden/umap_generated.png` 🎉")
adata_generated.write('/Group16T/common/lcy/dslab_lcy/GitRepo/myBulk2SC/saved_files/PBMC_real_Leiden/adata_preprocessed_Leiden.h5ad')