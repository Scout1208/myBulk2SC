import torch
import pandas as pd
import scanpy as sc
from utils import get_celltype2int_dict, get_colormap  # 確保 utils.py 中有這些函數
import numpy as np

# 1. 讀取儲存的 cell type 標籤
sampled_celltypes = torch.load("saved_files/PBMC_real/sampled_celltypes.pt")

# 取得細胞類型對應表（假設 get_celltype2int_dict() 回傳字典）
mapping_dict = get_celltype2int_dict()
int_to_celltype = {v: k for k, v in mapping_dict.items()}  # 建立反向對應

# 2. 將數字標籤轉換成細胞類型名稱
predicted_cell_types = [int_to_celltype[cell.item()] for cell in sampled_celltypes]

# 3. 讀取生成的 scRNA-seq 數據
generated_data = torch.load("saved_files/PBMC_real/generated_aggregate_tensor.pt")

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
sc.pp.pca(adata_generated)
sc.pp.neighbors(adata_generated)
sc.tl.umap(adata_generated)

# 7. 設定調色盤，使得 UMAP 圖符合原始 color_map
# 取得資料中實際出現的細胞類型
present_types = adata_generated.obs['cell_type'].unique().tolist()

# 過濾 color_map，只保留出現的細胞類型顏色
filtered_palette = [get_colormap()[ct] for ct in present_types]

# 將 cell_type 設定為 categorical 並只包含出現的細胞類型
adata_generated.obs['cell_type'] = pd.Categorical(adata_generated.obs['cell_type'], categories=present_types)

# 畫 UMAP 時傳入 filtered_palette
sc.pl.umap(adata_generated, color='cell_type', title="Generated scRNA-seq Data", palette=filtered_palette, save="_generated.png")

print("✅ 已生成 `figures/umap_generated.png` 🎉")
