import torch
import pandas as pd
import scanpy as sc
from utils import get_celltype2int_dict  # 確保 utils.py 中有這個函數

# 1. 讀取儲存的 cell type 標籤
sampled_celltypes = torch.load("saved_files/PBMC/sampled_celltypes.pt")

# 取得細胞類型對應表（假設 utils.get_celltype2int_dict() 回傳字典）
mapping_dict = get_celltype2int_dict("Immune")
int_to_celltype = {v: k for k, v in mapping_dict.items()}  # 建立反向對應

# 2. 將數字標籤轉換成細胞類型名稱
predicted_cell_types = [int_to_celltype[cell.item()] for cell in sampled_celltypes]

# 3. 讀取生成的 scRNA-seq 數據
generated_data = torch.load("saved_files/PBMC/generated_aggregate_tensor.pt")

# 轉換為 AnnData 格式
adata_generated = sc.AnnData(generated_data.detach().cpu().numpy())

# 4. 將細胞類型標籤加入 AnnData 的 obs 中
adata_generated.obs['cell_type'] = predicted_cell_types

# 5. 執行數據前處理與降維 (PCA + UMAP)
sc.pp.normalize_total(adata_generated, target_sum=1e4)
sc.pp.log1p(adata_generated)
sc.pp.pca(adata_generated)
sc.pp.neighbors(adata_generated)
sc.tl.umap(adata_generated)

# 6. 根據 'cell_type' 上色繪製 UMAP
sc.pl.umap(adata_generated, color='cell_type', title="Generated scRNA-seq Data", save="_generated.png")
print("✅ 已生成 `figures/umap_generated.png` 🎉")
