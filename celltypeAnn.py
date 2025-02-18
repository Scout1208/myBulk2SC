#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scanpy as sc
import pandas as pd

# 讀取預先處理好的 AnnData 檔案
adata = sc.read('adata_preprocessed.h5ad')

# 建立 cluster 到細胞類型的對照字典
# 請依照實際分群結果與生物學知識修改對照內容
cluster_to_celltype = {
    "0": "Germ cells",
    "1": "Keratinocytes",
    "2": "Erythroid-like",
    "3": "Enterocytes",
    "4": "Fibroblasts",
    "5": "Basal cells",
    "6": "Dendritic cells"
}

# 根據 adata.obs['leiden'] 為每個細胞指定細胞類型
adata.obs['celltype'] = adata.obs['leiden'].map(cluster_to_celltype)

# 利用 UMAP 以細胞類型標註進行視覺化
sc.pl.umap(adata, color=['celltype'], save='_celltype.png')

# 產生條碼與細胞類型對應的 DataFrame
df_mapping = pd.DataFrame({
    "Barcode": adata.obs_names,
    "CellType": adata.obs['celltype']
})

# 輸出為 CSV 與 TSV 檔案
df_mapping.to_csv("barcode_to_celltype.csv", index=False)
df_mapping.to_csv("barcode_to_celltype.tsv", sep='\t', index=False)

# 顯示前幾筆結果作檢查
print(df_mapping.head())
