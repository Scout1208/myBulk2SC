import scanpy as sc
import pandas as pd

# Step 1. 讀取資料 
adata = sc.read_10x_mtx(
    '/Group16T/common/lcy/dslab_lcy/GitRepo/B2SC/raw_gene_bc_matrices/hg19/',  # 根據實際路徑修改
    var_names='gene_symbols',            # 或使用 'gene_ids' 根據檔案內容選擇
    cache=True                           # 可加速重複讀取
)

# 確保基因名稱唯一
adata.var_names_make_unique()

# Step 2. 基本篩選: 篩選細胞與基因
sc.pp.filter_cells(adata, min_genes=200)   # 篩選至少偵測到200個基因的細胞
sc.pp.filter_genes(adata, min_cells=3)       # 篩選至少在3個細胞中表達的基因

# Step 3. 計算 QC 指標（例如線粒體基因比例）
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
# 篩選出線粒體基因比例低於 5% 的細胞
adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :].copy()

# Step 4. 正規化與對數轉換
sc.pp.normalize_total(adata, target_sum=1e4)  # 每個細胞的總 counts 標準化到 10,000
sc.pp.log1p(adata)                            # 對數轉換

# Step 5. 篩選高變基因 (Highly Variable Genes, HVG)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)

# Step 6. 資料縮放 (將每個基因的表達值縮放至均值0, 標準差1)
sc.pp.scale(adata, max_value=10)

# Step 7. PCA 降維
sc.tl.pca(adata, svd_solver='arpack')

# Step 8. 計算鄰近圖 (使用 PCA 結果，預設選擇 10 個鄰近細胞)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

# Step 9. 使用 Leiden 方法進行分群 (可以調整 resolution 參數以影響分群數目)
sc.tl.leiden(adata, resolution=0.8)

# Step 10. UMAP 降維視覺化
sc.tl.umap(adata)
sc.pl.umap(adata, color=['leiden'], save='_leiden_clusters.png')

# Find marker genes
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save='_rank_genes_t.png')
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save='_rank_genes_wil.png')
sc.tl.rank_genes_groups(adata, "leiden", method="logreg", max_iter=1000)
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, save='_rank_genes_log.png')

## ===== 修改處開始 =====
## 新增：建立條碼與細胞型態對應表 (barcode_to_celltype.csv)
## 此處 CellType 採用 Leiden 分群結果，如有需要可根據其他註釋方法進行修改
mapping = pd.DataFrame({
    'Barcode': adata.obs_names,       # 條碼資訊來自於 adata.obs 的索引
    'CellType': adata.obs['leiden']     # 細胞型態以 Leiden 分群結果表示
})
# 儲存為 CSV 檔案，檔名與路徑與 scType.R 保持一致
mapping.to_csv('/Group16T/common/lcy/dslab_lcy/GitRepo/myBulk2SC/raw_gene_bc_matrices/hg19/barcode_to_celltype_Leiden.csv', index=False)
## ===== 修改處結束 =====

# 儲存預處理後的 AnnData 物件
adata.write('/Group16T/common/lcy/dslab_lcy/GitRepo/myBulk2SC/raw_gene_bc_matrices/hg19/adata_preprocessed.h5ad')
