#===========================================
# 0. 讀取 command line 參數，指定細胞特異性基因集
#===========================================
args <- commandArgs(trailingOnly = TRUE)
if(length(args) == 0){
  stop("請提供細胞特異性基因集，例如: 'Immune system' 或 'Liver'")
}
tissue <- args[1]
cat("使用細胞特異性基因集: ", tissue, "\n")

#===========================================
# 1. 載入必要套件與函數
#===========================================
lapply(c("dplyr", "Seurat", "HGNChelper", "openxlsx"), library, character.only = TRUE)

# 載入 sc-type 相關的基因集準備與細胞型態打分函數
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/gene_sets_prepare.R")
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/sctype_score_.R")

#===========================================
# 2. 使用範例資料進行 sc-type 細胞型態打分
#===========================================
# 從內建資料庫取得細胞型態特異性基因集，根據 command line 參數設定 (例如 "Immune system" 或 "Liver")
gs_list = gene_sets_prepare("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_short.xlsx", 
                             tissue)

# 讀取範例 scRNA-seq 矩陣資料
scRNAseqData = readRDS(gzcon(url('https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/exampleData.RDS')))
# 對範例資料計算細胞型態打分
es.max = sctype_score(scRNAseqData = scRNAseqData, scaled = TRUE, 
                      gs = gs_list$gs_positive, gs2 = gs_list$gs_negative)

# 檢視打分結果
print(head(es.max))

#===========================================
# 3. 使用 PBMC 資料進行分析
#===========================================
# 載入 10X Genomics PBMC 資料 (請根據您的實際路徑調整 data.dir)
pbmc.data <- Read10X(data.dir = "/Group16T/common/lcy/dslab_lcy/GitRepo/B2SC/raw_gene_bc_matrices/hg19/")

# 建立 Seurat 物件 (設定 min.cells 與 min.features)
pbmc <- CreateSeuratObject(counts = pbmc.data, project = "pbmc1k", min.cells = 3, min.features = 200)

# 計算粒線體基因比例
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")
# 若需要，可依 QC 指標篩選細胞 (以下程式碼為範例，可根據需求調整)
# pbmc <- subset(pbmc, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)

# 標準化資料及尋找變異基因
pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize", scale.factor = 10000)
pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)

# 資料縮放與 PCA 分析
pbmc <- ScaleData(pbmc, features = rownames(pbmc))
pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))

# 繪製 Elbow Plot (非互動式環境下可能會產生 Rplots.pdf)
ElbowPlot(pbmc)

# 建立鄰近圖、群集及 UMAP 可視化
pbmc <- FindNeighbors(pbmc, dims = 1:40, k.param = 10)
pbmc <- FindClusters(pbmc, resolution = 0.8)
pbmc <- RunUMAP(pbmc, dims = 1:40)
DimPlot(pbmc, reduction = "umap")

#===========================================
# 4. 使用 sc-type 對 PBMC 資料進行細胞型態打分與註釋
#===========================================
# 載入完整資料庫檔案 (ScTypeDB_full.xlsx) 與設定組織類型（使用 command line 參數 tissue）
db_ = "https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_full.xlsx"
# tissue 參數已從 command line 讀入
gs_list = gene_sets_prepare(db_, tissue)

# 取得 Seurat 物件中 RNA 資料經縮放後的矩陣
scaled_data <- GetAssayData(pbmc, assay = "RNA", slot = "scale.data")

# 利用 sctype_score 進行細胞型態打分
es.max = sctype_score(scRNAseqData = scaled_data, scaled = TRUE, 
                      gs = gs_list$gs_positive, gs2 = gs_list$gs_negative)

# 根據群集合併打分結果，選出每個群集中分數最高的細胞型態
cL_results = do.call("rbind", lapply(unique(pbmc@meta.data$seurat_clusters), function(cl){
  es.max.cl = sort(rowSums(es.max[, rownames(pbmc@meta.data[pbmc@meta.data$seurat_clusters == cl, ])]), 
                   decreasing = TRUE)
  head(data.frame(cluster = cl, type = names(es.max.cl), scores = es.max.cl, 
                  ncells = sum(pbmc@meta.data$seurat_clusters == cl)), 10)
}))

sctype_scores = cL_results %>% group_by(cluster) %>% top_n(n = 1, wt = scores)

# 將信心較低的群集標記為 "Unknown"
sctype_scores$type[as.numeric(as.character(sctype_scores$scores)) < sctype_scores$ncells/4] = "Unknown"
print(sctype_scores[, 1:3])

# 將註釋結果存入 Seurat 物件的 meta.data 中
pbmc@meta.data$customclassif = ""
for(j in unique(sctype_scores$cluster)){
  cl_type = sctype_scores[sctype_scores$cluster == j, ]
  pbmc@meta.data$customclassif[pbmc@meta.data$seurat_clusters == j] = as.character(cl_type$type[1])
}

#===========================================
# 5. 自訂顏色並將 UMAP 圖依據自訂 mapping 進行上色
#===========================================
custom_colors <- c(
  "Naive B cells" = "red", 
  "Non-classical monocytes" = "black", 
  "Classical Monocytes" = "orange", 
  "Natural killer  cells" = "cyan",
  "CD8+ NKT-like cells" = "pink", 
  "Memory CD4+ T cells" = "magenta", 
  "Naive CD8+ T cells" = "blue", 
  "Platelets" = "yellow", 
  "Pre-B cells" = "cornflowerblue",
  "Plasmacytoid Dendritic cells" = "lime", 
  "Effector CD4+ T cells" = "grey", 
  "Macrophages" = "tan", 
  "Myeloid Dendritic cells" = "green",
  "Effector CD8+ T cells" = "brown", 
  "Plasma B cells" = "purple", 
  "Memory B cells" = "darkred", 
  "Naive CD4+ T cells" = "darkblue",
  "Progenitor cells" = "darkgreen", 
  "γδ-T cells" = "darkcyan", 
  "Eosinophils" = "darkolivegreen", 
  "Neutrophils" = "darkorchid", 
  "Basophils" = "darkred",
  "Mast cells" = "darkseagreen", 
  "Intermediate monocytes" = "darkslateblue", 
  "Megakaryocyte" = "darkslategrey", 
  "Endothelial" = "darkturquoise",
  "Erythroid-like and erythroid precursor cells" = "darkviolet", 
  "HSC/MPP cells" = "deeppink", 
  "Granulocytes" = "deepskyblue",
  "ISG expressing immune cells" = "dimgray", 
  "Cancer cells" = "dodgerblue", 
  "Memory CD8+ T cells" = "darkkhaki", 
  "Pro-B cells" = "darkorange",
  "Immature B cells" = "darkgoldenrod"
)

DimPlot(pbmc, reduction = "umap", label = TRUE, repel = TRUE, 
        group.by = 'customclassif', cols = custom_colors)

#===========================================
# 6. 建立條碼與細胞型態對應表並儲存 (CSV 與 TSV)
#===========================================
barcodes <- read.table("/Group16T/common/lcy/dslab_lcy/GitRepo/B2SC/raw_gene_bc_matrices/hg19/barcodes.tsv", 
                       header = TRUE, sep = "\t")
seurat_barcodes <- rownames(pbmc)
print(all(barcodes$barcodes %in% seurat_barcodes))

mapping <- data.frame(
  Barcode = rownames(pbmc@meta.data),
  CellType = pbmc@meta.data$customclassif
)
print(head(mapping))

write.table(mapping, file = "/Group16T/common/lcy/dslab_lcy/GitRepo/B2SC/raw_gene_bc_matrices/hg19/barcode_to_celltype.csv", 
            sep = ",", row.names = FALSE, quote = FALSE)
write.table(mapping, file = "/Group16T/common/lcy/dslab_lcy/GitRepo/B2SC/raw_gene_bc_matrices/hg19/barcode_to_celltype.tsv", 
            sep = "\t", row.names = FALSE, quote = FALSE)
