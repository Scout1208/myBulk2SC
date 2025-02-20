# 載入必要的函式庫
lapply(c("dplyr", "Seurat", "HGNChelper", "openxlsx", "Matrix"), library, character.only = TRUE)

# 載入 scType 的函式
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/gene_sets_prepare.R")
source("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/R/sctype_score_.R")

### Part 1: 基於 myData 的基本前處理與個別細胞細胞類型得分計算

# 1. 從 .mtx 檔案讀取 myData 數據（需同時提供 barcodes 與 features 檔案）
expr_matrix <- readMM("/Group16T/common/lcy/dslab_lcy/GitRepo/myBulk2SC/raw_gene_bc_matrices/hg19/matrix.mtx")
barcodes <- read.delim("/Group16T/common/lcy/dslab_lcy/GitRepo/myBulk2SC/raw_gene_bc_matrices/hg19/barcodes.tsv", 
                       header = FALSE, stringsAsFactors = FALSE)
features <- read.delim("/Group16T/common/lcy/dslab_lcy/GitRepo/myBulk2SC/raw_gene_bc_matrices/hg19/genes.tsv", 
                       header = FALSE, stringsAsFactors = FALSE)

# 指定矩陣的行名稱與列名稱
# 假設 features 檔案第二欄為基因名稱，barcodes 檔案第一欄為細胞條碼
# 添加一個後綴來確保基因名稱唯一
features$V2 <- make.unique(features$V2)

# 再次設定 rownames
rownames(expr_matrix) <- features$V2  
colnames(expr_matrix) <- barcodes$V1

# 2. 建立 Seurat 物件
seurat_obj <- CreateSeuratObject(counts = expr_matrix, project = "ExampleMTX", 
                                 min.cells = 3, min.features = 200)

# 3. 基本前處理：標準化、挑選高變異基因、縮放、PCA 與 UMAP
seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = 10000)
seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000)
seurat_obj <- ScaleData(seurat_obj, features = rownames(seurat_obj))
seurat_obj <- RunPCA(seurat_obj, features = VariableFeatures(object = seurat_obj))
seurat_obj <- RunUMAP(seurat_obj, dims = 1:10)
DimPlot(seurat_obj, reduction = "umap")

# 4. 取得正向與反向標記基因集（以 "Immune system" 為例，使用 ScTypeDB_short）
gs_list <- gene_sets_prepare("https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_short.xlsx", 
                             "Immune system")

# 5. 使用 scType 進行個別細胞層級的細胞類型得分計算
sc_data <- as.matrix(GetAssayData(seurat_obj, slot = "data"))
es.max <- sctype_score(
  scRNAseqData = sc_data, 
  scaled = TRUE, 
  gs = gs_list$gs_positive, 
  gs2 = gs_list$gs_negative
)
# 檢查前幾個細胞的得分矩陣
print(head(es.max))

### Part 2: 基於 myData 的聚類及依聚類整合後的細胞類型預測流程
#（原本以 PBMC 為例，此處改為使用 myData）

# 可選：進行額外的 QC 與聚類分析（若需要重新檢查 QC 可執行下列步驟）
seurat_obj[["percent.mt"]] <- PercentageFeatureSet(seurat_obj, pattern = "^MT-")
seurat_obj <- NormalizeData(seurat_obj, normalization.method = "LogNormalize", scale.factor = 10000)
seurat_obj <- FindVariableFeatures(seurat_obj, selection.method = "vst", nfeatures = 2000)
seurat_obj <- ScaleData(seurat_obj, features = rownames(seurat_obj))
seurat_obj <- RunPCA(seurat_obj, features = VariableFeatures(object = seurat_obj))
ElbowPlot(seurat_obj)

# 執行聚類流程
seurat_obj <- FindNeighbors(seurat_obj, dims = 1:10)
seurat_obj <- FindClusters(seurat_obj, resolution = 0.8)
seurat_obj <- RunUMAP(seurat_obj, dims = 1:10)
DimPlot(seurat_obj, reduction = "umap")

# 讀取完整數據庫並設定組織類型（以 "Liver" 為例，此處可依實際需求修改）
db_ <- "https://raw.githubusercontent.com/IanevskiAleksandr/sc-type/master/ScTypeDB_full.xlsx"
tissue <- "Immune system"
gs_list <- gene_sets_prepare(db_, tissue)

# 取得聚類後 Seurat 物件中標準化後的數據矩陣
sc_data_mydata <- as.matrix(GetAssayData(seurat_obj, slot = "data"))

# 進行細胞類型得分計算（聚類層面）
es.max_mydata <- sctype_score(
  scRNAseqData = sc_data_mydata, 
  scaled = TRUE, 
  gs = gs_list$gs_positive, 
  gs2 = gs_list$gs_negative
)

# 聚類層面合併得分並選取最佳細胞類型
cL_results <- do.call("rbind", lapply(unique(seurat_obj@meta.data$seurat_clusters), function(cl) {
  es.max.cl <- sort(rowSums(es.max_mydata[, rownames(seurat_obj@meta.data[seurat_obj@meta.data$seurat_clusters == cl, ])]), 
                    decreasing = TRUE)
  head(data.frame(cluster = cl, type = names(es.max.cl), scores = es.max.cl, 
                  ncells = sum(seurat_obj@meta.data$seurat_clusters == cl)), 10)
}))
sctype_scores <- cL_results %>% group_by(cluster) %>% top_n(n = 1, wt = scores)

# 將低信心的聚類標記為 "Unknown"
sctype_scores$type[as.numeric(as.character(sctype_scores$scores)) < sctype_scores$ncells/4] <- "Unknown"
print(sctype_scores[,1:3])

# 將細胞類型結果加入 seurat_obj 物件的 meta.data
seurat_obj@meta.data$customclassif <- ""
for(j in unique(sctype_scores$cluster)){
  cl_type <- sctype_scores[sctype_scores$cluster == j, ]
  seurat_obj@meta.data$customclassif[seurat_obj@meta.data$seurat_clusters == j] <- as.character(cl_type$type[1])
}

# 以細胞類型標註結果繪製 UMAP
DimPlot(seurat_obj, reduction = "umap", label = TRUE, repel = TRUE, group.by = 'customclassif')
png("/Group16T/common/lcy/dslab_lcy/GitRepo/myBulk2SC/raw_gene_bc_matrices/hg19/scRNA_UMAP.png", width=1000, height=800)
DimPlot(seurat_obj, reduction = "umap", label = TRUE, repel = TRUE, group.by = 'customclassif')
dev.off()
# 輸出條碼與細胞類型對應表（CSV 與 TSV 格式）
mapping <- data.frame(Barcode = rownames(seurat_obj@meta.data), 
                      CellType = seurat_obj@meta.data$customclassif)
head(mapping)
write.table(mapping, file = "/Group16T/common/lcy/dslab_lcy/GitRepo/myBulk2SC/raw_gene_bc_matrices/hg19/barcode_to_celltype.csv", sep = ",", 
            row.names = FALSE, quote = FALSE)
write.table(mapping, file = "/Group16T/common/lcy/dslab_lcy/GitRepo/myBulk2SC/raw_gene_bc_matrices/hg19/barcode_to_celltype.tsv", sep = "\t", 
            row.names = FALSE, quote = FALSE)
