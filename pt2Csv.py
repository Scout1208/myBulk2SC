import torch
import pandas as pd
from utils import get_celltype2int_dict  # 確保 utils.py 有這個函數
import scanpy as sc
# 讀取 `sampled_celltypes.pt`
sampled_celltypes = torch.load("saved_files/PBMC_real/sampled_celltypes.pt")

# 取得 cell type 對應表
mapping_dict = get_celltype2int_dict()
int_to_celltype = {v: k for k, v in mapping_dict.items()}  # 反向對應

# 轉換數字標籤為細胞類型名稱
predicted_cell_types = [int_to_celltype[cell.item()] for cell in sampled_celltypes]

# 存成 CSV
df = pd.DataFrame({"Predicted_Cell_Type": predicted_cell_types})
df.to_csv("saved_files/predicted_cell_types.csv", index=False)

print("✅ 細胞類型已儲存到 `saved_files/predicted_cell_types.csv` 🎉")

# 讀取 `generated_aggregate_tensor.pt`
generated_data = torch.load("saved_files/PBMC_real/generated_aggregate_tensor.pt")

# 轉換成 DataFrame
df = pd.DataFrame(generated_data.detach().cpu().numpy())

# 存成 CSV
df.to_csv("saved_files/generated_scRNAseq.csv", index=False)

print("✅ 預測的 scRNA-seq 數據已儲存到 `saved_files/generated_scRNAseq.csv` 🎉")

