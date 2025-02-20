import torch
import umap
import matplotlib.pyplot as plt
import numpy as np

def generate_(encoder, GMVAE_model, dataloader, device='cuda'):
    generated_list = []
    labels_list = []
    # Generate one cell per batch.
    for _, (data, labels) in enumerate(dataloader):
        data = data.to(device)
        bulk_data = data.sum(dim=0)
        bulk_data = bulk_data.unsqueeze(0)
        # Forward pass
        mus, logvars, pis = encoder(bulk_data)
        mus = mus.squeeze()
        logvars = logvars.squeeze()
        pis = pis.squeeze()
        generated, k = GMVAE_model.module.decode_bulk(mus, logvars, pis)
        generated_list.append(generated)
        labels_list.append(k.item())
    
    generated_tensor = torch.stack(generated_list)
    return generated_tensor, labels_list

def generate(encoder, GMVAE_model, dataloader, num_cells, mapping_dict, reverse_mapping, color_map, device='cuda'):
    encoder.eval()
    GMVAE_model.eval()
    encoder = encoder.to(device)
    GMVAE_model = GMVAE_model.to(device)
        
    generated_aggregate = []
    sampled_celltypes = []
    
    print(f"Generating {num_cells} cells...")

    for i in range(num_cells):
        if (i + 1) % 100 == 0:
            print(f"Generating {i + 1}th cell...")
        gt, label = generate_(encoder, GMVAE_model, dataloader, device=device)
        generated_aggregate.append(gt)
        for l in label:
            sampled_celltypes.append(l)
        
        if (i + 1) % 500 == 0 or (i + 1) == num_cells:
            generated_aggregate_tensor = torch.stack(generated_aggregate)
            generated_aggregate_tensor = generated_aggregate_tensor.squeeze().cpu()
            sampled_celltypes_tensor = torch.LongTensor(sampled_celltypes)
            input_dim = generated_aggregate_tensor.shape[-1]
            generated_aggregate_tensor = generated_aggregate_tensor.reshape(-1, input_dim)
            torch.save(generated_aggregate_tensor, f"saved_files/generated_aggregate_tensor.pt")
            torch.save(sampled_celltypes_tensor, f"saved_files/sampled_celltypes.pt")
            print(f"{i + 1}th generated_aggregate_tensor saved.")

    generated_aggregate_tensor = torch.load("saved_files/generated_aggregate_tensor.pt")
    sampled_celltypes = torch.load("saved_files/sampled_celltypes.pt")
    sampled_celltypes = torch.LongTensor(sampled_celltypes)
    
    # 利用 reverse_mapping 將重新映射的標籤轉換回原始 mapping 的數字
    original_labels = [reverse_mapping[label.item()] for label in sampled_celltypes]
    original_labels_tensor = torch.LongTensor(original_labels)
    
    print("Generated cell type proportion (original mapping):")
    unique_labels, counts = np.unique(original_labels_tensor.numpy(), return_counts=True)
    print(unique_labels)
    print(counts / len(original_labels_tensor))
    
    # 你也可以做其他可視化或輸出處理
