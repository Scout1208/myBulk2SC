import torch
import torch.nn as nn
import argparse
import os
from utils import configure

# Train GMVAE. Refer to train_GMVAE.py for the implementation.
def train_model_GMVAE(max_epochs,
                      dataloader,
                      proportion_tensor,
                      mapping_dict,
                      color_map,
                      model_param_tuple,
                      device='cuda'):
    if os.path.exists('saved_files/GMVAE_mus.pt') and os.path.exists('saved_files/GMVAE_logvars.pt') and os.path.exists('saved_files/GMVAE_pis.pt'):
        print("Pre-trained GMVAE_mus and GMVAE_logvars EXIST. Skipping training.")
        return 0
    else:
        print(f"Pre-trained GMVAE_mus and GMVAE_logvars DO NOT EXIST. Training for {max_epochs} epochs.")
        from models import GaussianMixtureVAE
        from train_GMVAE import train_GMVAE

        input_dim, hidden_dim, latent_dim, K = model_param_tuple
        GMVAE_model = GaussianMixtureVAE(input_dim, hidden_dim, latent_dim, K)
        optimizer = torch.optim.Adam(GMVAE_model.parameters(), lr=1e-3)
        print(f"Using {torch.cuda.device_count()} GPUs!")
        GMVAE_model = nn.DataParallel(GMVAE_model)
        try:
            gmvae_state_dict = torch.load("saved_files/GMVAE_model.pt")
            GMVAE_model.load_state_dict(gmvae_state_dict, strict=True)
            print("Loaded existing GMVAE_model.pt")
        except:
            for m in GMVAE_model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
            print("Initialized GMVAE_model")

        kl_weight = 0.0
        kl_weight_max = 1.0
        losses = []
        
        for epoch in range(0, max_epochs):
            kl_weight_increment = kl_weight_max / (100000)
            if kl_weight < kl_weight_max:
                kl_weight += kl_weight_increment
                kl_weight = min(kl_weight, kl_weight_max)
            total_loss = train_GMVAE(GMVAE_model, epoch, dataloader, optimizer, proportion_tensor, kl_weight, mapping_dict, color_map, max_epochs, device)
            losses.append(total_loss)

def train_model_BulkEncoder(max_epochs,
                            dataloader,
                            model_param_tuple,
                            device='cuda',
                            train_more=False):
    if os.path.exists('saved_files/bulkEncoder_model.pt'):
        if train_more:
            print(f"Pre-trained bulkEncoder_model EXIST. Additionally training for {max_epochs} epochs.")
        else:
            print("Pre-trained bulkEncoder_model EXIST. Skipping training.")
            return 0
    else:
        print(f"Pre-trained bulkEncoder_model DOES NOT exist. Training for {max_epochs} epochs.")

    from models import GaussianMixtureVAE, bulkEncoder
    from train_bulkEncoder import train_BulkEncoder

    scMus = torch.load('saved_files/GMVAE_mus.pt').to(device).detach().requires_grad_(False)
    scLogVars = torch.load('saved_files/GMVAE_logvars.pt').to(device).detach().requires_grad_(False)
    scPis = torch.load('saved_files/GMVAE_pis.pt').to(device).detach().requires_grad_(False)

    input_dim, hidden_dim, latent_dim, K = model_param_tuple
    bulkEncoder_model = bulkEncoder(input_dim, hidden_dim, latent_dim, K)
    
    if os.path.exists('saved_files/bulkEncoder_model.pt'):
        encoder_state_dict = torch.load("saved_files/bulkEncoder_model.pt")
        bulkEncoder_model.load_state_dict(encoder_state_dict, strict=True)
    else:
        for m in bulkEncoder_model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    optimizer = torch.optim.Adam(bulkEncoder_model.parameters(), lr=1e-3)
    from models import GaussianMixtureVAE
    GMVAE_model = GaussianMixtureVAE(input_dim, hidden_dim, latent_dim, K)
    GMVAE_model = nn.DataParallel(GMVAE_model)
    gmvae_state_dict = torch.load("saved_files/GMVAE_model.pt")
    GMVAE_model.load_state_dict(gmvae_state_dict, strict=True)
    bulkEncoder_model = bulkEncoder_model.to(device)

    for epoch in range(0, max_epochs):
        train_BulkEncoder(epoch,
                          bulkEncoder_model,
                          GMVAE_model,
                          max_epochs,
                          optimizer,
                          dataloader,
                          scMus,
                          scLogVars,
                          scPis,
                          device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify tissue type")
    parser.add_argument("tissue", type=str, help="Specify tissue type (e.g., Immune, Liver)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    data_dir = "/Group16T/common/lcy/dslab_lcy/bulk2sc/B2SC/myData/"
    barcode_path = data_dir + 'barcode_to_celltype.csv'
    
    config_args = configure(data_dir, barcode_path, args.tissue)
    print(f"Training for {args.tissue} cell types with {config_args.K} clusters.")

    # 使用 config_args 內的參數
    input_dim = config_args.input_dim
    hidden_dim = config_args.hidden_dim
    latent_dim = config_args.latent_dim
    K = config_args.K

    # 1. Train GMVAE for scMu and scLogVar.
    train_model_GMVAE(
        max_epochs=config_args.train_GMVAE_epochs,
        dataloader=config_args.dataloader,
        proportion_tensor=config_args.cell_type_fractions,
        mapping_dict=config_args.mapping_dict,
        color_map=config_args.color_map,
        model_param_tuple=(input_dim, hidden_dim, latent_dim, K),
        device=device
    )
    
    # 2. Train scDecoder for reconstruction using trained scMu and scLogVar.
    train_model_BulkEncoder(
        max_epochs=config_args.bulk_encoder_epochs,
        dataloader=config_args.dataloader,
        model_param_tuple=(input_dim, hidden_dim, latent_dim, K),
        device=device,
        train_more=False
    )

    # 3. Generate. Refer to generate.py for the implementation.
    from models import GaussianMixtureVAE, bulkEncoder
    from generate import generate

    num_cells = config_args.num_cells
    GMVAE_model = GaussianMixtureVAE(input_dim, hidden_dim, latent_dim, K)
    bulkEncoder_model = bulkEncoder(input_dim, hidden_dim, latent_dim, K)
    
    encoder_state_dict = torch.load("saved_files/bulkEncoder_model.pt")
    gmvae_state_dict = torch.load("saved_files/GMVAE_model.pt")

    bulkEncoder_model.load_state_dict(encoder_state_dict, strict=True)
    GMVAE_model = nn.DataParallel(GMVAE_model)
    gmvae_state_dict = torch.load("saved_files/GMVAE_model.pt")
    GMVAE_model.load_state_dict(gmvae_state_dict, strict=True)

    generate(bulkEncoder_model, GMVAE_model, config_args.dataloader, num_cells, config_args.mapping_dict, config_args.color_map, device=device)
