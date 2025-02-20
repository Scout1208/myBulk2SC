import torch
import torch.nn.functional as F

def train_BulkEncoder(epoch, model, GMVAE_model, max_epochs, optimizer, dataloader, scMus, scLogVars, scPis, device='cuda'):
    model.train()
    model = model.to(device)
    GMVAE_model.eval()
    GMVAE_model = GMVAE_model.to(device)

    for _, (data, _) in enumerate(dataloader):
        data = data.to(device)

        # 使用所有細胞資料的總和作為 bulk 資料
        bulk_data = data.sum(dim=0)
        bulk_data = bulk_data.unsqueeze(0)

        mus, logvars, pis = model(bulk_data)

        mus = mus.squeeze()
        logvars = logvars.squeeze()
        pis = pis.squeeze()

        # 針對 pis_loss，若 scPis 為標量則將其擴展成與 pis 相同的 shape
        mus_loss = F.mse_loss(mus, scMus)
        logvars_loss = F.mse_loss(logvars, scLogVars)
        pis_loss = F.mse_loss(pis, scPis.expand_as(pis))

        loss = mus_loss + logvars_loss + pis_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print("Epoch[{}/{}]: mus_loss:{:.3f}, vars_loss:{:.3f}, pis_loss:{:.3f}".format(
            epoch + 1, max_epochs,
            mus_loss.item(),
            logvars_loss.item(),
            pis_loss.item()))

    if (epoch + 1) % 500 == 0:
        torch.save(model.state_dict(), "saved_files/bulkEncoder_model.pt")
