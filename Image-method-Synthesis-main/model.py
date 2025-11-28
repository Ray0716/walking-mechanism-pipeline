import torch
import torch.nn as nn
import pytorch_lightning as pl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self,):
        super(UnFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), 16, 8, 8)


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=11, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding="same",),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Linear(8192, latent_dim * 2),)

    def forward(self, x):
        return self.conv_stack(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.inverse_conv_stack = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            UnFlatten(),
            nn.ConvTranspose2d(in_channels=16, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=11, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=2, stride=1, padding=3))

    def forward(self, x):
        return self.inverse_conv_stack(x)


class VAE(pl.LightningModule):
    def __init__(self, latentDim = 25):
        super().__init__()
        self.latent_dim = latentDim

        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.latent_dim)

    def reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        return eps * torch.exp(logvar * 0.5) + mean

    def forward(self, x):
        x = self.encoder(x)
        mean, logvar = x[:, : self.latent_dim], x[:, self.latent_dim :]
        z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        return x, mean, logvar

    def loss_fn(self, recon_x, x, mu, logvar):
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # bce_loss = nn.BCEWithLogitsLoss(reduction='sum')(recon_x, x)
        bce_loss = nn.MSELoss(reduction="sum")(recon_x, x)
        return bce_loss + KLD, bce_loss, KLD

    def training_step(self, batch, batch_idx):
        data, _ = batch
        recon_data, mu, logvar = self.forward(data)

        loss, mse, kld = self.loss_fn(recon_data, data, mu, logvar)

        self.log("train_loss", loss)
        self.log("mse_loss", mse)
        self.log("kld_loss", kld)

        return loss

    def validation_step(self, batch, batch_idx):
        data, _ = batch
        recon_data, mu, logvar = self.forward(data)

        loss, mse, kld = self.loss_fn(recon_data, data, mu, logvar)

        self.log("val_loss", loss, sync_dist=True)
        self.log("val_mse_loss", mse, sync_dist=True)
        self.log("val_kld_loss", kld, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        data, _ = batch
        recon_data, mu, logvar = self.forward(data)
        loss, mse, kld = self.loss_fn(recon_data, data, mu, logvar)
        return loss

    def on_epoch_end(self):
        val_losses = self.trainer.callback_metrics["val_loss"]
        self.log("val_losses", val_losses, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=(0.001))
