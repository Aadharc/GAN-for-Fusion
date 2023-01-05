import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
# from dataset import MapDataset
from Dataset import CustomDataSet
from Generator import Generator
from Discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image
from ssim import SSIM
from MaskedFeatures import MaskedFeatures
torch.backends.cudnn.benchmark = True


def train_fn(
    mask_feat, disc_ir, disc_vis, gen, loader, opt_disc_ir, opt_disc_vis, opt_gen, l1_loss, bce, ssim, KL, g_scaler, d_scaler_ir,d_scaler_vis
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        masked_feat = mask_feat(x, y)
        mask_x, mask_y = masked_feat[2].to(config.DEVICE), masked_feat[3].to(config.DEVICE)
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(mask_x, mask_y)
            D_real_ir = disc_ir(y, y)
            D_real_loss_ir = bce(D_real_ir, torch.ones_like(D_real_ir))
            D_fake_ir = disc_ir(y, y_fake.detach())
            D_fake_loss_ir = bce(D_fake_ir, torch.zeros_like(D_fake_ir))
            D_loss_ir = (D_real_loss_ir + D_fake_loss_ir) / 2

            D_real_vis = disc_vis(x, x)
            D_real_loss_vis = bce(D_real_vis, torch.ones_like(D_real_vis))
            D_fake_vis = disc_vis(x, y_fake.detach())
            D_fake_loss_vis = bce(D_fake_vis, torch.zeros_like(D_fake_vis))
            D_loss_vis = (D_real_loss_vis + D_fake_loss_vis) / 2

        disc_ir.zero_grad()
        disc_vis.zero_grad()
        d_scaler_ir.scale(D_loss_ir).backward()
        d_scaler_vis.scale(D_loss_vis).backward()
        d_scaler_ir.step(opt_disc_ir)
        d_scaler_vis.step(opt_disc_vis)
        d_scaler_ir.update()
        d_scaler_vis.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake_ir = disc_ir(y, y_fake)
            D_fake_vis = disc_vis(x, y_fake)
            G_fake_loss_ir = bce(D_fake_ir, torch.ones_like(D_fake_ir))
            G_fake_loss_vis = bce(D_fake_vis, torch.ones_like(D_fake_vis))
            L1 = (l1_loss(y_fake, y) + l1_loss(y_fake, x) - l1_loss(masked_feat[0].to(config.DEVICE), masked_feat[1].to(config.DEVICE))) * config.L1_LAMBDA
            G_loss = G_fake_loss_ir + G_fake_loss_vis + L1 + KL(y_fake.clone(),y.clone()) * 10

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real_ir=torch.sigmoid(D_real_ir).mean().item(),
                D_fake_ir=torch.sigmoid(D_fake_ir).mean().item(),
                D_real_vis=torch.sigmoid(D_real_vis).mean().item(),
                D_fake_vis=torch.sigmoid(D_fake_vis).mean().item(),
            )


def main():
    masked_feat = MaskedFeatures(in_chan = 3, features = 8)
    disc_ir = Discriminator(in_channels=3).to(config.DEVICE)
    disc_vis = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=64, features=64).to(config.DEVICE)
    opt_disc_ir = optim.Adam(disc_ir.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_disc_vis = optim.Adam(disc_vis.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    transform = transforms.Compose([transforms.Resize((512,512),transforms.InterpolationMode.BILINEAR), transforms.ToTensor()])
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    ssim = SSIM()
    KL = nn.KLDivLoss()

    # if config.LOAD_MODEL:
    #     load_checkpoint(
    #         config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
    #     )
    #     load_checkpoint(
    #         config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
    #     )

    train_dataset = CustomDataSet(config.TRAIN_DIR_VIS, config.TRAIN_DIR_IR, transform= transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler_ir = torch.cuda.amp.GradScaler()
    d_scaler_vis = torch.cuda.amp.GradScaler()
    val_dataset = CustomDataSet(config.VAL_DIR_VIS, config.VAL_DIR_IR, transform)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=True)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            masked_feat, disc_ir, disc_vis, gen, train_loader, opt_disc_ir, opt_disc_vis, opt_gen, L1_LOSS, BCE, ssim, KL, g_scaler, d_scaler_ir,d_scaler_vis
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc_ir, opt_disc_ir, filename=config.CHECKPOINT_DISC_IR)
            save_checkpoint(disc_vis, opt_disc_vis, filename=config.CHECKPOINT_DISC_VIS)
            save_some_examples(masked_feat, gen, val_loader, epoch, folder="More_data2")


if __name__ == "__main__":
    main()
