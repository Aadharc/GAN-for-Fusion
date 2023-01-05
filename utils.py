import torch
import config
from torchvision.utils import save_image

def save_some_examples(mask_feat, gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    masked_feat = mask_feat(x, y)
    mask_x, mask_y = masked_feat[2].to(config.DEVICE), masked_feat[3].to(config.DEVICE)
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(mask_x, mask_y)
        # y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        # y_fake = y_fake  
        save_image(y_fake, folder + f"/Fused_{epoch}.png")
        # save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        save_image(x, folder + f"/VIS_{epoch}.png")
        save_image(y, folder + f"/IR_{epoch}.png")
        # if epoch == 0:
        #     save_image(y, folder + f"/label_{epoch}.png")
            # save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
