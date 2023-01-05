from torch.utils.data import Dataset, DataLoader
import natsort
import os
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

class CustomDataSet(Dataset):
    def __init__(self, main_dir_vis, main_dir_ir, transform):
        self.main_dir_vis = main_dir_vis
        self.main_dir_ir = main_dir_ir
        self.transform = transform
        all_imgs_vis = os.listdir(main_dir_vis)
        all_imgs_ir = os.listdir(main_dir_ir)
        self.total_imgs_vis = natsort.natsorted(all_imgs_vis)
        self.total_imgs_ir = natsort.natsorted(all_imgs_ir)

    def __len__(self):
        return len(self.total_imgs_ir)

    def __getitem__(self, idx):
        vis_img_loc = os.path.join(self.main_dir_vis, self.total_imgs_vis[idx])
        image_vis = Image.open(vis_img_loc)
        tensor_image_vis = self.transform(image_vis)

        ir_img_loc = os.path.join(self.main_dir_ir, self.total_imgs_ir[idx])
        image_ir = Image.open(ir_img_loc)
        tensor_image_ir = self.transform(image_ir)
        return (tensor_image_vis, tensor_image_ir)

if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((512,512),transforms.InterpolationMode.BILINEAR), transforms.ToTensor()])
    dataset = CustomDataSet("data/vis/train/", "data/ir/train/", transform)
    loader = DataLoader(dataset, batch_size=10)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()