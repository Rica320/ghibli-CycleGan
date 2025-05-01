from utils import LoadData, Discriminator, weights_init, Generator, test, train_epoch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import albumentations 
from albumentations.pytorch import ToTensorV2

trainA = './data/final_trainA/'
trainB = './data/final_trainB/'

transforms = albumentations.Compose(
    [albumentations.Resize(width=256, height=256),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Normalize(mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(transpose_mask=True)],
    additional_targets={"image0": "image"}, 
    is_check_shapes=False)

dataset = LoadData(root_A=[trainA],
    root_B=[trainB],
    transform=transforms)

loader=DataLoader(dataset,batch_size=1,
    shuffle=True, pin_memory=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")
print(torch.__version__)
print(f"Devices available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

disc_A = Discriminator().to(device)
disc_B = Discriminator().to(device)
weights_init(disc_A)
weights_init(disc_B)


gen_A = Generator(img_channels=3, num_residuals=16).to(device)
gen_B = Generator(img_channels=3, num_residuals=16).to(device)
weights_init(gen_A)
weights_init(gen_B)


l1 = nn.L1Loss()
mse = nn.MSELoss()
g_scaler = torch.amp.GradScaler(device=device)
d_scaler = torch.amp.GradScaler(device=device)

lr = 0.00001
opt_disc = torch.optim.Adam(list(disc_A.parameters()) + 
  list(disc_B.parameters()),lr=lr,betas=(0.5, 0.999))
opt_gen = torch.optim.Adam(list(gen_A.parameters()) + 
  list(gen_B.parameters()),lr=lr,betas=(0.5, 0.999))

print("Starting training...")
for epoch in range(10):
    train_epoch(disc_A, disc_B, gen_A, gen_B, loader, opt_disc,
        opt_gen, l1, mse, d_scaler, g_scaler, device)

torch.save(gen_A.state_dict(), "files/gen_black.pth")
torch.save(gen_B.state_dict(), "files/gen_blond.pth")

