from start import init_train
from utils.weights import LoadWeights
from PIL import Image
from utils.images import TensorToImage
import torch


# Loading models and the latest weights without loading the dataset
Trainer = init_train("configs/StyleGAN2.json", load_dataset=False)

# Loading custom weights with an inaccurate match
LoadWeights(Trainer, 'weight/StyleGAN2 StyleGAN2 64/weight 79.pth')


z = torch.randn((1, Trainer.z_dim), device=Trainer.device)
img = Trainer.G(z)[0]
image = Image.fromarray(TensorToImage(img.detach().cpu()))
print(z)
image.save('t.png')
