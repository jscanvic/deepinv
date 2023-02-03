import torch
import deepinv as dinv
from torchvision import datasets, transforms

# base train dataset
transform_data = transforms.Compose([transforms.ToTensor()])

G = 1  # number of operators
max_datapoints = 1e7
num_workers = 4  # set to 0 if using cpu

# problem
problem = 'denoising'
dataset = 'MNIST'
dir = f'../datasets/MNIST/{problem}/G{G}/'


if dataset == 'MNIST':
    data_train = datasets.MNIST(root='../datasets/', train=True, transform=transform_data, download=True)
    data_test = datasets.MNIST(root='../datasets/', train=False, transform=transform_data)

elif dataset =='CelebA':
    data_train = datasets.CelebA(root='../datasets/', split='train', transform=transform_data, download=True)
    data_test = datasets.CelebA(root='../datasets/', split='test', transform=transform_data)

elif dataset =='FashionMNIST':
    data_train = datasets.FashionMNIST(root='../datasets/', train=True, transform=transform_data, download=True)
    data_test = datasets.FashionMNIST(root='../datasets/', train=False, transform=transform_data)


x = data_train[0]
im_size = x[0].shape if isinstance(x, list) or isinstance(x, tuple) else x.shape

physics = []
for g in range(G):
    if problem == 'CS':
        p = dinv.physics.CompressedSensing(m=300, img_shape=im_size, device=dinv.device)
    elif problem == 'onebitCS':
        p = dinv.physics.CompressedSensing(m=300, img_shape=im_size, device=dinv.device)
        p.sensor_model = lambda x: torch.sign(x)
    elif problem == 'inpainting':
        p = dinv.physics.Inpainting(tensor_size=im_size, mask=.5, device=dinv.device)
    elif problem == 'blind_deblur':
        p = dinv.physics.BlindBlur(kernel_size=11)
    elif problem == 'denoising':
        p = dinv.physics.Denoising(sigma=.2)
    elif problem == 'CT':
        p = dinv.physics.CT(img_width=im_size[-1], views=30)
    elif problem == 'deblur':
        p = dinv.physics.Blur(dinv.physics.blur.gaussian_blur(sigma=(1, .5)), device=dinv.device)
    else:
        raise Exception("The inverse problem chosen doesn't exist")

    # p.sensor_model = lambda x: torch.sign(x)
    physics.append(p)

# generate paired dataset
dinv.datasets.generate_dataset(train_dataset=data_train, test_dataset=data_test,
                               physics=physics, device=dinv.device, save_dir=dir, max_datapoints=max_datapoints,
                               num_workers=num_workers)