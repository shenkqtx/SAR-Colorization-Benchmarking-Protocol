import torch

def trim_image(image, L=0, R=2 ** 12 - 1):
    L = torch.Tensor([L]).float().cuda()
    R = torch.Tensor([R]).float().cuda()
    return torch.min(torch.max(image, L), R)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".tif", ".png", ".jpg", ".jpeg"])


