import torch
import matplotlib.pyplot as plt
import os
from U_model.unet import *  # ⚠️ 修改为你的实际路径
import utils
import cv2
from dataset_RGB import *

import torch.nn.functional as funct
###########################################################
# 1. 找到模型中“最浅层”的 EDFFN
###########################################################
def find_first_edffn(model):
    for name, module in model.named_modules():
        if isinstance(module, EDFFN):
            print("[Found EDFFN] ->", name)
            return name
    raise ValueError("No EDFFN found in the model!")


###########################################################
# 2. 根据路径字符串获取 module 对象
###########################################################
def get_module_by_path(model, path: str):
    module = model
    for part in path.split("."):
        module = getattr(module, part)
    return module


###########################################################
# 3. 注册 hook（只注册最浅层的 EDFFN）
###########################################################
activations = {}

def save_activation(name):
    def hook(module, inp, out):
        activations[name] = out.detach().cpu()
    return hook


def register_first_edffn_hooks(model):
    path = find_first_edffn(model)  # 找最浅层的
    edffn = get_module_by_path(model, path)
    print("[Hook Registered On] ->", path)

    # ⭐ 关键：抓取三段激活
    edffn.project_out.register_forward_hook(save_activation("x_main"))
    edffn.low_freq_conv.register_forward_hook(save_activation("low_freq"))
    edffn.high_freq_conv.register_forward_hook(save_activation("high_freq"))

    return path


###########################################################
# 4. 可视化函数
###########################################################
def save_map(tensor, name, save_dir="./vis"):
    os.makedirs(save_dir, exist_ok=True)
    fmap = tensor[0, 0].numpy()  # 取 batch 0，channel 0

    plt.figure(figsize=(4, 4))
    plt.imshow(fmap, cmap='gray')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=300)
    plt.close()


###########################################################
# 5. 使用范例 (你直接改模型加载 & 输入图像)
###########################################################
if __name__ == "__main__":
    # 加载你的模型
    model = Restoration(3, 6, 3, None).cuda()  # ⚠️换成你的模型类
    utils.load_checkpoint(model, "TFM-UNet-512/models/AHDINet/model_epoch_280.pth")
    model.eval()

    # 注册最浅层 EDFFN hook
    first_edffn_path = register_first_edffn_hooks(model)

    # 你的输入图像
    img_path = '/mnt/afs/users/fandawei/data/event-deblur/GOPRO_AHDINet/test/blur/GOPR0384_11_00/0000.png'
    event_path = '/mnt/afs/users/fandawei/data/event-deblur/GOPRO_AHDINet/test/event/GOPR0384_11_00/0000.npz'
    blur_img = cv2.imread(img_path)
    blur_img = np.float32(blur_img) / 255.0

    event = np.load(event_path)
    event_window = np.stack((event['t'],event['x'],event['y'],event['p']),axis=1)
    event_div_tensor = binary_events_to_voxel_grid(event_window,
                                             num_bins=6,
                                             width=1280,
                                             height=720)
    event_frame = np.float32(event_div_tensor)
    blur_img = blur_img.transpose([2, 0, 1])
    blur_img = torch.from_numpy(np.expand_dims(blur_img, axis=0)).cuda()
    event_frame = torch.from_numpy(np.expand_dims(event_frame, axis=0)).cuda()
    # inference

    with torch.no_grad():
        factor=64
        h, w = blur_img.shape[2], blur_img.shape[3]
        H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        blur_img = funct.pad(blur_img, (0, padw, 0, padh), 'reflect')
        event_frame = funct.pad(event_frame, (0,padw,0,padh), 'reflect')
        restored = model(blur_img, event_frame)
        restored = restored[:,:,:h,:w]

    # 保存可视化
    for name, tensor in activations.items():
        save_map(tensor, name)

    print("\nSaved activation maps in ./vis/:")
    print(" - x_main.png")
    print(" - low_freq.png")
    print(" - high_freq.png")
    print("\nTarget EDFFN layer:", first_edffn_path)

