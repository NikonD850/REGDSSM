import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset_RGB import *

from matplotlib.colors import Normalize

def auto_pad(voxel, kernel_size):
    """
    自动 padding：保证 H,W 对 kernel_size 整除，不改变图像比例。
    """
    C, H, W = voxel.shape

    pad_h = (kernel_size - (H % kernel_size)) % kernel_size
    pad_w = (kernel_size - (W % kernel_size)) % kernel_size

    if pad_h == 0 and pad_w == 0:
        return voxel

    voxel_padded = np.pad(
        voxel,
        ((0, 0), (0, pad_h), (0, pad_w)),
        mode='constant'
    )
    return voxel_padded

def visualize_event_stream_3d(
    npz_path,
    save_path="events_3d.pdf",
    downsample=None,
    alpha=0.8,
    point_size=2,
    elev=25,
    azim=-60,
    auto_pad=True,
    pad_multiple=32,
    use_timestamp=False,
    num_bins=None
):
    """
    3D event visualization —— event_utils 风格（逐点绘制，无连线）

    Parameters
    ----------
    npz_path : str
        Path to npz with fields t, x, y, p
    save_path : str
        Output PDF
    downsample : int or None
        If provided, uniformly sample N events
    alpha : float
        Transparency
    point_size : float
    elev, azim : view angles
    auto_pad : bool
        If True, pad H,W to nearest divisible by pad_multiple
    use_timestamp : bool
        If True, z-axis=t;
        If False, need num_bins to quantize t->bin
    num_bins : int
        Only used when use_timestamp=False
    """

    data = np.load(npz_path)
    t = data["t"]
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    p = data["p"]

    # -------------------------
    # optional downsample
    # -------------------------
    N = len(t)
    if downsample is not None and N > downsample:
        idx = np.linspace(0, N - 1, downsample, dtype=np.int32)
        t = t[idx]
        x = x[idx]
        y = y[idx]
        p = p[idx]
        N = len(t)

    # -------------------------
    # 自动 padding 到 multiple
    # -------------------------
    H = int(y.max()) + 1
    W = int(x.max()) + 1

    if auto_pad:
        new_H = ((H + pad_multiple - 1) // pad_multiple) * pad_multiple
        new_W = ((W + pad_multiple - 1) // pad_multiple) * pad_multiple
    else:
        new_H, new_W = H, W

    # -------------------------
    # Z 轴：timestamp 或 bin index
    # -------------------------
    if use_timestamp:
        z = t.astype(np.float32)
    else:
        assert num_bins is not None, \
            "num_bins must be set when use_timestamp=False"
        # 量化 t → bin
        t_min, t_max = t.min(), t.max()
        if t_max == t_min:
            z = np.zeros_like(t)
        else:
            z = (t - t_min) / (t_max - t_min + 1e-9) * (num_bins - 1)
        z = z.astype(np.float32)

    # polarity 映射：0 → -1，1 → +1
    # -------------------------
    p_signed = np.where(p > 0, 1, -1)

    # event_utils 配色逻辑：p>0 红, p<0 蓝
    colors = np.zeros((N, 4), dtype=np.float32)
    colors[p_signed > 0] = [1, 0, 0, alpha]
    colors[p_signed < 0] = [0, 0, 1, alpha]

    # -------------------------
    # 3D scatter
    # -------------------------
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        x, y, z,
        c=colors,
        s=point_size,
        depthshade=False
    )

    # -------------------------
    # 让 XY 平面与屏幕平行
    # -------------------------
    ax.set_xlim(0, new_W)
    ax.set_ylim(new_H, 0)     # 上下翻转以符合图像坐标
    ax.set_zlim(0, np.max(z) + 1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Time/Bin")

    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def visualize_voxel_bins_3d_redblue(
    voxel,
    save_path="3D.pdf",
    point_size=1,
    alpha=0.8,
):
    """
    voxel: [num_bins, H, W]
    polarity: >0 = red, <0 = blue
    intensity: 根据 abs(value) 调节亮度
    """
    voxel = voxel[:, ::-1, ::-1]
    num_bins, H, W = voxel.shape
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=65, azim=115,)
    for b in range(num_bins):
        ys, xs = np.nonzero(voxel[b])
        if len(xs) == 0:
            continue

        zs = np.ones_like(xs) * b

        vals = voxel[b, ys, xs]
        abs_vals = np.abs(vals)
        abs_vals = abs_vals / (abs_vals.max() + 1e-6)  # 归一化强度

        # --- 红蓝颜色 ---
        colors = []
        for v, a in zip(vals, abs_vals):
            if v > 0:
                colors.append((1.0, 0.0, 0.0, alpha * (0.3 + 0.7 * a)))  # 红色
            else:
                colors.append((0.0, 0.3, 1.0, alpha * (0.3 + 0.7 * a)))  # 蓝色

        ax.scatter(
            xs, ys, zs,
            s=point_size,
            c=colors,
            marker="o",
            rasterized=True
        )

    # 坐标系设置
    #ax.set_xlabel("W")
    #ax.set_ylabel("H")
    #ax.set_zlabel("Bins")

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_zlim(0, num_bins - 1)

    #ax.set_title(f"3D Event Voxel (red=+, blue=-, bins={num_bins})")
    ax.set_box_aspect((W, H, H))
    plt.tight_layout()
    plt.savefig(save_path,dpi=120, bbox_inches="tight")
    plt.close()

    print(f"Saved to {save_path}")

npz_path = '/mnt/afs/users/fandawei/data/event-deblur/GOPRO_AHDINet/train/event/GOPR0384_11_01/0010.npz'
event = np.load(npz_path)
num_bins=6
W=1280
H=720
# ---- 按你给的逻辑生成 voxel ----
if len(event['t']) == 0:
    voxel = np.zeros((num_bins, H, W), dtype=np.float32)
else:
    event_window = np.stack(
        (event['t'], event['x'], event['y'], event['p']),
        axis=1
    )  # [N, 4]
    voxel = binary_events_to_voxel_grid(
        event_window,
        num_bins=num_bins,
        width=W,
        height=H
    ).astype(np.float32)

#visualize_event_stream_3d(
#    npz_path=npz_path,
#    save_path="events_3d.pdf",
#    use_timestamp=False,   # 或 True
#    num_bins=6,            # 用于 Z -> bin
#    downsample=150000,     # 大 event 文件建议下采样避免太慢
#    point_size=2,
#    alpha=0.9,
#    elev=20,
#    azim=-70,
#    auto_pad=True,
#    pad_multiple=32
#)

visualize_voxel_bins_3d_redblue(
    voxel,
    save_path="voxel_3D_redblue.pdf",
    point_size=0.2,
    alpha=0.65,
)

