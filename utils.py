import numpy as np
import cv2

def events_to_event_frame(event_stream, H, W):
    ts = event_stream[:, 0]
    xs = event_stream[:, 2]
    ys = event_stream[:, 1]
    ps = event_stream[:, 3]
    ys, xs, ps = ys.astype(int), xs.astype(int), ps.astype(int)
    
    coords = np.stack((xs, ys))
    try:
        abs_coords = np.ravel_multi_index(coords, (H, W))
    except:
        print(coords[0].min(), coords[1].min(), coords[0].max(), coords[1].max())
        print(H, W)
        raise ValueError()
    event_frame = np.bincount(abs_coords, weights=ps, minlength=H*W).reshape([H, W, 1])
    return event_frame

def quad_bayer_to_rgb_d2(bayer):
    """
    Converts a bayer image to a rgb image.
    rg
    gb
    """
    w, h = bayer.shape

    r = bayer[0:w:2, 0:h:2]
    g_0 = bayer[1:w:2, 0:h:2]
    g_1 = bayer[0:w:2, 1:h:2]
    g = (g_0 + g_1) / 2
    b = bayer[1:w:2, 1:h:2]
    rgb = []
    for c in [r, g, b]:
        rgb.append(cv2.resize(c, (h, w), interpolation=cv2.INTER_NEAREST))
    rgb = np.stack(rgb, axis=-1)
    return rgb

def plot_event_frame(event_frame, threhold=3):
    if event_frame.ndim == 3:
        event_frame_viz = plot_event_frame_multi_channel(event_frame, threhold)
    else:
        event_frame_viz = plot_event_frame_single_channel(event_frame, threhold)
    return event_frame_viz

def plot_event_frame_single_channel(event_frame, threhold=3):
    H, W = event_frame.shape
    background = np.ones([H, W, 3])
    foreground = np.ones([H, W, 3])
    foreground[event_frame>0] = np.asarray([1, 0, 0])
    foreground[event_frame<0] = np.asarray([0, 0, 1])
    alpha = (np.abs(event_frame).clip(0, threhold) / threhold)[..., None]
    event_frame_viz = foreground * alpha + background * (1 - alpha)
    event_frame_viz = (np.clip(event_frame_viz, 0, 1) * 255).astype(np.uint8)
    return event_frame_viz

def plot_event_frame_multi_channel(event_frame, threhold=3):
    H, W, C = event_frame.shape
    assert C == 3
    event_frame_viz_r = plot_event_frame_single_channel(event_frame[..., 0], threhold)
    event_frame_viz_g = plot_event_frame_single_channel(event_frame[..., 1], threhold)
    event_frame_viz_b = plot_event_frame_single_channel(event_frame[..., 2], threhold)
    event_frame_viz = np.concatenate([event_frame_viz_r, event_frame_viz_g, event_frame_viz_b], axis=1)
    return event_frame_viz


