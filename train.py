from argparse import ArgumentParser
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from event_data import EventData
from model import EvINRModel

def config_parser():
    parser = ArgumentParser(description="EvINR")
    parser.add_argument('--exp_name', '-n', type=str, help='Experiment name')
    parser.add_argument('--data_path', '-d', type=str, help='Path of events.npy to train')
    parser.add_argument('--output_dir', '-o', type=str, default='logs', help='Directory to save output')
    parser.add_argument('--t_start', type=float, default=0, help='Start time')
    parser.add_argument('--t_end', type=float, default=2, help='End time')
    parser.add_argument('--H', type=int, default=260, help='Height of frames')
    parser.add_argument('--W', type=int, default=346, help='Width of frames')
    parser.add_argument('--color_event', action='store_true', default=False, help='Whether to use color event')
    parser.add_argument('--event_thresh', type=float, default=1, help='Event activation threshold')
    parser.add_argument('--train_resolution', type=int, default=50, help='Number of training frames')
    parser.add_argument('--val_resolution', type=int, default=50, help='Number of validation frames')
    parser.add_argument('--no_c2f', action='store_true', default=False, help='Whether to use coarse-to-fine training')
    parser.add_argument('--iters', type=int, default=1000, help='Training iterations')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--net_layers', type=int, default=3, help='Number of layers in the network')
    parser.add_argument('--net_width', type=int, default=512, help='Hidden dimension of the network')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')

    return parser

def main(args):
    events = EventData(
        args.data_path, args.t_start, args.t_end, args.H, args.W, args.color_event, args.event_thresh, args.device)
    model = EvINRModel(
        args.net_layers, args.net_width, H=events.H, W=events.W, recon_colors=args.color_event
    ).to(args.device)
    optimizer = torch.optim.AdamW(params=model.net.parameters(), lr=3e-4)

    writer = SummaryWriter(os.path.join(args.output_dir, args.exp_name))
    print(f'Start training ...')
    events.stack_event_frames(args.train_resolution)
    for i_iter in trange(1, args.iters + 1):
        optimizer.zero_grad()

        log_intensity_preds = model(events.timestamps)
        loss = model.get_losses(log_intensity_preds, events.event_frames)

        loss.backward()
        optimizer.step()
        
        if i_iter % args.log_interval == 0:
            tqdm.write(f'iter {i_iter}, loss {loss.item():.4f}')
            writer.add_scalar('loss', loss.item(), i_iter)

        if not args.no_c2f and i_iter == (args.iters // 2):
            events.stack_event_frames(args.train_resolution * 2)


    with torch.no_grad():
        val_timestamps = torch.linspace(-1, 1, args.val_resolution).to(args.device).reshape(-1, 1)
        log_intensity_preds = model(val_timestamps)
        intensity_preds = model.tonemapping(log_intensity_preds)
        for i in range(0, intensity_preds.shape[0]):
            writer.add_image(f'recon/frame_{i}', intensity_preds[i], i_iter, dataformats='HWC')

        # print(intensity_preds.shape)
        # writer.add_images('recon/frames', intensity_preds.repeat_interleave(3, -1), i_iter, dataformats='NHWC')
        vid_tensor = intensity_preds.permute(0, 3, 1, 2).unsqueeze(0)
        if vid_tensor.shape[2] == 1:
            vid_tensor = vid_tensor.repeat_interleave(3, 2)
        writer.add_video('video', vid_tensor, i_iter, fps=16)


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    main(args)


