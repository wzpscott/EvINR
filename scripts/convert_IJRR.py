import os 
import os.path as path
from argparse import ArgumentParser
from bagpy import bagreader
from PIL import Image
from tqdm import tqdm
import numpy as np


class IJRRconverter():
    def __init__(self, raw_data_dir, output_data_dir):
        self.raw_data_dir = raw_data_dir
        self.output_data_dir = output_data_dir
        os.makedirs(output_data_dir, exist_ok=True)
        self.scenes = [
            'slider_depth',
            'boxes_6dof',
            'calibration',
            'dynamic_6dof',
            'office_zigzag',
            'poster_6dof',
            'shapes_6dof'
        ]

    def convert(self):
        for scene in tqdm(self.scenes):
            tqdm.write(f'start to convert scene: {scene}')
            self.convert_one_scene(scene)
        
    def convert_one_scene(self, scene):
        events_path = path.join(self.output_data_dir, scene, 'events.txt')
        events_np_path = path.join(self.output_data_dir, scene, 'events.npy')
        fp_event = open(events_path, 'w')

        events = []
        for _ in range(10000000):
            line = fp_event.readline()
            if not line:
                break
            e = self.line_to_event(line)
            if e is not None:
                events.append(e)
        fp_event.close()
        np.save(events_np_path, events)


    def line_to_event(self, line):
        e = list(map(float, line.split(' ')))
        if len(e) != 4: # skip lines without 4 elements
            return None
        else:
            t, x, y, p = e
            e = np.array([t, int(x), int(y), -1 if p==0 else 1])
            return e
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--raw_data_dir', type=str, required=True, help='raw data directory')
    parser.add_argument('--output_data_dir', type=str, required=True, help='output data directory')
    args = parser.parse_args()

    raw_data_dir = args.raw_data_dir
    output_data_dir = args.output_data_dir
    converter = IJRRconverter(raw_data_dir, output_data_dir)
    converter.convert()