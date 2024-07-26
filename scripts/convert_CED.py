import os 
import os.path as path
from argparse import ArgumentParser
from bagpy import bagreader
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
from cv_bridge import CvBridge


class CEDconverter():
    def __init__(self, raw_data_dir, output_data_dir):
        self.raw_data_dir = raw_data_dir
        self.output_data_dir = output_data_dir
        os.makedirs(output_data_dir, exist_ok=True)
        self.scenes = [
            'simple_fruit',
            'simple_carpet',
            'simple_objects',
            'simple_jenga_1',
            'simple_jenga_2',
            'simple_rabbits',
            'simple_wires_1'
        ]
    
    def convert(self):
        for scene in tqdm(self.scenes):
            tqdm.write(f'start to convert scene: {scene}')
            self.convert_one_scene(scene)
        
    def convert_one_scene(self, scene):
        bag_path = path.join(self.raw_data_dir, f'{scene}.bag')
        frame_dir = path.join(self.output_data_dir, scene, 'images')
        frame_timestamps_path = path.join(self.output_data_dir, scene, 'images.txt')
        color_frame_dir = path.join(self.output_data_dir, scene, 'color_images')
        color_frame_timestamps_path = path.join(self.output_data_dir, scene, 'color_images.txt')
        events_path = path.join(self.output_data_dir, scene, 'events.txt')
        events_np_path = path.join(self.output_data_dir, scene, 'events.npy')
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(color_frame_dir, exist_ok=True)

        fp_frame = open(frame_timestamps_path, 'w')
        fp_event = open(events_path, 'w')
        fp_color_frame = open(color_frame_timestamps_path, 'w')
        
        b = bagreader(bag_path)
        print(b.topic_table)

        self.bridge = CvBridge()

        for i, (topic, msg, t) in enumerate(tqdm(b.reader.read_messages(topics='/dvs/image_raw'))): 
            frame, frame_timestamp = self.parse_msg_frame(msg, t)
            frame.save(path.join(frame_dir, f'{i:08d}.png'))
            fp_frame.write(f'{frame_timestamp} images/{i:08d}.png\n')
        fp_frame.close()

        for i, (topic, msg, t) in enumerate(tqdm(b.reader.read_messages(topics='/dvs/image_color'))): 
            frame, frame_timestamp = self.parse_msg_color_frame(msg, t)
            frame.save(path.join(color_frame_dir, f'{i:08d}.png'))
            fp_color_frame.write(f'{frame_timestamp} images/{i:08d}.png\n')
        fp_color_frame.close()

        all_events = []
        for i, (topic, msg, t) in enumerate(tqdm(b.reader.read_messages(topics='/dvs/events'))): 
            events = self.parse_msg_event(msg)
            for event in events:
                fp_event.write(f'{event[0]} {event[1]} {event[2]} {event[3]}\n')
                if len(all_events) < 10000000:
                    all_events.append(np.asarray(event))
        fp_event.close()
        np.save(events_np_path, np.stack(all_events))

          
    def parse_msg_event(self, msg):
        events = []
        vals = getattr(msg, 'events')
        for val in vals:
            t = eval(str(getattr(val, 'ts'))) / 1e9
            x = eval(str(getattr(val, 'x')))
            y = eval(str(getattr(val, 'y')))
            p = 1 if (str(getattr(val, 'polarity'))=='True') else -1
            events.append(
                [t, x, y, p]
            )
        return events

    def parse_msg_frame(self, msg, t):
        frame_ts = eval(str(t)) / 1e9
        img = self.bridge.imgmsg_to_cv2(msg)
        frame = Image.fromarray(img)
        return frame, frame_ts
    
    def parse_msg_color_frame(self, msg, t):
        frame_ts = eval(str(t)) / 1e9
        img = self.bridge.imgmsg_to_cv2(msg)
        frame = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return frame, frame_ts
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--raw_data_dir', type=str, required=True, help='raw data directory')
    parser.add_argument('--output_data_dir', type=str, required=True, help='output data directory')
    args = parser.parse_args()

    raw_data_dir = args.raw_data_dir
    output_data_dir = args.output_data_dir
    converter = CEDconverter(raw_data_dir, output_data_dir)
    converter.convert()