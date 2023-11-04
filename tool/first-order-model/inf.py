import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import cv2

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull
import face_alignment

#BASE_DIR = os.environ['PYTHONPATH']
BASE_DIR = os.environ['PATH']
WORK_DIR = sys.path[0]

parser = ArgumentParser()
parser.add_argument("--config", default=os.path.join(WORK_DIR, 'config/vox-adv-256.yaml'),help="path to config")
parser.add_argument("--checkpoint", default=os.path.join(WORK_DIR, 'vox-adv-cpk.pth.tar'), help="path to checkpoint to restore")

parser.add_argument("--source_image", required=True, help="path to source image")
parser.add_argument("--driving_video", required=True, help="path to driving video")
parser.add_argument("--result_video", required=True, help="path to output")

parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                    help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,  
                    help="Set frame to start from.")

parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")


parser.set_defaults(relative=False)
parser.set_defaults(adapt_scale=False)

opt = parser.parse_args()

opt.source_image = os.path.join(BASE_DIR, opt.source_image)
opt.driving_video = os.path.join(BASE_DIR, opt.driving_video)
opt.result_video = os.path.join(BASE_DIR, opt.result_video)

class Fomm():
    def __init__(self):
        self.letsee = False
    def generate_inference(self,source_image_path,driving_video_path, result_video_path):

        source_image = imageio.imread(source_image_path)

        """ face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')
        pred = face_detector.get_landmarks(source_image[:, :, :3])[0]
        pts = []
        for i, (x,y) in enumerate(pred):
            if i > 26:
                break
            # print(x, y)
            pts.append([int(x), int(y)])
        pts = np.array(pts)
        pts[17:] = pts[17:][::-1]
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        source_image = source_image[y - 50: y+h + 20, x - 20:x+w + 20, :]

        img_height = source_image.shape[0]
        img_width = source_image.shape[1]
        # print(source_image.shape) """
        print(source_image,"ops")
        # exit()
        reader = imageio.get_reader(driving_video_path)
        print(driving_video_path)
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        source_image = resize(source_image, (256, 256))[..., :3]
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
        generator, kp_detector = self.load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)

        if opt.find_best_frame or opt.best_frame is not None:
            i = opt.best_frame if opt.best_frame is not None else self.find_best_frame(source_image, driving_video, cpu=opt.cpu)
            print ("Best frame: " + str(i))
            driving_forward = driving_video[i:]
            driving_backward = driving_video[:(i+1)][::-1]
            predictions_forward = self.make_animation(source_image, driving_forward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
            predictions_backward = self.make_animation(source_image, driving_backward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
            predictions = predictions_backward[::-1] + predictions_forward[1:]
        else:
            predictions = self.make_animation(source_image, driving_video, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)

        imageio.mimsave(result_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)
        # imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)


# if sys.version_info[0] < 3:
    # raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    def load_checkpoints(self,config_path, checkpoint_path, cpu=False):

        with open(config_path) as f:
            config = yaml.load(f)

        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
        if not cpu:
            generator.cuda()

        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                 **config['model_params']['common_params'])
        if not cpu:
            kp_detector.cuda()
        
        if cpu:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_path)
     
        generator.load_state_dict(checkpoint['generator'])
        kp_detector.load_state_dict(checkpoint['kp_detector'])
        
        if not cpu:
            generator = DataParallelWithCallback(generator)
            kp_detector = DataParallelWithCallback(kp_detector)

        generator.eval()
        kp_detector.eval()
        
        return generator, kp_detector

    def find_best_frame(self, source, driving, cpu=False):
        import face_alignment

        def normalize_kp(kp):
            kp = kp - kp.mean(axis=0, keepdims=True)
            area = ConvexHull(kp[:, :2]).volume
            area = np.sqrt(area)
            kp[:, :2] = kp[:, :2] / area
            return kp

        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                          device='cpu' if cpu else 'cuda')
        kp_source = fa.get_landmarks(255 * source)[0]
        kp_source = normalize_kp(kp_source)
        norm  = float('inf')
        frame_num = 0
        for i, image in tqdm(enumerate(driving)):
            kp_driving = fa.get_landmarks(255 * image)[0]
            kp_driving = normalize_kp(kp_driving)
            new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
            if new_norm < norm:
                norm = new_norm
                frame_num = i
        return frame_num

    def make_animation(self,source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
        with torch.no_grad():
            predictions = []
            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            if not cpu:
                source = source.cuda()
            driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
            kp_source = kp_detector(source)
            kp_driving_initial = kp_detector(driving[:, :, 0])

            for frame_idx in tqdm(range(driving.shape[2])):
                driving_frame = driving[:, :, frame_idx]
                if not cpu:
                    driving_frame = driving_frame.cuda()
                kp_driving = kp_detector(driving_frame)
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                       kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                       use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

                frame_re = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
                # frame_res = cv2.resize(frame_re,dsize=(img_width,img_height))
                predictions.append(frame_re)
        return predictions


### TEST
f = Fomm()
# f.generate_inference('../MakeItTalk/examples/cesi.jpg','./out.mp4')
f.generate_inference(opt.source_image,opt.driving_video, opt.result_video)
