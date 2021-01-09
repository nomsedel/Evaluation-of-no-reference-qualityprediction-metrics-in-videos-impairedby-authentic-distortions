"""Test Demo for Quality Assessment of In-the-Wild Videos, ACM MM 2019"""
#
# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2018/3/27
#

import skvideo
import skvideo.io
import torch
from torchvision import transforms
from PIL import Image
import numpy
import numpy as np
import mat4py as m4p
from pathlib import Path
from VSFA import VSFA
from CNNfeatures import get_features
from argparse import ArgumentParser
import time
import os




# ingresa la direccion donde esta la base de datos (esta linea debe terminar con un /)
path='/home/stidl/Downloads/videofalra-20201211T212209Z-001/kondi/'
#os.path.join('/home/stidl/Desktop/THESIS/Databases/KoNViD_1k_videos/')
#os.chdir('/home/stidl/Desktop/THESIS/Databases/KoNViD_1k_videos')
# ingresa la direccion donde se van a guardar los archivos .mat de las caracteristicas (esta linea debe terminar con un /)
path_features = '/home/stidl/Desktop/THESIS/VSFA-master/feature/'
# ingresa la direccion donde se van a guardar los archivos .mat de los scores (esta linea debe terminar con un /)
path_scores = '/home/stidl/Desktop/THESIS/VSFA-master/score/'

mylist = os.listdir(path)
# en range pon el numero total de videos que tiene esa categoria
#for i in range(len(mylist)):
for i in range(len(mylist)):
    start = time.time()
    n_video=mylist[(i)]	
    video= path+mylist[(i)]
    name_video_features=path_features+n_video+'_featuresVSFA.mat'
    name_video_score=path_scores+n_video+'_scoreVSFA.mat'
    print (i)
    print (n_video)

    if __name__ == "__main__":
        parser = ArgumentParser(description='"Test Demo of VSFA')
        parser.add_argument('--model_path', default='models/VSFA.pt', type=str,
                            help='model path (default: models/VSFA.pt)')
        parser.add_argument('--video_path', default=video, type=str,
                            help='video path (default: video que cargo)')
        parser.add_argument('--video_format', default='RGB', type=str,
                            help='video format: RGB or YUV420 (default: RGB)')
        parser.add_argument('--video_width', type=int, default=None,
                            help='video width')
        parser.add_argument('--video_height', type=int, default=None,
                            help='video height')
    
        parser.add_argument('--frame_batch_size', type=int, default=1,
                            help='frame batch size for feature extraction (default: 32)')
        args = parser.parse_args()
        #f = open("/home/vigilcali/roger_nieto/VSFA-master/results/")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        start = time.time()
    
        # data preparation
        assert args.video_format == 'YUV420' or args.video_format == 'RGB'
        if args.video_format == 'YUV420':
            video_data = skvideo.io.vread(args.video_path, args.video_height, args.video_width, inputdict={'-pix_fmt': 'yuvj420p'})
        else:
            video_data = skvideo.io.vread(args.video_path)
    
        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[frame_idx] = frame
    
    
        # feature extraction
        features = get_features(transformed_video, frame_batch_size=args.frame_batch_size, device=device)
        features = torch.unsqueeze(features, 0)  # batch size 1
        c=features.cpu()
        a=np.array(c)
        data1 = {'a' : a.tolist()}

        m4p.savemat(name_video_features, data1)
    
        # quality prediction using VSFA
        model = VSFA()
        model.load_state_dict(torch.load(args.model_path))  #
        model.to(device)
        model.eval()
        with torch.no_grad():
            input_length = features.shape[1] * torch.ones(1, 1)
            outputs = model(features, input_length)
            y_pred = outputs[0][0].to('cpu').numpy()
            qv=video+'Quality'
            b=np.array(y_pred)
            data = {'b' : b.tolist()}
            m4p.savemat(name_video_score, data)

            
    

    end = time.time()
    print('Time: {} s'.format(end-start))
    
