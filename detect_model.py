import argparse

import cv2
import requests
import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import sys
from utils import torch_utils

#This variable holds the path to object detection model

weights = 'srar_model\\best_mask.pt' if len(sys.argv) == 1 else sys.argv[1]
device_number = '' if len(sys.argv) <=2  else sys.argv[2]

device = torch_utils.select_device(device_number)

model = attempt_load(weights, map_location=device)  # load FP32 model

# function to apply detection model over video or webcam  
def detect():
    save_img=False

    # The source is for the path of video or webcam. It is initially set to 0 which is for webcam.
    # For the source to be a video(.mp4 file) just replace '0' with 'YOUR VIDEO PATH'
    # i.e. source = 'YOUR VIDEO PATH'

    source = '0'
    imgsz = 640
    conf_thres = 0.4
    iou_thres = 0.45
    view_img = True
    save_txt = False
    classes = None
    agnostic_nms = False
    augment = False
    update = False

    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http')

    half = device.type != 'cpu'  # half precision only supported on CUDA

    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    print('this is ', names)
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        NoHelmet_counter = 0
        Helmet_counter = 0
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 6 + '\n') % (cls, *xywh, conf))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        print(names[int(cls)])
                        if int(cls) == 0:
                            Helmet_counter += 1
                        if int(cls) == 1:
                            NoHelmet_counter += 1

                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                cv2.rectangle(im0, (0, 420), (170, 480), (255,255,255), thickness=cv2.FILLED)
                cv2.putText(im0, '{} = {}'.format(names[0], Helmet_counter), (10, 450), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,0), 1 )
                cv2.putText(im0, '{} = {}'.format(names[1],NoHelmet_counter), (10, 470), cv2.FONT_HERSHEY_PLAIN, 1.2, (0,0,0), 1 )
                


            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration


    print('Done. (%.3fs)' % (time.time() - t0))
    return 'DONE'



if __name__ == '__main__':
    detect()
    
