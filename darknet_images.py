import argparse
import os
import glob
import random
import darknet
import time
import cv2
import numpy as np
import darknet
import math
import py360convert2 as py360convert
from PIL import Image



def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="j__0560.jpg",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="../../../yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default="cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.50,
                        help="remove detections with lower confidence")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))


def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)


def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    #print(width,height)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    w= image.shape[1]
    h= image.shape[0]
    # img[top : bottom, left : right]
    # サンプル1の切り出し、保存
    img_center = image[int(h/6) : int(5*h/6) , 0 : w]
    img_center = cv2pil(img_center)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    start_time = time.perf_counter()
    #execution_time = time.perf_counter() - start_time
    img_bottom = py360convert.e2p(image,fov_deg=(60,60),u_deg=0,v_deg=-90,out_hw=(300,300),in_rot_deg=0,mode='bilinear')
    img_top = py360convert.e2p(image,fov_deg=(60,60),u_deg=0,v_deg=90,out_hw=(300,300),in_rot_deg=0,mode='bilinear')
    execution_time = time.perf_counter() - start_time
    img_bottom = Image.fromarray(img_bottom)
    img_top = Image.fromarray(img_top)
    get_concat_v_blank(img_bottom, img_center,img_top).save('/root/1202/image_result2.jpg')
    print(execution_time)
    image_rgb = get_concat_v_blank(img_bottom, img_center,img_top)
    image_rgb  = np.array(image_rgb)
    #print(type(image_rgb))
    #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def image_classification(image, network, class_names):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    darknet_image = darknet.make_image(width, height, 3)
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.predict_image(network, darknet_image)
    predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
    darknet.free_image(darknet_image)
    return sorted(predictions, key=lambda x: -x[1])


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = os.path.splitext(name)[0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def batch_detection_example():
    args = parser()
    check_arguments_errors(args)
    batch_size = 3
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )
    image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    images = [cv2.imread(image) for image in image_names]
    images, detections,  = batch_detection(network, images, class_names,
                                           class_colors, batch_size=batch_size)
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)
    print(detections)

def phidis(cx,cy,x1,y1,x2,y2,x3,y3,x4,y4):
    a0 = np.array([cx, cy])
    b0 = np.array([x1, y1])
    phi0= -(90 - (np.linalg.norm(a0-b0) / 5))
    #phi0= np.linalg.norm(a0-b0) / 5

    a1 = np.array([cx, cy])
    b1 = np.array([x2, y2])
    phi1= -(90 - (np.linalg.norm(a1-b1) / 5))
    #phi1= np.linalg.norm(a1-b1) / 5


    a2 = np.array([cx, cy])
    b2 = np.array([x3, y3])
    phi2= -(90 - (np.linalg.norm(a2-b2) / 5))
    #phi2= np.linalg.norm(a2-b2) / 5

    a3 = np.array([cx, cy])
    b3 = np.array([x4, y4])
    phi3= -(90 - (np.linalg.norm(a3-b3) / 5))
    #phi3= np.linalg.norm(a3-b3) / 5

    return phi1,phi0,phi3,phi2

def thetadis(cx,cy,x,y):
    a = np.array([cx, cy])
    b = np.array([x, y])
    vec = b - a
    theta = np.arctan2(vec[0], vec[1])
    if x < cx:
      theta = -180 - math.degrees(theta)
    else:
      theta = 180 - math.degrees(theta)
    return theta

def convertpano(w,h,phi,theta):
  new_x = w* ( (theta/360) + (1/2) )
  new_y = h* ( (1/2) - (phi/180) )
  return new_x,new_y

def get_concat_v_blank(im1, im2, im3,color=(0, 0, 0)):
    dst = Image.new('RGB', (max(im1.width, im2.width,im3.width), im1.height + im2.height + im3.height), color)
    dst.paste(im1, (int(im2.width/2 - im1.width/2), 0))
    dst.paste(im2, (0, im1.height))
    dst.paste(im3, (int(im2.width/2 - im3.width/2), im1.height + im2.height))
    return dst

def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image



def main():
    args = parser()
    check_arguments_errors(args)

    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )
    images = load_images(args.input)
    index = 0
    while True:
        # loop asking for new image paths if no list is given
        if args.input:
            if index >= len(images):
                break
            image_name = images[index]
        else:
            image_name = input("Enter Image Path: ")
        prev_time = time.time()
        image, detections = image_detection(
            image_name, network, class_names, class_colors, args.thresh
            )
        #print(detections)

        # img = cv2.rectangle(image,(340,0),(340,608),(0,0,255),3)
        # img = cv2.rectangle(image,(0,326),(608,326),(0,0,255),3)
        # #img = cv2.rectangle(image,(135,51),(545,601),(255,0,0),3)
        if args.save_labels:
            save_annotations(image_name, image, detections, class_names)
        #x,y,w,h= darknet.print_detections(detections, args.ext_output)
        darknet.print_detections(detections, args.ext_output)
        # x1= (((x-w/2)*300)/608) - (x*300/608)
        # y1 = -( (((y- h/2)*300)/608) - (y*300/608) )
        # x2= ((x - w/2)*300)/608 - (x*300/608)
        # y2 = -( ((y + h/2)*300)/608 - (y*300/608) )
        # x3= ((x + w/2)*300)/608 - (x*300/608)
        # y3 = -( ((y + h/2)*300)/608 - (y*300/608) )
        # x4= ((x + w/2)*300)/608 - (x*300/608)
        # y4 = -( ((y - h/2)*300)/608 - (y*300/608) )
        # #print(x1,y1)
        # #print('({:.0f}'.format(x1), '{:.0f}'.format(y1),')')
        # x1=int('{:.0f}'.format(x1))
        # y1=int('{:.0f}'.format(y1))
        # x2=int('{:.0f}'.format(x2))
        # y2=int('{:.0f}'.format(y2))
        # #print(x2,y2)
        # #print(x3,y3)
        # #print(x4,y4)
        # x3=int('{:.0f}'.format(x3))
        # y3=int('{:.0f}'.format(y3))
        # x4=int('{:.0f}'.format(x4))
        # y4=int('{:.0f}'.format(y4))

        # cx= 150 - int('{:.0f}'.format(x*300/608))
        # cy= 150 - int('{:.0f}'.format(y*300/608))
        # phi1,phi2,phi3,phi4 = phidis(cx,cy,x1,y1,x2,y2,x3,y3,x4,y4)
        # theta2 = thetadis(cx,cy,x1,y1)
        # theta1 = thetadis(cx,cy,x2,y2)
        # theta4 = thetadis(cx,cy,x3,y3)
        # theta3 = thetadis(cx,cy,x4,y4)
        # # print(phi1)
        # # print(phi2)
        # # print(phi3)
        # # print(phi4)
        # print(theta1)
        # print(theta2)
        # print(theta3)
        # print(theta4)


        # print('{:.0f}'.format((((x-w/2)*300)/608)),'{:.0f}'.format((((y-h/2)*300)/608)))
        # #print(x2,y2)
        # print('{:.0f}'.format((((x+w/2)*300)/608)),'{:.0f}'.format((((y+h/2)*300)/608)))

        # #img = cv2.rectangle(image,(int(x1),int(y1)),(int(x3),int(y3)),(255,255,0),3)
        # image2 = cv2.imread('/content/drive/MyDrive/j__0560.jpg')
        # new_w,new_h = image2.shape[1],image2.shape[0]
        # #print(new_w,new_h)

        # new_x1,new_y1 = convertpano(new_w,new_h,phi1,theta1)
        # new_x2,new_y2 = convertpano(new_w,new_h,phi2,theta2)
        # new_x3,new_y3 = convertpano(new_w,new_h,phi3,theta3)
        # new_x4,new_y4 = convertpano(new_w,new_h,phi4,theta4)

        # add_new_x1,add_new_y1 = convertpano(new_w,new_h,-60,0)
        # add_new_x2,add_new_y2 = convertpano(new_w,new_h,-60,-90)
        # add_new_x3,add_new_y3 = convertpano(new_w,new_h,-60,180)
        # add_new_x3_2,add_new_y3_2 = convertpano(new_w,new_h,-60,-180)
        # add_new_x4,add_new_y4 = convertpano(new_w,new_h,-60,90)

        # cv2.circle(image2, (int(new_x1),int(new_y1)), 1, (255, 0, 255), thickness=10)
        # cv2.circle(image2, (int(new_x2),int(new_y2)), 1, (255, 0, 255), thickness=10)
        # cv2.circle(image2, (int(new_x3),int(new_y3)), 1, (255, 0, 255), thickness=10)
        # cv2.circle(image2, (int(new_x4),int(new_y4)), 1, (255, 0, 255), thickness=10)

        # cv2.circle(image2, (int(add_new_x1),int(add_new_y1)), 1, (255, 0, 0), thickness=10)
        # cv2.circle(image2, (int(add_new_x2),int(add_new_y2)), 1, (255, 0, 0), thickness=10)
        # cv2.circle(image2, (int(add_new_x3),int(add_new_y3)), 1, (255, 0, 0), thickness=10)
        # cv2.circle(image2, (int(add_new_x4),int(add_new_y4)), 1, (255, 0, 0), thickness=10)
        # cv2.circle(image2, (int(add_new_x3_2),int(add_new_y3_2)), 1, (255, 0, 0), thickness=10)

        fps = int(1/(time.time() - prev_time))
        print("FPS: {}".format(fps))
        cv2.imwrite('/root/1202/detect_res.jpg',image)
        #cv2.imwrite('res_2.jpg',image2)
        if not args.dont_show:
            cv2.imshow('Inference', image)
            if cv2.waitKey() & 0xFF == ord('q'):
                break
        index += 1


if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    main()
