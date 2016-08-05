import time
import caffe
import cv2
import numpy as np
import argparse

caffe_root = '/home/alex/caffe-yuchen/'
debug = False


class caffe_feat:
    def __init__(self, proto, model, imgmean,
                 feat_1d, feat_2d,
                 targetsize=224, lmdb=None):
        self.proto = proto
        self.model = model
        self.net = caffe.Net(proto, model, caffe.TEST)
        self.target_size = targetsize
        self.feat_1d = feat_1d
        self.feat_2d = feat_2d

        self.img_mean = imgmean

        if not debug:
            return
        if 'action' not in model:
            imagenet_labels_filename = caffe_root + \
                                       'data/ilsvrc12/synset_words.txt'
            try:
                self.labels = np.loadtxt(imagenet_labels_filename,
                                         str, delimiter='\t')
            except:
                self.labels = np.loadtxt(imagenet_labels_filename,
                                         str, delimiter='\t')
        else:
            self.labels = np.array([
                'unknown', 'Diving', 'GolfSwing', 'Kicking', 'Lifting',
                'Riding', 'Running', 'SkateBoarding', 'SwingBench',
                'SwingSide', 'Walking'
            ])

    def preprocess(self, image):
        # Warning! Only works with OpenCV Image in uint8 and BGR.
        assert image.dtype == np.uint8
        resize_image = cv2.resize(image, (self.target_size, self.target_size))
        # Make sure img_mean can be broadcasted to caffe_in
        caffe_in = resize_image.astype(np.float32) - self.img_mean
        # From (w x h x ch) to (ch x w x h)
        caffe_in = caffe_in.transpose((2, 0, 1))
        return caffe_in.astype(np.float32)

    def extract_feat_file(self, img_file=None):
        if img_file is None:
            image = cv2.imread(caffe_root + 'examples/images/' + 'cat.jpg')
        else:
            image = cv2.imread(caffe_root + str(img_file))
        print("img shape", image.shape)
        f2, f1 = self.extract_feat(image)
        if f2 is not None:
            print("2d feature", np.max(f2), np.min(f2), np.mean(f2))
        if f1 is not None:
            print("1d feature", np.max(f1), np.min(f1), np.mean(f1))

    # raw image, without transfer to gray scale
    def getflow(self, image_prev, image_curr):
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(image_prev, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(image_curr, cv2.COLOR_RGB2GRAY),
            None,
            0.5, 3, 15, 3, 5, 1.2, 0)
        flow_frame = self.getflow_img(flow)
        return flow_frame

    def getflow_img(self, flow):
        max_flow = 8
        scale = 128 / max_flow
        mag_flow = np.sqrt(np.sum(np.square(flow), 2))

        flow = flow * scale
        flow = flow + 128
        flow[flow < 0] = 0
        flow[flow > 255] = 255

        mag_flow = mag_flow * scale
        mag_flow = mag_flow + 128
        mag_flow[mag_flow < 0] = 0
        mag_flow[mag_flow > 255] = 255
        mag_flow = mag_flow.reshape((flow.shape[0], flow.shape[1], 1))

        flow_img = np.concatenate((flow, mag_flow), 2).astype(np.uint8)
        flow_img = flow_img[..., [2, 1, 0]]  # RGB -> BGR
        return flow_img

    def extract_flowfeat(self, image_prev, image_curr):
        return self.extract_feat(self.getflow(image_prev, image_curr))

    def extract_feat(self, image):
        time1 = time.time()
        feat2dlist, feat1dlist = self.extract_feat_batch(1, [image])
        # print("extract feature, cost time = ", time.time() - time1)
        if len(feat2dlist) > 0:
            return feat2dlist[0], feat1dlist[0]
        else:
            return None, feat1dlist[0]

    def extract_feat_batch(self, batch_size, list_of_image):
        """
        Input: list of images.
        Output: list of feat2d, list of feat1d.
        """
        self.net.blobs['data'].reshape(batch_size, 3,
                                       self.target_size,
                                       self.target_size)

        num_batches = int(len(list_of_image) / batch_size)
        if len(list_of_image) % batch_size > 0:
            num_batches += 1

        feat2d_list = []
        feat1d_list = []
        for i_batch in range(num_batches):
            batch_start = i_batch * batch_size
            batch_end = np.min(((i_batch+1) * batch_size, len(list_of_image)))
            for i_ind in range(batch_start, batch_end):
                self.net.blobs['data'].data[i_ind - batch_start, ...] = \
                    self.preprocess(list_of_image[i_ind])

            if self.feat_2d is not None:
                feat2d = self.net.blobs[self.feat_2d].data
                for i_ind in range(batch_start, batch_end):
                    feat2d_list.append(feat2d[i_ind - batch_start])

            if self.feat_1d is not None:
                feat1d = self.net.blobs[self.feat_1d].data
                for i_ind in range(batch_start, batch_end):
                    feat1d_list.append(feat1d[i_ind - batch_start])

        return feat2d_list, feat1d_list


def initialize_model(model='vgg16', mode='cpu'):
    if mode is None:
        mode = 'gpu'
    if 'cpu' in mode.lower():
        caffe.set_mode_cpu()
    if 'gpu' in mode.lower():
        caffe.set_device(0)
        caffe.set_mode_gpu()

    if model is None:
        model = 'vgg16'
    if 'action' in model.lower():
        cnn_proto = (caffe_root +
                     'models/action_cube/deploy_extractpred.prototxt')
        cnn_model = (caffe_root +
                     'models/action_cube/action_cube.caffemodel')
        cnn_imgmean = np.array([128, 128, 128])
        cnn_imgsize = 227
        cnn_feat1d = 'fc6'
        cnn_feat2d = 'conv5'

    print(cnn_proto, cnn_model)
    cf = caffe_feat(cnn_proto, cnn_model, cnn_imgmean,
                    cnn_feat1d, cnn_feat2d, cnn_imgsize)
    return cf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False)
    parser.add_argument('--mode', type=str, required=False)
    args = parser.parse_args()
    cf = initialize_model(args.model, args.mode)
    cf.extract_feat_file()
