from utils.vgg19 import Vgg19
from utils.caffe_feat import initialize_model
import tensorflow as tf
# import cv2
import numpy as np
from utils.s2ao import s2ao


class CookingActionModel(object):
    def __init__(self):
        self.vgg = Vgg19()
        self.images = tf.placeholder(tf.float32, (None, 224, 224, 3))
        self.vgg.build(self.images)
        self.s2ao = s2ao(1000, 1000, 25, 3, 1769)
        self.flow_model = initialize_model('action_cube')

    def get_vgg_feats(self, images):
        return self.vgg.fc6.eval(feed_dict={self.images: images},
                                 session=self.s2ao.sess)

    # def get_optical_flow(self, prev_image, image):
    #     prev_image = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     return cv2.calcOpticalFlowFarneback(
    #         prev_image, image, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    def get_flow_feats(self, prev_image, image):
        return self.flow_model.extract_flowfeat(prev_image, image)[1]

    def get_feats(self, prev_image, image):
        # prev_image_item = np.expand_dims(prev_image, 0)
        # print(self.get_flow_feats(prev_image, image))
        feats = np.concatenate(
            [np.squeeze(self.get_vgg_feats(np.expand_dims(image, 0)), axis=0),
             self.get_flow_feats(prev_image, image)],
            axis=0)
        return feats

    def get_preds_from_feats(self, feats):
        feats = np.expand_dims(feats, axis=0)
        action_preds = np.squeeze(self.s2ao.predict(feats)[0], axis=0)
        return {'cut': action_preds[0],
                'stir': action_preds[2],
                'other': action_preds[1]}
