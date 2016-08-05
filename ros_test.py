import numpy as np
import cv2
from PIL import Image
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from time import sleep

bridge = CvBridge()


def handle_recv(ros_data):
    np_array = np.fromstring(ros_data.data, np.uint8)
    image_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    blue, green, red = image_np.T
    image_np = np.array([red, green, blue]).transpose()
    image = Image.fromarray(image_np, 'RGB')
    image.show()

rospy.init_node('camera_subscriber', anonymous=True)
subscriber = rospy.Subscriber('/cameras/head_camera/image/compressed',
                              CompressedImage,
                              handle_recv, queue_size=1)
print('Subscribed')
sleep(10)
