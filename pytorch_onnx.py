from onnx2pytorch import ConvertModel
import onnx
import torch
import cv2
from torchvision import transforms
import dlib
from PIL import Image as pil_image


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        transforms.functional.normalize(tensor, self.mean, self.std, inplace=True)
        # for t, m, s in zip(tensor, self.mean, self.std):
        #     t.sub(m).div(s)
        #     # The normalize code -> t.sub_(m).div_(s)
        return tensor


mesonet_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),

    # Added these transforms for attack
    'to_tensor': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'normalize': transforms.Compose([
        Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'unnormalize': transforms.Compose([
        UnNormalize([0.5] * 3, [0.5] * 3)
    ])
}

onnx_model = onnx.load("model.onnx")
pytorch_model = ConvertModel(onnx_model)
video_path = r'F:\original_sequences\youtube\c23\videos\033.mp4'

reader = cv2.VideoCapture(video_path)
classification_list = []
preprocess_func = mesonet_default_data_transforms['test']
real_list = []
fake_list = []
while reader.isOpened():
    _, image = reader.read()
    if image is None:
        break
    height, width = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    if len(faces):
        # For now only take biggest face
        face = faces[0]
        # Face crop with dlib and bounding box scale enlargement
        x, y, size = get_boundingbox(face, width, height)
        cropped_face = image[y:y + size, x:x + size]
        # resize to 256, 256
        img = preprocess_func(pil_image.fromarray(cropped_face))
        img = img.permute((1, 2, 0)).contiguous()
        img = img.unsqueeze(0)
        # img = tf.expand_dims(img, 0)
        classification = pytorch_model.cuda()(img.cuda())
        # classification = round(classification[0, 0])
        real_pred = classification[0][0].item()
        fake_pred = 1 - real_pred
        pred = [real_pred, fake_pred]
        classification_list.append(pred)
        real_list.append(real_pred)
        fake_list.append(fake_pred)
        print('Predicted :', 'real' if pred[0] > pred[1] else 'fake', '\nReal class :', 'real')
print('finished')
pass
