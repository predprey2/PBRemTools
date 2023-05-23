import os
import cv2

from scripts.td_abg import get_foreground
from scripts.convertor import pil2cv

try:
    from modules.paths_internal import extensions_dir
except Exception:
    from modules.extensions import extensions_dir

from collections import OrderedDict

model_cache = OrderedDict()
sam_model_dir = os.path.join(
    extensions_dir, "PBRemTools/models/")
model_list = [f for f in os.listdir(sam_model_dir) if os.path.isfile(
    os.path.join(sam_model_dir, f)) and f.split('.')[-1] != 'txt']

def process_image(target_image, *rem_args):
    image = pil2cv(target_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask, image = get_foreground(image, *rem_args)
    return image, mask