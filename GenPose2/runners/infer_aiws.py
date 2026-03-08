import cutoop
import numpy as np
from cutoop.data_loader import Dataset
from cutoop.eval_utils import GroundTruth, DetectMatch, DetectOutput
from dataclasses import asdict
import cv2

# take this prefix for example
path = "data/AIWS/"
prefix = "001_"
# load RGB color image (in RGB order)
image = Dataset.load_color(path+"color/"+prefix + "color.png")
# load object metadata
meta = Dataset.load_meta(path+"meta/"+ prefix + "meta.json")
# load object mask (0 and 255 are both backgrounds)
mask = Dataset.load_mask(path+"mask/"+ prefix + "mask.exr")
# load depth image. 0 and very large values are considered invisible
depth = Dataset.load_depth(path+"depth/"+ prefix + "depth.exr")
depth[depth > 1e5] = 0

cv2.imshow("image",image)
cv2.imshow("mask",mask)
cv2.imshow("depth",depth)
cv2.waitKey(0)
cv2.destroyAllWindows()
