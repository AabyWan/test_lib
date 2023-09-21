import os
import pathlib
from PIL import Image, ImageDraw
from copy import deepcopy
from abc import ABC, abstractmethod

from ..utils import ImageLoader

class Transformer(ABC):
    # Abstract class for transformers. Specifies the interfaces to use when defining a transformer.
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self):
        pass


class Border(Transformer):
    def __init__(self, border_color, border_width, saveToPath=''):
        self.bc = border_color
        self.bw = border_width
        self.aug_name = f"Border_bw{self.bw}_bc{'.'.join([str(n) for n in self.bc])}"
        self.saveToPath = saveToPath
    
    def fit(self, image_obj):
        image = deepcopy(image_obj.image)

        # Draw the rectangle border on the image
        canvas = ImageDraw.Draw(image)
        canvas.rectangle(
            [(0, 0), (image.width - 1, image.height - 1)], #type:ignore
            outline=self.bc, width=self.bw)

        if self.saveToPath: 
            path = os.path.join(self.aug_name,image_obj.filename)
            pathlib.Path(self.aug_name).mkdir(exist_ok=True)
            image.save(path)

        return image

class Flip(Transformer):
    def __init__(self, direction, saveToPath='') -> None:
        self.direction = direction.lower()
        self.aug_name = f"Flip_{direction}"
        self.saveToPath = saveToPath

    def fit(self, image_obj):
        image = deepcopy(image_obj.image)
        
        if self.direction == 'horizontal':
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        if self.direction == 'vertical':
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

        if self.saveToPath : 
            path = os.path.join(self.aug_name,image_obj.filename)
            pathlib.Path(self.aug_name).mkdir(exist_ok=True)
            image.save(path)
        
        return image

class TransformFromDisk(Transformer):
     def __init__(self, aug_name) -> None:
         self.aug_name = aug_name
     
     def fit(self, image_obj):
        filename = image_obj.filename
        path = os.path.join(self.aug_name, filename)
        image_obj = ImageLoader(path=path)
        
        return image_obj.image # image from disk
     
# This func will add a border by expanding the image dimensions.
#from PIL import ImageOps
#def add_boarder(img, border_color = (255, 0, 0), border_width=10):
#    return ImageOps.expand(img, border=border_width, fill=border_color)