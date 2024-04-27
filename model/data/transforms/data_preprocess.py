from .transforms import *

class TrainTransforms:
    def __init__(self, cfg):
        comp = []
        # print(cfg.DATASET.DATA_AUGMENTATION)
        for trans in cfg.DATASET.DATA_AUGMENTATION:
            # print(trans)
            func, args = trans
            print(func, args, type(args))
            if func == "RandomResizedCrop":
                comp.append(eval(func)(cfg.INPUT.IMAGE_SIZE, **args[0]))
            elif func == "RandomCrop":
                comp.append(eval(func)(cfg.INPUT.IMAGE_SIZE))
            elif args == None or args == "None":
                comp.append(eval(func)())
            else:
                eval(func)(args)
        self.augment = Compose(comp)
        # self.augment = Compose([
        #     ConvertFromInts(),
        #     RandomMirror(),
        #     # PhotometricDistort(),
        #     # Normalize(cfg),
        #     ToTensor(),
        #     # RandomGrayscale(p=0.25),
        #     RandomVerticalFlip(p=0.3),
        #     RandomResizedCrop(cfg=cfg, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
        # ])

    def __call__(self, image, mask):
        image, mask = self.augment(image, mask)
        if mask is not None:
            return image/255, mask/255
        
        return image/255, None

class TestTransforms:
    def __init__(self, cfg):
        self.augment = Compose([
            ConvertFromInts(),
            ToTensor(),
        ])

    def __call__(self, image, mask):
        image, mask = self.augment(image, mask)
        if mask is not None:
            return image/255, mask/255
        
        return image/255, None
