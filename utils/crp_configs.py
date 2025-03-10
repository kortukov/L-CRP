from zennit.torchvision import ResNetCanonizer

from LCRP.utils.crp import CondAttributionLocalization, CondAttributionSegmentation, FeatureVisualizationLocalization, \
    FeatureVisualizationSegmentation
from LCRP.utils.zennit_canonizers import YoloV5V6Canonizer, DeepLabV3PlusCanonizer
from LCRP.utils.zennit_composites import EpsilonPlusFlat, EpsilonGammaFlat
from LCRP.utils.galip_canonizers import YoloV6Canonizer as YoloV6CanonizerGalip

COMPOSITES = {
    # object detectors
    "yolov5": EpsilonPlusFlat,
    "yolov6": EpsilonGammaFlat,
    "yolov6s6": EpsilonGammaFlat,
    "ssd": EpsilonPlusFlat,
    # segmentation models
    "unet": EpsilonPlusFlat,
    "deeplabv3plus": EpsilonPlusFlat,
}

CANONIZERS = {
    # object detectors
    "yolov5": YoloV5V6Canonizer,
    "yolov6": YoloV5V6Canonizer,
    "yolov6s6": YoloV6CanonizerGalip,
    "ssd": ResNetCanonizer,
    # segmentation models
    "unet": ResNetCanonizer,
    "deeplabv3plus": DeepLabV3PlusCanonizer,
}

ATTRIBUTORS = {
    # object detectors
    "yolov5": CondAttributionLocalization,
    "yolov6": CondAttributionLocalization,
    "yolov6s6": CondAttributionLocalization,
    "ssd": CondAttributionLocalization,
    # segmentation models
    "unet": CondAttributionSegmentation,
    "deeplabv3plus": CondAttributionSegmentation,
}

VISUALIZATIONS = {
    # object detectors
    "yolov5": FeatureVisualizationLocalization,
    "yolov6": FeatureVisualizationLocalization,
    "yolov6s6": FeatureVisualizationLocalization,
    "ssd": FeatureVisualizationLocalization,
    # segmentation models
    "unet": FeatureVisualizationSegmentation,
    "deeplabv3plus": FeatureVisualizationSegmentation,
}