from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def get_model(num_classes):
    """
    Creates a Mask R-CNN model with a ResNet50 backbone.

    Args:
        num_classes (int): Number of output classes for the model.

    Returns:
        torch.nn.Module: Mask R-CNN model.
    """
    backbone = resnet_fpn_backbone(backbone_name="resnet50", weights="DEFAULT")
    model = MaskRCNN(backbone, num_classes=num_classes)
    return model
