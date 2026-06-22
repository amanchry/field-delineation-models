import segmentation_models_pytorch as smp


def build_unet(
    backbone: str = "efficientnet-b5",
    in_channels: int = 8,
    num_classes: int = 3,
    pretrained: bool = True,
) -> smp.Unet:
    """
    UNet with EfficientNet encoder.

    backbone    : "efficientnet-b3" (~12 M params, fast) or
                  "efficientnet-b5" (~30 M params, more accurate)
    in_channels : 8 = 4 bands × 2 seasons (March + August)
    num_classes : 3 = background / field / boundary
    pretrained  : initialise encoder with ImageNet weights
    """
    return smp.Unet(
        encoder_name=backbone,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=num_classes,
    )
