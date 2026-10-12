import segmentation_models_pytorch as smp


# Model definition
def create_pretrained_unetplusplus():
    model_pretrained = smp.UnetPlusPlus(
        encoder_name="resnet34",        # pretrained encoder (knows edges, textures, shapes)
        encoder_weights="imagenet",     # weights trained on 1.2M images
        in_channels=3,                  # RGB input
        classes=6,                      # our 6 classes
    )
    return model_pretrained
