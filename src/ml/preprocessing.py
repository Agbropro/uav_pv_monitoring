from torchvision import transforms

# MobileNetV3 is pretrained on ImageNet, where inputs were normalized
# using these mean and standard deviation values.
# Using the same normalization keeps our input distribution compatible
# with the pretrained model's learned weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def get_train_transform(image_size:tuple[int,int]=(48,80)):
    return transforms.Compose([
        transforms.Resize(
            image_size,
            antialias=True
        ),
        transforms.RandomHorizontalFlip(
            p=0.5
        ),
        transforms.RandomVerticalFlip(
            p=0.5
        ),
        transforms.RandomRotation(
            degrees=5
        ),
        transforms.Grayscale(
            num_output_channels=3
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])
    
def get_eval_transform(image_size: tuple[int,int] = (int,int)):
    return transforms.Compose([
        transforms.Resize(
            image_size,
            antialias=True
        ),
        #The grayscale image is converted into 3 channels because the pretrained MobileNet expects 3-channel input.
        transforms.Grayscale(
            num_output_channels=3
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])