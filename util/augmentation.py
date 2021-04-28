import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(aug_type):
    """데이터 Augmentation객체를 불러오는 함수

    Args:
        aug_type(str) : augmentation타입 지정

    Returns :
        list :: train, validation, test데이터 셋에 대한 transform
    """
    
    if aug_type == 'basic':
        train_transform = A.Compose([
            ToTensorV2()
            ])
            
        val_transform = A.Compose([
            ToTensorV2()
            ])
            
        test_transform = A.Compose([
            ToTensorV2()
            ])
    elif aug_type == "aug1":
        train_transform = A.Compose([
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
            ], p=0.8),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.8),    
            A.RandomGamma(p=0.8),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
            ])
            
        val_transform = A.Compose([
            ToTensorV2()
            ])
            
        test_transform = A.Compose([
            ToTensorV2()
            ])

    return [train_transform,val_transform,test_transform]