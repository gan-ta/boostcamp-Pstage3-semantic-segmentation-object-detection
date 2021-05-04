import cv2

import torch

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def _PILImage2numpy(im_pil):
    """이미지 numpy formatting 변환 함수
    """

    return np.array(im_pil)

def custom_mixup(img1_path, img2_path, alpha = 0.5):
    """image mix up 
    
    Args:
        img1_path(str) : image1 경로
        img2_path(str): image2 경로
        
    Returns:
        im_pil : numay image array
        beta : image1 influence
        alpha : image2 influence
    """
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    beta = 1.0 - alpha
    
    
    dst = cv2.addWeighted(img1, beta, img2, alpha, 0)
    img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    
    im_pil = Image.fromarray(img)
    return torch.FloatTensor(_PILImage2numpy(im_pil)), beta, alpha


def custom_cutmix(img1, img2, mask1, mask2):
    """ cutmix 이미지, 마스크 생성 함수
    
    Args:
        img1(numpy.ndarray) : 합칠 이미지
        img2(numpy.ndarray) : 합칠 이미지
        mask1(torch.tensor) : 이미지에 대한 마스크
        mask2(torch.tensor) : 이미지에 대한 마스크
    """
    img1_copy = img1.copy()
    img2_copy = img2.copy()
    mask1_copy = torch.tensor(mask1)
    mask2_copy = torch.tensor(mask2)
    
    size = img1.shape
    W = size[1]

    img1_copy[:,:W//2,:] = img2[:,W//2:,:]
    
    img2_copy[:,W//2:,:] = img1[:,:W//2,:]
    mask1_copy[:,:W//2] = mask2[:,W//2:]
    mask2_copy[:,W//2:] = mask1[:,:W//2]
    
    return [torch.FloatTensor(img1_copy), torch.FloatTensor(img2_copy), mask1_copy, mask2_copy]


def get_transforms(aug_type):
    """데이터 Augmentation객체를 불러오는 함수

    Args:
        aug_type(str) : augmentation타입 지정

    Returns :
        list :: train, validation, test데이터 셋에 대한 transform
    """
    
    if aug_type == 'basic':
        train_transform = A.Compose([
            A.Normalize(
                mean=(0.46009655,0.43957878,0.41827092),
                std=(0.2108204,0.20766491,0.21656131),
                max_pixel_value=255.0,
                p=1.0),
            ToTensorV2()
            ])
            
        val_transform = A.Compose([
            A.Normalize(
                mean=(0.46009655,0.43957878,0.41827092),
                std=(0.2108204,0.20766491,0.21656131),
                max_pixel_value=255.0,
                p=1.0),
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

    elif aug_type == "aug2":
        train_transform = A.Compose([
            A.OneOf([
                A.RandomShadow(p=1),
                A.RandomSunFlare(num_flare_circles_lower=1, num_flare_circles_upper=5, src_radius=250,p=1),
                A.RandomRain(p=1),
                A.RandomSnow(brightness_coeff=1.5, p=1),
                A.RandomFog(fog_coef_lower=0.8, fog_coef_upper=1, alpha_coef=0.08,p=1)
            ], p=0.8),
            ToTensorV2()
            ])
            
        val_transform = A.Compose([
            ToTensorV2()
            ])
            
        test_transform = A.Compose([
            ToTensorV2()
            ])


    elif aug_type == "aug3":
        train_transform = A.Compose([
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=1),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=1),
                A.RGBShift(r_shift_limit=0.1, g_shift_limit=0.1, b_shift_limit=0.1,p=1),
                A.ChannelShuffle(0.05)
            ], p=0.8),

            ToTensorV2()
            ])
            
        val_transform = A.Compose([
            ToTensorV2()
            ])
            
        test_transform = A.Compose([
            ToTensorV2()
            ])

    elif aug_type == "aug4":
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
            ])
            
        val_transform = A.Compose([
            ToTensorV2()
            ])
            
        test_transform = A.Compose([
            ToTensorV2()
            ])
    
    elif aug_type == "aug5":
        train_transform = A.Compose([
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            #A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            #A.GridDistortion(p=1.0),
            #A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
            #A.VerticalFlip(p=0.5),
            #A.RandomBrightnessContrast(p=0.8),    
            #A.RandomGamma(p=0.8),
            #A.RandomRotate90(p=0.5),
            #A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.8),
            ToTensorV2()
            ])
            
        val_transform = A.Compose([
            ToTensorV2()
            ])
            
        test_transform = A.Compose([
            ToTensorV2()
            ])

    elif aug_type == "aug6":
        train_transform = A.Compose([
            ToTensorV2()
            ])
            
        val_transform = A.Compose([
            ToTensorV2()
            ])
            
        test_transform = A.Compose([
            ToTensorV2()
            ])

            



    return [train_transform,val_transform,test_transform]