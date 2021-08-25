# Heesup Object Detection

## 디렉토리 구성

**`mmdetection_config/myconfig`** : mmdetection 라이브러리를 사용한 model config 파일

**`notebook&metric`** : 노트북 작업 파일 및 평가 지표 함수 구현

**`swin_config/myconfig`** : mmdetection swin 라이브러리를 사용한 model config 파일

**`inference.ipynb`** : 결과 추론을 위한 노트북 파일

**`train.py`** : 모델 훈련을 위한 python파일


## 문제 해결법

- 해당 문제를 풀기위해 어떻게 접근하고, 구현하고, 최종적으로 사용한 솔루션에 대해서는 [report]()에서 확인 할 수 있습니다
- 위 Report는 개인의 회고도 포함하고 있습니다.

## 팀에 대한 역활
- backbone, neck, Detector에 대한 성능 조사

- mosaic Augmentation 데이터 셋을 만들기 위한 코드 구축 및 생성

- deformable convolution, soft nms 등의 기법들을 사용한 실험

- 데이터 Augmentation에 대한 실험, 결과에 대한 원인 분석 및 이를 해결하기 위한 대한 제안

## 📜 최종 모델 및 파라미터
   1. model
        1. Encoder : ResNest
        2. neck : fpn-carafe
        3. Decoder : HTC
   2. Learning Rate:
       1. Age Label : 1e-4 
   3. Optimizer : AdamW
   4. Loss : CE
   5. Epoch: 36
   6. Scheduler : linear
   7. batch_size : 8
   
## 🌟 문제 해결 측면
### **[EDA]**
- bbox의 클래스별 개수 분포도 및 크기 분포도 분석

- bbox와 segmentation의 차지 비율 분석 및 아이디어 도출 과정

### **[model]**
- encoder, neck, decoder 조합의 다양한 model에 대한 실험 및 Validation set과 LB 차이의 차이 추이 확인

- Anchor Boxes, soft nms, deformable convolution 등의 모델 튜닝 시도

### **[Augmentation]**
- mosaic Augmentation 코드를 구현 및 데이터 셋 첨가

-> small size 박스 사이즈 문제 성능 개선

<img src = "https://user-images.githubusercontent.com/51118441/120467440-291db900-c3db-11eb-89e6-7a95b7ae7372.png" width="50%">


- Alubumentation라이브러리 사용(Augmenation 성능 파악)
1. ShiftScalRotate
2. RandomSizedBBox
3. Blur
4. RandomSizedBBoxCrop
5. RandomBrightness
6. CLAHE

### **[loss]**
- 2stage model에서 focal loss를 사용하여 성능 개선 시도(알파 값과 감마 값 조절)

### **[결과분석 및 개선방향]**
#### 👉 문제점의 파악

1. linear 한 박스의 유무(사진의 윤곽 부분에 선의 모양으로 박스가 쳐지는 문제를 시각화를 통해 결과 도출)

2. 클래스에 대한 모델 성능 파악(unknown class에 대한 성능이 좋지 않다는 것을 True Positive, False Positive, False Negative 관점에서 파악)

#### 💡 해결 방안

1. 결과의 후처리를 통한 문제 해결 -> 큰 폭의 점수 상승(Map 0.5150 -> 0.5302)

2. mosaic Augemntaion 데이터 추가 시 unknown class에 대한 oversampling 기법 시도 -> false positive 관점에서의 성능 향상 수치 확인

#### ❗️ 문제 관련 해결법

**`linear한 박스`**  👉  **`oversampling`**

**적용전**<br>
<img src = "https://user-images.githubusercontent.com/51118441/120467595-4d799580-c3db-11eb-9fee-38ef68268055.png" width="50%"><br>
**적용후**<br>
<img src = "https://user-images.githubusercontent.com/51118441/120467607-50748600-c3db-11eb-894e-6e470869eb9d.png" width="50%">

### **[기타 시도 방법]**
1. YOLO, EfficientDet, swin 라이브러리를 활용한 모델 학습 시도

2. WBF 앙상블 기법 시도
