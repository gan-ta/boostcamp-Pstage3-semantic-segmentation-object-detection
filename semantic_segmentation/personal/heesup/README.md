## 문제 해결법

- 해당 문제를 풀기 위해 어떻게 접근하고, 구현하고, 최종적으로 사용한 솔루션에 대해서는 [report]()에서 확인할 수 있습니다

- 위 Report는 개인의 회고도 포함하고 있습니다.

## 팀에 대한 역활
- 베이스라인 코드 모듈화 구축

- 다양한 모델에 대한 코드 구현 및 적용

- 결과에 대한 분석 및 해결 방안 제시

## 📜 최종 모델 및 파라미터
   1. model
        1. Encoder : RegNetY
        2. Decoder : DeepLab V3
   2. Learning Rate:
       1. Age Label : 1e-4 
   3. Optimizer : AdamP
   4. Loss : CE + Dice
   5. Epoch: 20
   6. Scheduler : SGDR
   7. batch_size : 4
   
## 🌟 문제 해결 측면
### **[model]**
- 다양한 encoder와 decoder의 조합을 사용하여 기본 베이스 모델 탐색

- encoder, decoder의 pretrain의 유무에 따른 결과 차이 탐색

- DeeplabV3에서 Atrous Convolution 및 Eoncoder 부분의 Feature를 사용한 문제 해결 시도

### **[Augmentation]**
- customize cutmix Augmentation기법 시도 (Object가 항상 잘리도록 customize)

- Alubumentation 라이브러리 사용 
1. ElasticTransform, GridDistortion, OpticalDistortion
2. vertical Flip
3. Random Brightness Contrast
4. RandomGamma
5. RandomRotate 90

### **[loss]**
- 분류 문제를 해결하는 loss와 공간 정보 문제를 해결하는 Dice 계열의 loss를 합친 hybrid loss의 사용

1. classification loss : CE, Focal

2, Dice loss : Dice, Tversky, Lovasz, Jaccard

###  **[결과분석 및 개선방향]**
👉 문제점의 파악

1, 윤곽선의 디테일 문제

2, background를 class로 분류하는 문제

3, 클래스 내부에 여러 클래스로 오판단 하는 현상

💡 해결 방안

1, 별도의 이진 분류 모델을 만들어 필터링 역할을 하는 모델 생성

![1](https://user-images.githubusercontent.com/51118441/120461972-772fbe00-c3d5-11eb-9482-58c900a11a09.png)

2, CRF 후처리를 통하여 윤곽선의 디테일을 살리는 방법 시도

3, 모델의 Pretraine Weight로 인해 오판단을 하는지 혹은 receptive field의 문제로 인하여 object 안의 물체를 잡아 오판단 한다든지 확인 후 그에 대한 수치 조절

###  **[기타 시도 방법]**
1. scheduler 조정 시도

2. pesudo labeling
