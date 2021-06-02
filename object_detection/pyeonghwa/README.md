# Object Detection

## | Neck
![크기변환 neck](https://user-images.githubusercontent.com/39492330/120482593-3a22f600-c3ec-11eb-80be-09288659a577.png)
> ### PAN
-	neck 부분을 mmdetection의 default인 top-down방식의 FPN구조가 아닌 FPN구조에서 bottom-up형식의 layer에 대한 정보를 추가하는PAN구조로 변경하여 더 많은 정보를 가진 layer를 전송하면 성능의 향상이 있을 것이라 생각하여 PAN구조를 적용하였습니다.
(성능 향상 0.4165 -> 0.4212)

> ### NASFPN
-	layer에 대한 정보를 추가하는 과정에서 신경망을 활용한 NAS-FPN구조를 적용하면 효율적으로 정보가 추가될 것이라 생각되어 NAS-FPN구조를 적용하였습니다.
(성능 하락 - 신경망 구조를 가지다 보니 적합한 하이퍼파라미터를 선정하지 못하였다고 생각합니다.)

> ### BiFPN
-	efficinetDet에 소개되었던 biFPN구조를 적용하였을 때 기존의 neck과는 어떠한 성능차이가 있는지 알아보고자 구현하여 적용해보았습니다.
(성능 하락 0.4165 -> 0.3892 논문에서는 BIFPN구조가 3번의 반복으로 구성되었는데 구현한 코드에서는 1번의 반복만을 거쳐 기대했던 성능향상이 이루어지지 않았던 것으로 추측됩니다.)

![크기변환 carafe](https://user-images.githubusercontent.com/39492330/120483232-dea53800-c3ec-11eb-90ae-54da2b9c9979.png)

> ### CARAFE
-	기존의 FPN에서 UP-Sampling 과정에서 CARAFE기법을 팀원이 발견하여 이를 토대로 NECK에 적용하였습니다
(성능향상 0.4135 -> 0.4301)

> ### PAN CARAFE
-	실험 결과를 토대로 FPN구조보다 PAN구조가 Dataset에 적합하다고 생각하여 FPN에 CARAFE기법을 적용한 것을 바탕으로 PAN구조에서 up-sampling과정에서 CARAFE기법을 적용하였습니다. (성능항샹 0.4301 -> 0.4344)
-	모델의 크기가 작은 regnetx 3.2GF의 경우 성능의 향상을 확인하였지만, backbone의 크기가 큰 resnest 101의 경우 성능이 오히려 하락하는 현상을 확인하였습니다. 모델의 크기가 커짐에 따라 필요한 파라미터의 수가 증가하고 이에 따라 neck을 통과할 때 오히려 bottom-up구조를 추가함으로써 손실된 정보보다 필요하지 않은 정보들이 추가됨으로 성능이 떨어지는 것이 아닐까 추측합니다.
(resnest101기준 성능 하락 0.4699 -> 0.4344)

----

## | DCN
![크기변환 DNC1](https://user-images.githubusercontent.com/39492330/120485137-c1716900-c3ee-11eb-9c3c-b46e1f835fa6.png) ![크기변환 DCN2](https://user-images.githubusercontent.com/39492330/120485141-c33b2c80-c3ee-11eb-80a6-9ff442dfcad6.png)

-	기존Conv layer는해당되는pixel에대해서만(3*3conv layer)로 이루어지지만 중요한 물체에 기울어져서 흩어지는 Conv layer를 적용하는 dcn 기법을 적용하였습니다. (성능 하락 mmdetection에서 dcn을 적용시키는 도중 dcn layer에 pretrained를 불러오지못한 결과라고 추측합니다.)
