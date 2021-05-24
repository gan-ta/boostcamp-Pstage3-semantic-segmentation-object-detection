import torch
from collections import Counter
import numpy as np

def get_iou(pred_box, gt_box):
    """iou 계산 함수
    
    Args :
        pred_box(list) : 모델이 예측한 bbox
        gt_box(list) : ground truth
        
    Returns :
        float : iou값
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni
    
    return iou

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=11):
    """Map계산 로직
    
    Args :
        pred_boxes (list) : [[train_idx, class_pred, prob_score, x1, y1, x2, y2], ...]  train_idx: bounding box가 어디에서 왔는지
        rue_boxes(list) : pred_boxes와 동일(prob_score는 사용하지 않으니 1로 설정하여 사용해도 됨)
        iou_threshold(float) : map계산시 bbox의 threshold의 값
        num_class(int) : 클래스 분류 값
        
    Returns:
        tensor : Map계산 값
    """
    average_precisions = []

    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []
        
        # 해당 클래스인 것만 모아주기(pred)
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        
        # 해당 클래스인 것만 모아주기(true)
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
        
        # img0 has 3 bboxes
        # img1 has 5 bboxes
        # amount_bboxes = {0 : 3,  1: 5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths]) # ground truth에서 현재 해당하는 클래스에서 이미지 마다 몇개의 object가 있는지 카운드
        
        # amount_bboxes = {0 : torch.tensor([0,0,0]), 1 : torch.tensor([0,0,0,0,0])}
        
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # dection정보 하나 가지고 와서 이미지 인덱스 같은거에 대하여만 ground truth가져오기
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0
            
            # gt에서 현재 조사하고자 하는 bbox중 가장 높은 iou을 가지고 있는 bbox를 산출
            for idx, gt in enumerate(ground_truth_img):
                iou = get_iou(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
             # 0.5가 넘을때만 고려
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0: # 이 위치 bbox가 업데이트가 되어지지 않음
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1 # bounding box를 커버
                else:
                    FP[detection_idx] = 1

            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)
