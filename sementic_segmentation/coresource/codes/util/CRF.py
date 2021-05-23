import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

MAX_ITER = 10

POS_W = 3
POS_XY_STD = 1
Bi_W = 4
Bi_XY_STD = 67
Bi_RGB_STD = 3

def dense_crf(t_img, t_output_probs):
    """dense crf적용
    """
    res = []
    
    for i in range(len(t_img)):
        img = t_img[i].numpy()
        img = np.uint8(255 * img)
        
        output_probs = t_output_probs[i]
        
        c = output_probs.shape[0]
        h = output_probs.shape[1]
        w = output_probs.shape[2]

        U = utils.unary_from_softmax(output_probs)
        U = np.ascontiguousarray(U)

        img = np.ascontiguousarray(img)

        d = dcrf.DenseCRF2D(w, h, c)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
        d.addPairwiseBilateral(sxy=Bi_XY_STD, srgb=Bi_RGB_STD, rgbim=img.reshape(512,512,3), compat=Bi_W)

        Q = d.inference(MAX_ITER)
        Q = np.array(Q).reshape((c, h, w))
        res.append(Q)
    return np.array(res)