from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, precision_recall_curve, average_precision_score
from scipy.interpolate import interp1d
from inspect import signature
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import numpy as np

def roc_auc(labels, scores, show=False):
    """Compute ROC curve and ROC area for each class"""
    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    if show:
        # Equal Error Rate
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        plt.figure()
        lw = 2
        plt.plot(fpr,
                 tpr,
                 color='darkorange',
                 lw=lw,
                 label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1 - eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig("../img/roc_auc_onecls.png")
        plt.close()
    return {'roc_auc': roc_auc}


def pre_rec_curve(labels, scores, show=False):
    average_precision = average_precision_score(labels, scores)
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    if show:
        precision, recall, _ = precision_recall_curve(labels, scores)
        step_kwargs = ({
            'step': 'post'
        } if 'step' in signature(plt.fill_between).parameters else {})
        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall,
                         precision,
                         alpha=0.2,
                         color='b',
                         **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        plt.savefig("../img/pre_rec_curve_onecls.png")
        plt.close()
    return {'average_precision': average_precision}

def accuracy():
    
    return 0

def convertToAbsoluteValues(size, box):
    
    xIn = round(((2 * float(box[0]) - float(box[2])) * size[0] / 2))
    yIn = round(((2 * float(box[1]) - float(box[3])) * size[1] / 2))
    xEnd = xIn + round(float(box[2]) * size[0])
    yEnd = yIn + round(float(box[3]) * size[1])
    
    if xIn < 0:
        xIn = 0
    if yIn < 0:
        yIn = 0
    if xEnd >= size[0]:
        xEnd = size[0] - 1
    if yEnd >= size[1]:
        yEnd = size[1] - 1
    return (xIn, yIn, xEnd, yEnd)

def getArea(box):
    return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

def calculateIoU(GroundTruthBox,BoundingBox):

    InterSection_min_x = max(BoundingBox[0], GroundTruthBox[0])
    InterSection_min_y = max(BoundingBox[1], GroundTruthBox[1])

    InterSection_max_x = min(BoundingBox[2], GroundTruthBox[2])
    InterSection_max_y = min(BoundingBox[3], GroundTruthBox[3])

    InterSection_Area = 0

    InterSection_Area = max(0, InterSection_max_x - InterSection_min_x) * max(0, InterSection_max_y - InterSection_min_y)

    box1_area = abs(BoundingBox[0]-BoundingBox[2])*abs(BoundingBox[1]-BoundingBox[3])
    box2_area = abs(GroundTruthBox[0]-GroundTruthBox[2])*abs(GroundTruthBox[1]-GroundTruthBox[3])
    Union_Area = box1_area + box2_area - InterSection_Area

    floatIou=InterSection_Area/(Union_Area+1e-7)
    return floatIou

def calculateAveragePrecision(rec, prec):
    
    mrec = [0] + [e for e in rec] + [1]
    mpre = [0] + [e for e in prec] + [0]

    for i in range(len(mpre)-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])

    ii = []

    for i in range(len(mrec)-1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i+1)

    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i-1]) * mpre[i])
    
    return [ap, mpre[0:len(mpre)-1], mrec[0:len(mpre)-1], ii]

def ElevenPointInterpolatedAP(rec, prec):

    mrec = [e for e in rec]
    mpre = [e for e in prec]

    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp, recallValid = [], []

    for r in recallValues:
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0

        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])

        recallValid.append(r)
        rhoInterp.append(pmax)

    ap = sum(rhoInterp) / 11

    return [ap, rhoInterp, recallValues, None]

def AP(detectionbox, detectionlabel, ObjectScore, groundtruthbox ,groundtruthlabel, nclasses, IOUThreshold = 0.001, method = 'AP'):
    while True :
        try :
            result = [] 
            detections = []
            groundtruths = []
            for i in range(len(detectionbox)):
                if len(detectionlabel[i])!=0:
                    for j in range(len(detectionbox[i])):
                        imagedetections=[i, np.argmax(tf.squeeze(detectionlabel[i][j]).numpy()), ObjectScore[i][j].numpy()[0][0], detectionbox[i][j][0].numpy(), detectionbox[i][j][1].numpy(), detectionbox[i][j][2].numpy(), detectionbox[i][j][3].numpy()]
                        detections.append(imagedetections)            
                    for j in range(len(groundtruthlabel[i])):
                        imagegroundtruths=[i, np.argmax(groundtruthlabel[i][j]), 1, groundtruthbox[i][j][0], groundtruthbox[i][j][1], groundtruthbox[i][j][2], groundtruthbox[i][j][3]]
                        groundtruths.append(imagegroundtruths)
                
              
            classes=list(range(nclasses)) 
            for c in classes:

                dects = [d for d in detections if d[1] == c]
                gts = [g for g in groundtruths if g[1] == c]

                npos = len(gts)

                dects = sorted(dects, key = lambda dects : dects[2], reverse=True)

                TP = np.zeros(len(dects))
                FP = np.zeros(len(dects))

                det = Counter(cc[0] for cc in gts)

                # 각 이미지별 ground truth box의 수
                # {99 : 2, 380 : 4, ....}
                # {99 : [0, 0], 380 : [0, 0, 0, 0], ...}
                for key, val in det.items():
                    det[key] = np.zeros(val)


                for d in range(len(dects)):

                    gt = [gt for gt in gts if gt[0] == dects[d][0]]

                    iouMax = 0

                    for j in range(len(gt)):
                        iou1 = calculateIoU(dects[d][3:], gt[j][3:])
                        if iou1 > iouMax:
                            iouMax = iou1
                            jmax = j

                    if iouMax >= IOUThreshold:
                        if det[dects[d][0]][jmax] == 0:
                            TP[d] = 1
                            det[dects[d][0]][jmax] = 1
                        else:
                            FP[d] = 1
                    else:
                        FP[d] = 1

                acc_FP = np.cumsum(FP)
                acc_TP = np.cumsum(TP)
                if npos==0:
                    rec = np.zeros(len(acc_TP))
                else:
                    rec = acc_TP / npos
                prec = np.divide(acc_TP, (acc_FP + acc_TP))

                if method == "AP":
                    [ap, mpre, mrec, ii] = calculateAveragePrecision(rec, prec)
                else:
                    [ap, mpre, mrec, _] = ElevenPointInterpolatedAP(rec, prec)
                if len(acc_TP)==0:
                    r = {
                    'class' : c,
                    'precision' : 0,
                    'recall' : 0,
                    'AP' : 0,
                    'interpolated precision' : 0,
                    'interpolated recall' : 0,
                    'total positives' : 0,
                    'total TP' : 0,
                    'total FP' : 0
                    }
                    result.append(r)
    
                else:
                    r = {
                    'class' : c,
                    'precision' : prec[-1],
                    'recall' : rec[-1],
                    'AP' : ap,
                    'interpolated precision' : mpre,
                    'interpolated recall' : mrec,
                    'total positives' : npos,
                    'total TP' : np.sum(TP),
                    'total FP' : np.sum(FP)
                    }
                    result.append(r)
        except tf.errors.ResourceExhaustedError as e:
            strFuncName = sys._getframe().f_code.co_name
            strReturn  = "[%s] : " % (strFuncName) + str(e)
            break       
        except RuntimeError as e :              
            strFuncName = sys._getframe().f_code.co_name
            strReturn  = "[%s] : " % (strFuncName) + str(e)
            break
        except Exception as e :
            strFuncName = sys._getframe().f_code.co_name
            strReturn  = "[%s] : " % (strFuncName) + str(e)
            break 
        
        bReturn = True
        break

    return result

def calculatemAP(result):
    ap = 0
    for r in result:
        ap += r['AP']
    mAP = ap / len(result)
    
    return mAP

def calculatePrecisionRecall(result):
    pr=0
    re=0
    for r in result:
        pr += r['precision']
        re += r['recall']
    pr /= len(result)
    re /= len(result)
    return pr,re