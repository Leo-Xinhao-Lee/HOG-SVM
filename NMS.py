import numpy as np

def non_max_suppression_hand(rect, probs=None, overlapThresh=0.3, method='fast'):
    if rect.size == 0:
        return rect

    x1 = rect[:,0]
    x2 = rect[:,2]
    y1 = rect[:,1]
    y2 = rect[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idx = np.argsort(y2)
    if probs is not None:
        idx = np.argsort(probs)
    pick = []

    while idx.shape[0] > 0:
        last = idx.shape[0] - 1
        pick.append(idx[last])
        if method == 'fast':
            x_st = np.maximum(x1[idx[last]], x1[idx[:last]])
            x_ed = np.minimum(x2[idx[last]], x2[idx[:last]])
            y_st = np.maximum(y1[idx[last]], y1[idx[:last]])
            y_ed = np.minimum(y2[idx[last]], y2[idx[:last]])
            dx = np.maximum(0, x_ed - x_st + 1)
            dy = np.maximum(0, y_ed - y_st + 1)
            intersect = dx * dy
            union = area[idx[last]] + area[idx[:last]]
            #IoU = 1.0 * intersect / union
            IoU = 1.0 * intersect / area[idx[:last]]
            idx = np.delete(idx, np.concatenate(([last], np.where(IoU>overlapThresh)[0])))

        elif method == 'slow':
            suppress = []
            suppress.append(last)
            for i in range(last):
                x_st = np.maximum(x1[idx[last]], x1[idx[i]])
                x_ed = np.minimum(x2[idx[last]], x2[idx[i]])
                y_st = np.maximum(y1[idx[last]], y1[idx[i]])
                y_ed = np.minimum(y2[idx[last]], y2[idx[i]])
                dx = np.maximum(0, x_ed - x_st + 1)
                dy = np.maximum(0, y_ed - y_st + 1)
                intersect = dx * dy
                union = area[idx[last]] + area[idx[i]]
                # IoU = 1.0 * intersect / union
                IoU = 1.0 * intersect / area[idx[i]]
                if IoU > overlapThresh:
                    suppress.append(i)
            idx = np.delete(idx, suppress)

    pick_rect = rect[pick]
    pick_probs = probs[pick]
    return pick_rect, pick_probs
