def countdice(pred, target):
    smooth = 1e-6
    pred = pred.view(-1)
    target = target.view(-1)
    TP = (pred * target).sum()
    FP = ((1 - target) * pred).sum()
    FN = ((1 - pred) * target).sum()
    return (2. * TP) / ((2. * TP) + FP + FN + smooth)

def countiou(pred, target):
    smooth = 1e-6
    pred = pred.view(-1)
    target = target.view(-1)
    TP = (pred * target).sum()
    FP = ((1 - target) * pred).sum()
    FN = ((1 - pred) * target).sum()
    return (TP) / (TP + FP + FN + smooth)