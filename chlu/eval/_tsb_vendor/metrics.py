# Vendored from TSB-AD v1.5 (Apache-2.0), The Datum Lab.
#   Liu & Paparrizos, "The Elephant in the Room: Towards A Reliable Time-Series
#   Anomaly Detection Benchmark", NeurIPS 2024 Datasets & Benchmarks.
#   Source: https://github.com/TheDatumOrg/TSB-AD
#   sdist:  https://files.pythonhosted.org/packages/51/c8/b1656c4fba492904c79fec0891c952c9a5cbf4bc3f6e13714cf7741c183c/tsb_ad-1.5.tar.gz
#   sha256: 52e474cda6aeb3c2f8f6b3a45e58b11b5b7b55a1510bb5c6f6a15b9053f7b0da
# Modifications (see _tsb_vendor/README.md): point-adjust F1 removed
# (forbidden by the CHLU evaluation protocol; Kim et al., AAAI 2022).
# Do NOT edit by hand except through .claude/scratch/f2-eval-harness/vendor_tsb.py.
from .basic_metrics import basic_metricor, generate_curve

def get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250):
    metrics = {}

    '''
    Threshold Independent
    '''
    grader = basic_metricor()
    # [chlu vendor] REMOVED: point-adjust F1 line.
    AUC_ROC = grader.metric_ROC(labels, score)
    AUC_PR = grader.metric_PR(labels, score)

    # R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
    _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels.astype(int), score, slidingWindow, version, thre)


    '''
    Threshold Dependent
    if pred is None --> use the oracle threshold
    '''

    PointF1 = grader.metric_PointF1(labels, score, preds=pred)
    # [chlu vendor] REMOVED: point-adjust F1 line.
    EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)
    RF1 = grader.metric_RF1(labels, score, preds=pred)
    Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)

    metrics['AUC-PR'] = AUC_PR
    metrics['AUC-ROC'] = AUC_ROC
    metrics['VUS-PR'] = VUS_PR
    metrics['VUS-ROC'] = VUS_ROC

    metrics['Standard-F1'] = PointF1
    # [chlu vendor] REMOVED: point-adjust F1 line.
    metrics['Event-based-F1'] = EventF1PA
    metrics['R-based-F1'] = RF1
    metrics['Affiliation-F'] = Affiliation_F
    return metrics


def get_metrics_pred(score, labels, pred, slidingWindow=100):
    metrics = {}

    grader = basic_metricor()

    PointF1 = grader.metric_PointF1(labels, score, preds=pred)
    # [chlu vendor] REMOVED: point-adjust F1 line.
    EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)
    RF1 = grader.metric_RF1(labels, score, preds=pred)
    Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)
    VUS_R, VUS_P, VUS_F = grader.metric_VUS_pred(labels, preds=pred, windowSize=slidingWindow)

    metrics['Standard-F1'] = PointF1
    # [chlu vendor] REMOVED: point-adjust F1 line.
    metrics['Event-based-F1'] = EventF1PA
    metrics['R-based-F1'] = RF1
    metrics['Affiliation-F'] = Affiliation_F

    metrics['VUS-Recall'] = VUS_R
    metrics['VUS-Precision'] = VUS_P
    metrics['VUS-F'] = VUS_F

    return metrics
