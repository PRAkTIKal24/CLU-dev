"""Python-3 port of the AUTHORS' OWN SAM-kNN reference implementation.

Source: github.com/vlosing/SAMkNN  ->  SAMKNN/SAMKNN.py  (Losing, Hammer, Wersing, ICDM 2016)
Fetched 2026-08-10; original is Python 2 and depends on the C extension `libNearestNeighbor`
(nearestNeighbor/nearestNeighbor.cpp, also fetched).

Changes made, EXHAUSTIVE list (nothing else touched):
  1. imports (py2 implicit-relative -> explicit), `libNearestNeighbor` -> `_nn` numpy shims below.
  2. py2 int division restored with `//` at the 3 sites where py2 `/` on ints was floor division:
     clusterDown newLength, and the two `numSamplesRange[-1]/2` window ladders.
  3. removed the visualizer/listener plumbing (no behavioural effect) and the logging calls.
  4. sklearn KMeans call kept identical (n_init=1, random_state=0); modern sklearn needs no change.
The `_nn` shims reproduce nearestNeighbor.cpp exactly:
  * get1ToNDistances  -> SQUARED euclidean (SAMKNN.getDistances takes the sqrt)
  * nArgMin(n, v)     -> indices of the n smallest, ties broken by LOWEST INDEX, shape (rows, n)
  * mostCommon        -> majority label, ties broken by LOWEST LABEL
  * getLinearWeightedLabels -> argmax_l sum_i[label_i==l] 1/max(d_i, 1e-9), ties -> LOWEST LABEL
"""

import numpy as np
from collections import deque
from sklearn.cluster import KMeans


# ---------------------------------------------------------------- libNearestNeighbor shims
class _nn:
    @staticmethod
    def get1ToNDistances(sample, samples):
        d = samples - sample
        return np.einsum("ij,ij->i", d, d)

    @staticmethod
    def nArgMin(n, values):
        v = np.atleast_2d(values)
        n = int(n)
        out = np.empty((v.shape[0], min(n, v.shape[1])), dtype=np.int64)
        for r in range(v.shape[0]):
            out[r] = np.argsort(v[r], kind="stable")[:n]
        return out

    @staticmethod
    def mostCommon(values):
        v = np.atleast_2d(values)
        res = np.empty(v.shape[0], dtype=np.int64)
        for r in range(v.shape[0]):
            lab, cnt = np.unique(v[r], return_counts=True)  # lab ascending
            res[r] = lab[np.argmax(cnt)]                    # first max == lowest label
        return res

    @staticmethod
    def getLinearWeightedLabels(labels, distances):
        L = np.atleast_2d(labels)
        D = np.atleast_2d(distances)
        res = np.empty(L.shape[0], dtype=np.int64)
        for r in range(L.shape[0]):
            w = 1.0 / np.maximum(D[r], 1e-9)
            lab = np.unique(L[r])                            # ascending
            s = np.array([w[L[r] == l].sum() for l in lab])
            res[r] = lab[np.argmax(s)]                       # first max == lowest label
        return res


# ---------------------------------------------------------------- the authors' class, ported
class SAMKNN(object):
    def __init__(self, n_neighbors=5, knnWeights='distance', maxSize=5000, LTMSizeProportion=0.4,
                 minSTMSize=50, recalculateSTMError=False, useLTM=True):
        self.n_neighbors = n_neighbors
        self._STMSamples = None
        self._STMLabels = np.empty(shape=(0), dtype=np.int32)
        self._LTMSamples = None
        self._LTMLabels = np.empty(shape=(0), dtype=np.int32)
        self.maxLTMSize = LTMSizeProportion * maxSize
        self.maxSTMSize = maxSize - self.maxLTMSize
        self.minSTMSize = minSTMSize

        self.recalculateSTMError = recalculateSTMError
        if recalculateSTMError is not None:
            self.STMDistances = np.zeros(shape=(maxSize + 1, maxSize + 1))
        if knnWeights == 'distance':
            self.getLabelsFct = SAMKNN.getDistanceWeightedLabel
        elif knnWeights == 'uniform':
            self.getLabelsFct = SAMKNN.getMajLabel
        self.useLTM = useLTM
        if useLTM:
            self.predictFct = self.predictByAllMemories
            self.sizeCheckFct = self.sizeCheckSTMLTM
        else:
            self.predictFct = self.predictBySTM
            self.sizeCheckFct = self.sizeCheckFadeOut

        self.interLeavedPredHistories = {}
        self.LTMPredHistory = deque([])
        self.STMPredHistory = deque([])
        self.CMPredHistory = deque([])

        self.trainStepCount = 0
        self.STMSizes = []
        self.LTMSizes = []
        self.numSTMCorrect = 0
        self.numLTMCorrect = 0
        self.numCMCorrect = 0
        self.numPossibleCorrectPredictions = 0
        self.numCorrectPredictions = 0
        self.classifierChoice = []

    @staticmethod
    def getDistances(sample, samples):
        return np.sqrt(_nn.get1ToNDistances(sample, samples))

    def clusterDown(self, samples, labels):
        uniqueLabels = np.unique(labels)
        newSamples = np.empty(shape=(0, samples.shape[1]))
        newLabels = np.empty(shape=(0), dtype=np.int32)
        for label in uniqueLabels:
            tmpSamples = samples[labels == label]
            newLength = int(max(tmpSamples.shape[0] // 2, 1))
            clustering = KMeans(n_clusters=newLength, n_init=1, random_state=0)
            clustering.fit(tmpSamples)
            newSamples = np.vstack([newSamples, clustering.cluster_centers_])
            newLabels = np.append(newLabels, label * np.ones(shape=newLength, dtype=np.int32))
        return newSamples, newLabels

    def sizeCheckFadeOut(self):
        STMShortened = False
        if len(self._STMLabels) > self.maxSTMSize + self.maxLTMSize:
            STMShortened = True
            self._STMSamples = np.delete(self._STMSamples, 0, 0)
            self._STMLabels = np.delete(self._STMLabels, 0, 0)
            if self.recalculateSTMError is not None:
                self.STMDistances[:len(self._STMLabels), :len(self._STMLabels)] = \
                    self.STMDistances[1:len(self._STMLabels) + 1, 1:len(self._STMLabels) + 1]
            if not self.recalculateSTMError:
                if 0 in self.interLeavedPredHistories:
                    self.interLeavedPredHistories[0].pop(0)
                for key in list(self.interLeavedPredHistories.keys()):
                    if key > 0:
                        if key == 1:
                            self.interLeavedPredHistories.pop(0, None)
                        tmp = self.interLeavedPredHistories[key]
                        self.interLeavedPredHistories.pop(key, None)
                        self.interLeavedPredHistories[key - 1] = tmp
            else:
                self.interLeavedPredHistories = {}
        return STMShortened

    def sizeCheckSTMLTM(self):
        STMShortened = False
        if len(self._STMLabels) + len(self._LTMLabels) > self.maxSTMSize + self.maxLTMSize:
            if len(self._LTMLabels) > self.maxLTMSize:
                self._LTMSamples, self._LTMLabels = self.clusterDown(self._LTMSamples, self._LTMLabels)
            else:
                if len(self._STMLabels) + len(self._LTMLabels) > self.maxSTMSize + self.maxLTMSize:
                    STMShortened = True
                    numShifts = int(self.maxLTMSize - len(self._LTMLabels) + 1)
                    shiftRange = range(numShifts)
                    self._LTMSamples = np.vstack([self._LTMSamples, self._STMSamples[:numShifts, :]])
                    self._LTMLabels = np.append(self._LTMLabels, self._STMLabels[:numShifts])
                    self._LTMSamples, self._LTMLabels = self.clusterDown(self._LTMSamples, self._LTMLabels)
                    self._STMSamples = np.delete(self._STMSamples, shiftRange, 0)
                    self._STMLabels = np.delete(self._STMLabels, shiftRange, 0)
                    self.STMDistances[:len(self._STMLabels), :len(self._STMLabels)] = \
                        self.STMDistances[numShifts:len(self._STMLabels) + numShifts,
                                          numShifts:len(self._STMLabels) + numShifts]
                    for i in shiftRange:
                        self.LTMPredHistory.popleft()
                        self.STMPredHistory.popleft()
                        self.CMPredHistory.popleft()
                    self.interLeavedPredHistories = {}
        return STMShortened

    def cleanSamples(self, samplesCl, labelsCl, onlyLast=False):
        if self._STMLabels.shape[0] > self.n_neighbors and samplesCl.shape[0] > 0:
            if onlyLast:
                loopRange = [len(self._STMLabels) - 1]
            else:
                loopRange = range(len(self._STMLabels))
            for i in loopRange:
                if len(labelsCl) == 0:
                    break
                samplesShortened = np.delete(self._STMSamples, i, 0)
                labelsShortened = np.delete(self._STMLabels, i, 0)
                distancesSTM = SAMKNN.getDistances(self._STMSamples[i, :], samplesShortened)
                nnIndicesSTM = _nn.nArgMin(self.n_neighbors, distancesSTM)[0]
                distancesLTM = SAMKNN.getDistances(self._STMSamples[i, :], samplesCl)
                nnIndicesLTM = _nn.nArgMin(min(len(distancesLTM), self.n_neighbors), distancesLTM)[0]
                correctIndicesSTM = nnIndicesSTM[labelsShortened[nnIndicesSTM] == self._STMLabels[i]]
                if len(correctIndicesSTM) > 0:
                    distThreshold = np.max(distancesSTM[correctIndicesSTM])
                    wrongIndicesLTM = nnIndicesLTM[labelsCl[nnIndicesLTM] != self._STMLabels[i]]
                    delIndices = np.where(distancesLTM[wrongIndicesLTM] <= distThreshold)[0]
                    samplesCl = np.delete(samplesCl, wrongIndicesLTM[delIndices], 0)
                    labelsCl = np.delete(labelsCl, wrongIndicesLTM[delIndices], 0)
        return samplesCl, labelsCl

    def singleFit(self, sample, sampleLabel, distancesSTM):
        if self._STMSamples is None:
            self._STMSamples = np.empty(shape=(0, sample.shape[0]))
            self._LTMSamples = np.empty(shape=(0, sample.shape[0]))

        self.trainStepCount += 1
        self._STMSamples = np.vstack([self._STMSamples, sample])
        self._STMLabels = np.append(self._STMLabels, sampleLabel)
        STMShortened = self.sizeCheckFct()

        self._LTMSamples, self._LTMLabels = self.cleanSamples(self._LTMSamples, self._LTMLabels, onlyLast=True)

        if self.recalculateSTMError is not None:
            if STMShortened:
                distancesSTM = SAMKNN.getDistances(sample, self._STMSamples[:-1, :])

            self.STMDistances[len(self._STMLabels) - 1, :len(self._STMLabels) - 1] = distancesSTM
            oldWindowSize = len(self._STMLabels)
            newWindowSize, self.interLeavedPredHistories = STMSizer.getNewSTMSize(
                self.recalculateSTMError, self._STMLabels, self.n_neighbors, self.getLabelsFct,
                self.interLeavedPredHistories, self.STMDistances, self.minSTMSize)

            if newWindowSize < oldWindowSize:
                delrange = range(oldWindowSize - newWindowSize)
                oldSTMSamples = self._STMSamples[delrange, :]
                oldSTMLabels = self._STMLabels[delrange]
                self._STMSamples = np.delete(self._STMSamples, delrange, 0)
                self._STMLabels = np.delete(self._STMLabels, delrange, 0)
                d0 = oldWindowSize - newWindowSize
                self.STMDistances[:len(self._STMLabels), :len(self._STMLabels)] = \
                    self.STMDistances[d0:d0 + len(self._STMLabels), d0:d0 + len(self._STMLabels)]

                if self.useLTM:
                    for i in delrange:
                        self.STMPredHistory.popleft()
                        self.LTMPredHistory.popleft()
                        self.CMPredHistory.popleft()
                    oldSTMSamples, oldSTMLabels = self.cleanSamples(oldSTMSamples, oldSTMLabels)
                    self._LTMSamples = np.vstack([self._LTMSamples, oldSTMSamples])
                    self._LTMLabels = np.append(self._LTMLabels, oldSTMLabels)
                    self.sizeCheckFct()
        self.STMSizes.append(len(self._STMLabels))
        self.LTMSizes.append(len(self._LTMLabels))

    def _partial_fit(self, sample, sampleLabel):
        distancesSTM = SAMKNN.getDistances(sample, self._STMSamples)
        predictedLabel = self.predictFct(sample, sampleLabel, distancesSTM)
        self.singleFit(sample, sampleLabel, distancesSTM)
        return predictedLabel

    def predictByAllMemories(self, sample, label, distancesSTM):
        predictedLabelLTM = 0
        predictedLabelSTM = 0
        predictedLabelCM = 0
        classifierChoice = 0
        predictedLabel = 0
        if len(self._STMLabels) > 0:
            predictedLabelSTM = self.getLabelsFct(distancesSTM, self._STMLabels,
                                                  min(len(self._STMLabels), self.n_neighbors))[0]
            distancesLTM = SAMKNN.getDistances(sample, self._LTMSamples)
            predictedLabelCM = self.getLabelsFct(
                np.append(distancesSTM, distancesLTM),
                np.append(self._STMLabels, self._LTMLabels),
                min(len(self._STMLabels) + len(self._LTMLabels), self.n_neighbors))[0]
            if len(self._LTMLabels) > 0:
                predictedLabelLTM = self.getLabelsFct(distancesLTM, self._LTMLabels,
                                                      min(len(self._LTMLabels), self.n_neighbors))[0]

            labels = [predictedLabelSTM, predictedLabelLTM, predictedLabelCM]
            correctSTM = np.sum(self.STMPredHistory)
            correctLTM = np.sum(self.LTMPredHistory)
            correctCM = np.sum(self.CMPredHistory)
            classifierChoice = np.argmax([correctSTM, correctLTM, correctCM])
            predictedLabel = labels[classifierChoice]

        self.classifierChoice.append(classifierChoice)
        self.CMPredHistory.append(predictedLabelCM == label)
        self.numCMCorrect += predictedLabelCM == label
        self.STMPredHistory.append(predictedLabelSTM == label)
        self.numSTMCorrect += predictedLabelSTM == label
        self.LTMPredHistory.append(predictedLabelLTM == label)
        self.numLTMCorrect += predictedLabelLTM == label
        self.numPossibleCorrectPredictions += label in [predictedLabelSTM, predictedLabelCM, predictedLabelLTM]
        self.numCorrectPredictions += predictedLabel == label
        return predictedLabel

    def predictBySTM(self, sample, label, distancesSTM):
        predictedLabel = 0
        currLen = len(self._STMLabels)
        if currLen > 0:
            predictedLabel = self.getLabelsFct(distancesSTM, self._STMLabels,
                                               min(self.n_neighbors, currLen))[0]
        return predictedLabel

    def alternateFitPredict(self, samples, labels, classes=None, progress=None):
        if self._STMSamples is None:
            self._STMSamples = np.empty(shape=(0, samples.shape[1]))
            self._LTMSamples = np.empty(shape=(0, samples.shape[1]))
        predictedTrainLabels = []
        _labels = labels.astype(np.int32)
        for i in range(len(_labels)):
            predictedTrainLabels.append(self._partial_fit(samples[i, :], _labels[i]))
            if progress is not None and (i + 1) % progress == 0:
                print("  %d/%d" % (i + 1, len(_labels)), flush=True)
        return np.array(predictedTrainLabels)

    @staticmethod
    def getMajLabel(distances, labels, numNeighbours):
        nnIndices = _nn.nArgMin(numNeighbours, distances)
        return _nn.mostCommon(labels[nnIndices])

    @staticmethod
    def getDistanceWeightedLabel(distances, labels, numNeighbours):
        nnIndices = _nn.nArgMin(numNeighbours, distances)
        return _nn.getLinearWeightedLabels(labels[nnIndices], distances[nnIndices])


class STMSizer(object):
    @staticmethod
    def getNewSTMSize(recalculateSTMError, labels, nNeighbours, getLabelsFct, predictionHistories,
                      distancesSTM, minSTMSize):
        if recalculateSTMError is None:
            return len(labels), predictionHistories
        elif recalculateSTMError:
            return STMSizer.getMinErrorRateWindowSize(labels, nNeighbours, getLabelsFct,
                                                      predictionHistories, distancesSTM, minSize=minSTMSize)
        else:
            return STMSizer.getMinErrorRateWindowSizeIncremental(labels, nNeighbours, getLabelsFct,
                                                                 predictionHistories, distancesSTM,
                                                                 minSize=minSTMSize)

    @staticmethod
    def errorRate(predLabels, labels):
        return 1 - np.sum(np.asarray(predLabels) == labels) / float(len(predLabels))

    @staticmethod
    def getInterleavedTestTrainErrorRate(labels, nNeighbours, getLabelsFct, distancesSTM):
        predLabels = []
        for i in range(nNeighbours, len(labels)):
            distances = distancesSTM[i, :i]
            predLabels.append(getLabelsFct(distances, labels[:i], nNeighbours)[0])
        return (STMSizer.errorRate(predLabels[:], labels[nNeighbours:]),
                (np.asarray(predLabels) == labels[nNeighbours:]).tolist())

    @staticmethod
    def getIncrementalInterleavedTestTrainErrorRate(labels, nNeighbours, getLabelsFct, predictionHistory,
                                                    distancesSTM):
        for i in range(len(predictionHistory) + nNeighbours, len(labels)):
            distances = distancesSTM[i, :i]
            label = getLabelsFct(distances, labels[:i], nNeighbours)[0]
            predictionHistory.append(label == labels[i])
        return 1 - np.sum(predictionHistory) / float(len(predictionHistory)), predictionHistory

    @staticmethod
    def adaptHistories(numberOfDeletions, predictionHistories):
        for i in range(numberOfDeletions):
            sortedKeys = np.sort(list(predictionHistories.keys()))
            predictionHistories.pop(sortedKeys[0], None)
            delta = sortedKeys[1]
            for j in range(1, len(sortedKeys)):
                predictionHistories[sortedKeys[j] - delta] = predictionHistories.pop(sortedKeys[j])
        return predictionHistories

    @staticmethod
    def getMinErrorRateWindowSize(labels, nNeighbours, getLabelsFct, predictionHistories, distancesSTM,
                                  minSize=50):
        numSamples = len(labels)
        if numSamples < 2 * minSize:
            return numSamples, predictionHistories
        else:
            numSamplesRange = [numSamples]
            while numSamplesRange[-1] // 2 >= minSize:
                numSamplesRange.append(numSamplesRange[-1] // 2)
            errorRates = []
            for key in list(predictionHistories.keys()):
                if key not in (numSamples - np.array(numSamplesRange)):
                    predictionHistories.pop(key, None)
            for numSamplesIt in numSamplesRange:
                idx = int(numSamples - numSamplesIt)
                if idx in predictionHistories:
                    errorRate, predHistory = STMSizer.getIncrementalInterleavedTestTrainErrorRate(
                        labels[idx:], nNeighbours, getLabelsFct, predictionHistories[idx],
                        distancesSTM[idx:, idx:])
                else:
                    errorRate, predHistory = STMSizer.getInterleavedTestTrainErrorRate(
                        labels[idx:], nNeighbours, getLabelsFct, distancesSTM[idx:, idx:])
                predictionHistories[idx] = predHistory
                errorRates.append(errorRate)
            errorRates = np.round(errorRates, decimals=4)
            bestNumTrainIdx = np.argmin(errorRates)
            windowSize = numSamplesRange[bestNumTrainIdx]
            if windowSize < numSamples:
                predictionHistories = STMSizer.adaptHistories(bestNumTrainIdx, predictionHistories)
            return int(windowSize), predictionHistories

    @staticmethod
    def getMinErrorRateWindowSizeIncremental(labels, nNeighbours, getLabelsFct, predictionHistories,
                                             distancesSTM, minSize=50):
        numSamples = len(labels)
        if numSamples < 2 * minSize:
            return numSamples, predictionHistories
        else:
            numSamplesRange = [numSamples]
            while numSamplesRange[-1] // 2 >= minSize:
                numSamplesRange.append(numSamplesRange[-1] // 2)
            errorRates = []
            for numSamplesIt in numSamplesRange:
                idx = int(numSamples - numSamplesIt)
                if idx in predictionHistories:
                    errorRate, predHistory = STMSizer.getIncrementalInterleavedTestTrainErrorRate(
                        labels[idx:], nNeighbours, getLabelsFct, predictionHistories[idx],
                        distancesSTM[idx:, idx:])
                elif idx - 1 in predictionHistories:
                    predHistory = predictionHistories[idx - 1]
                    predictionHistories.pop(idx - 1, None)
                    predHistory.pop(0)
                    errorRate, predHistory = STMSizer.getIncrementalInterleavedTestTrainErrorRate(
                        labels[idx:], nNeighbours, getLabelsFct, predHistory, distancesSTM[idx:, idx:])
                else:
                    errorRate, predHistory = STMSizer.getInterleavedTestTrainErrorRate(
                        labels[idx:], nNeighbours, getLabelsFct, distancesSTM[idx:, idx:])
                predictionHistories[idx] = predHistory
                errorRates.append(errorRate)
            errorRates = np.round(errorRates, decimals=4)
            bestNumTrainIdx = np.argmin(errorRates)
            if bestNumTrainIdx > 0:
                moreAccurateIndices = np.where(errorRates < errorRates[0])[0]
                for i in moreAccurateIndices:
                    idx = int(numSamples - numSamplesRange[i])
                    errorRate, predHistory = STMSizer.getInterleavedTestTrainErrorRate(
                        labels[idx:], nNeighbours, getLabelsFct, distancesSTM[idx:, idx:])
                    predictionHistories[idx] = predHistory
                    errorRates[i] = errorRate
                errorRates = np.round(errorRates, decimals=4)
                bestNumTrainIdx = np.argmin(errorRates)
            windowSize = numSamplesRange[bestNumTrainIdx]
            if windowSize < numSamples:
                predictionHistories = STMSizer.adaptHistories(bestNumTrainIdx, predictionHistories)
            return int(windowSize), predictionHistories
