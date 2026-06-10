import numpy as np
import pandas as pd


'''Based on the decision treee from the slides/report. 
The function takes as input a list of candidates, the sample table, 
the inferred outputs from the current best candidate and the current best candidate.
The candiates and the current best candidate needs to be strings representing the mathematical expression of the function.
The function exprects the LAST column of the table to be the true output and the other columns to be the input parameters.
The function also expects that the inferred outputs are in the same order as the rows of the table and not represented in the table.
For the inputs and registers, the function expects that they are named i0, i1, ... and r0, r1, ... respectively.
It returns a list of probable candidates and a list of not probable candidates.'''

def getProbableCandidates(candidates, table, inferredOutputs, currentBestCandidate):
    probableCandidates = []
    notProbableCandidates = []
    correlationsResiduals = {}
    correlationsTrueOut = {}
    columns = table.columns
    trueOut = table[columns[-1]].values
    possibleParams = columns[:-1] # exclude output column
    residuals = trueOut - inferredOutputs
    if trueOut.all() == trueOut[0]:  # if all true outputs are the same, correlation is not defined
        for candidate in candidates:
            if candidate.isdigit():
                probableCandidates.append(candidate)
            elif candidate in possibleParams:
                if table[candidate].nunique() == 1: 
                    probableCandidates.append(candidate)
            else:
                notProbableCandidates.append(candidate)
        return probableCandidates, notProbableCandidates
    elif residuals.all() == residuals[0]:  # if all residuals are the same, correlation is not defined
        if residuals[0] == 0:
            print("All inferred outputs are correct, current best candidate is likely correct.")
            probableCandidates.append(currentBestCandidate)
            notProbableCandidates = [c for c in candidates if c != currentBestCandidate]
            return probableCandidates, notProbableCandidates
        else:
            print("All inferred outputs have the same error, current best candidate is likely misses a constant.")
            for candidate in candidates:
                candidateWOcurrentBest = candidate.replace(currentBestCandidate, '')
                candiateWOcurrentBest = candidateWOcurrentBest.replace('()', '')
                candidateWOcurrentBest = candidateWOcurrentBest.replace('add', '')
                candidateWOcurrentBest = candidateWOcurrentBest.replace('sub', '')
                candidateWOcurrentBest = candidateWOcurrentBest.replace('-', '')
                if candidateWOcurrentBest in possibleParams:
                    probableCandidates.append(candidate)
                elif candidateWOcurrentBest.isdigit():
                    probableCandidates.append(candidate)
                else:
                    notProbableCandidates.append(candidate)
            return probableCandidates, notProbableCandidates
    else: 
        for param in possibleParams:
            if table[param].nunique() > 1:  # only consider parameters that vary in the table
                correlationsTrueOut[param] = np.corrcoef(table[param].values, trueOut)[0,1]
                correlationsResiduals[param] = np.corrcoef(table[param].values, residuals)[0,1]
            else:
                correlationsTrueOut[param] = -10  # if the parameter does not vary, correlation is not defined
                correlationsResiduals[param] = -10. # if the parameter does not vary, correlation is not defined
        peakThresholdRes = max(correlationsResiduals.values()) * 2/3 # set threshold at 2/3 of the maximum correlation with residuals
        peakThresholdTrueOut = max(correlationsTrueOut.values()) * 2/3 # set threshold at 2/3 of the maximum correlation with true output

        paramsInCurrentBestCandidate = [p for p in possibleParams if p in currentBestCandidate]

        shouldBeInProbable = [p for p in possibleParams if p in currentBestCandidate]
        shouldNotBeInProbable = [p for p in possibleParams if p not in currentBestCandidate]
        wrongInScale = []
        for p in possibleParams:
            if correlationsResiduals[p] > peakThresholdRes:
                if p in paramsInCurrentBestCandidate:
                    if correlationsTrueOut[p] > peakThresholdTrueOut:
                        if p not in shouldBeInProbable:
                            shouldBeInProbable.append(p)
                        if p in shouldNotBeInProbable:
                            shouldNotBeInProbable.remove(p)
                        wrongInScale.append(p)
                    else:
                        correspondingParam = p.replace('r', 'i') if 'r' in p else p.replace('i', 'r')
                        if correlationsTrueOut[correspondingParam] > peakThresholdTrueOut:
                            if p not in shouldNotBeInProbable:
                                shouldNotBeInProbable.append(p)
                            if p in shouldBeInProbable:
                                shouldBeInProbable.remove(p)
                            if correspondingParam in paramsInCurrentBestCandidate:
                                if correspondingParam not in shouldBeInProbable:
                                    shouldBeInProbable.append(correspondingParam)
                                if correspondingParam in shouldNotBeInProbable:
                                    shouldNotBeInProbable.remove(correspondingParam)
                else:
                    if correlationsTrueOut[p] > peakThresholdTrueOut:
                        if p not in shouldBeInProbable:
                            shouldBeInProbable.append(p)
                        if p in shouldNotBeInProbable:
                            shouldNotBeInProbable.remove(p)
                    else:
                        correspondingParam = p.replace('r', 'i') if 'r' in p else p.replace('i', 'r')
                        if correlationsTrueOut[correspondingParam] > peakThresholdTrueOut:
                            if p not in shouldNotBeInProbable:
                                shouldNotBeInProbable.append(p)
                            if p in shouldBeInProbable:
                                shouldBeInProbable.remove(p)
                            if correspondingParam in paramsInCurrentBestCandidate:
                                if correspondingParam not in shouldBeInProbable:
                                    shouldBeInProbable.append(correspondingParam)
                                if correspondingParam in shouldNotBeInProbable:
                                    shouldNotBeInProbable.remove(correspondingParam)
        for candidate in candidates:
            paramsInCandidate = [p for p in possibleParams if p in candidate]
            if all(p in shouldBeInProbable for p in paramsInCandidate) and all(p not in shouldNotBeInProbable for p in paramsInCandidate):
                probableCandidates.append(candidate)
            else:
                notProbableCandidates.append(candidate)
        return probableCandidates, notProbableCandidates
    

                


            
            


