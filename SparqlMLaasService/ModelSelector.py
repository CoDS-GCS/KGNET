import numpy
import pandas as pd
class ModelSelector:
     def __init__(self):
        ""
     @staticmethod
     def getBestModelIdx(lstParams, w1=0.7, w2=0.3):
        'lst of accuracy and inferTime pairs'
        scores = []
        for m in lstParams:
            # print(m, w1 * float(m[0]) , w2 * float(m[1]), w1 * float(m[0])- w2 * float(m[1]))
            scores.append(w1 * float(m[0]) - w2 * float(m[1]))
        return numpy.argmax(scores)

     @staticmethod
     def getBestPlanIdx(lstParams, w1=0.7, w2=0.3):
         'lst of accuracy and inferTime pairs'
         scores = []
         for m in lstParams:
             # print(m, w1 * float(m[0]) , w2 * float(m[1]), w1 * float(m[0])- w2 * float(m[1]))
             scores.append(w1 * float(m[0]) - w2 * float(m[1]))
         return numpy.argmax(scores)

if __name__ == '__main__':
    ""
    # models_lst=[[5,10],[8,9],[20,40],[5,9]]
    # df= pd.DataFrame(models_lst,columns=["Accuracy", "InferenceTime"])
    # gml_models_lst = df[["Accuracy", "InferenceTime"]].values.tolist()
    # res=ModelSelector.optimize_ModelSelection(gml_models_lst)
    # print(res)