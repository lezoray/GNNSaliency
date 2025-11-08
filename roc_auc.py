import os
import numpy as np 
import scipy
from pathlib import Path
from sklearn import metrics
from sklearn.metrics import roc_auc_score

#PID files of the schelling dataset
meshes_directory="/Users/lezoray/Recherche/Donnees/3D/EyeTrackerSaliency/SchellingPoints/SchellingData/Points"

auc_a=np.empty(0)
optimal_thresholds=np.empty(0)

for filename in os.listdir(meshes_directory):
    filename_noextension=f"{Path(filename).stem}"
    if filename.endswith('.pid'):

        point_filename=f"{meshes_directory}/{filename_noextension}.pid"
        points = np.loadtxt(point_filename)

        data_filename = f"./outputs/{filename_noextension}_S.txt"
        data = np.loadtxt(data_filename)

        vertices = np.arange(len(data))
        res=np.isin(vertices, points)
        res2=res.astype(int)

        auc=roc_auc_score(res2, data)
        auc_a=np.append(auc_a,auc)

        fpr, tpr, thresholds = metrics.roc_curve(res2,data)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        optimal_thresholds=np.append(optimal_thresholds,optimal_threshold)

print("AUC")
print(np.average(auc_a)," ",np.std(auc_a))
print("Optimal Threshold")
print(np.average(optimal_thresholds))

std=np.std(optimal_thresholds)
thres=np.average(optimal_thresholds)

# set a set of files to process
l=["230","381","380","281"]

for filename in l:
    data_filename = f"./outputsFeastConv/{filename}_S.txt"
    data = np.loadtxt(data_filename)
    points=np.argwhere(data>=thres)
    np.savetxt(f"./outputsFeastConv/{filename}.pid",points)









 