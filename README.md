# SARMA

Official implementation of Saliency Prediction on 3D Meshes Using Residual FeaStConv-Based Graph Neural Networks

[Olivier Lézoray](https://lezoray.users.greyc.fr/)

## [Paper](https://) | [Project page](https://lezoray.users.greyc.fr/projects/VCIP2025/)

The Schelling Dataset can be downloaded [here](https://drive.google.com/file/d/1rg7rCYeUY3Vm80wrFe0OFOan9KXzzXiF/view?usp=sharing)

## Using the code

The model is in `MeshGNN.py`. The dataset loader in `MeshDataset.py`.

The file `main.py` can be used to train SARMA on the Schelling Dataset. You can also use the weights `best_model_FeastConv.pt` from our training.
The files `mainGAT.py` and `mainGCN` correspond to the SARMA model with GAT and GCN convolution layers.

If you want to generate the saliency maps for the dataset, use `inference.py`.
If you want to generate the predicted interest points, use `roc_auc.py`
Once done, you can :
- Visualize a mesh, its ground truth and predicted saliencies with `plotMeshSaliency.py`
- Visualize a mesh and the predicted points with `plotMeshPointsPS.py`
- Save a mesh with its predicted overlaid with `saveMeshColored.py`

## Citation

If you found SARMA helpful, please cite our paper:
```bibtex
@inproceedings{cin-Lezoray-2025-2,
	author = {O. Lézoray and Z. Ibork and A. Nouri and C. Charrier},
	booktitle = {Visual Communications and Image Processing (VCIP)},
	editor = {IEEE},
	title = {Saliency Prediction on 3D Meshes Using Residual FeaStConv-Based Graph Neural Networks},
	volume = {to appear},
	year = {2025}
}
```