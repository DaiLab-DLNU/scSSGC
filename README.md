# scSSGC
This is the official codebase for **scSSGC: Self-supervised graph representation learning for single-cell classification.**.

<p align="center">
<img src="/images/title_page.png"/> 
</p>


### Python environment setup with Conda
#### 1. Install Python and Pytorch
```bash
conda create --name scSSGC python=3.8
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda activate scSSGC
conda clean --all
```
#### 2. Install torch_geometric
##### 2.1 Download the required dependencies
Go to the [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric?spm=5176.28103460.0.0.2a355d27SCtbt2) page and download the following files:
```bash
Download list：
torch_scatter-2.0.8-cp39-cp39-linux_x86_64.whl
torch_sparse-0.6.12-cp39-cp39-linux_x86_64.whl
torch_cluster-1.5.9-cp39-cp39-linux_x86_64.whl 
torch_spline_conv-1.2.1-cp39-cp39-linux_x86_64.whl
```
##### 2.1 Install the required dependencies and torch_geometric
```bash
pip install torch_scatter-2.0.8-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.12-cp39-cp39-linux_x86_64.whl
pip install torch_cluster-1.5.9-cp39-cp39-linux_x86_64.whl 
pip install torch_spline_conv-1.2.1-cp39-cp39-linux_x86_64.whl 
pip install torch_geometric

```
#### 3. Install others

```bash
pip install pandas fbpca faiss-gpu annoy matplotlib numpy==1.26.4 
```
### Running scSSGC (Remember to update the dataset addresses)
```bash
conda activate scSSGC
# Running scSSGC 
python main.py 
```

### Preprint and Citation

