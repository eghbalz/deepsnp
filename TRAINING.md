# TRAINING A MODEL FROM SCRATCH

If you want to train a model with your own data by following these steps:

## 1. Prepare the data
As indicated by its name, DeepSNP was developed for breakpoint detection in SNPa data.
Use the *data_from_raw MODE* ([main.py](main.py)) to convert your raw SNPa data.

The code expects your raw data stored as .dat (.csv like) files (e.g. exported from Rawcopy) containing the following information:
```
#Chr	Pos	LRR	BAF	tCN	cCN	GC	MAP
1	15253	0.2133	-1.0	2	2	0.58	0.02025
1	48168	-0.0587	-1.0	2	2	0.44	0.01094
1	60826	0.387	-1.0	2	2	0.38	0.02865
1	61722	-0.0517	-1.0	2	2	0.34	0.0164
```
* Chr: Chromosom number
* Pos: Position in the SNPa
* LRR: normalized Log Ratio
* BAF: B-allele frequency values
* tCN: Groundtruth labels
* cCN: Rawcopy predictions
* GC:
* MAP: 

LRR, BAF, tCN an cCN columns are mandatory! The others are not used right now and can be omitted.

The algorithm parses through the array and selects positive (with BPs) and negative (without BPs) windows with certain sizes (defined by the *config.hop_modifiers* parameter) and saves them as h5 files to *config.data_dir/features*.

## 2. Select a model
The following models are available:
* **Baseline**
    * **BLVGG**             - VGG like feed forward deep neural network
    * **BLDenseNet**        - DenseNet (Densely Connected Convolutional Network)
    * **BLDilDenseNet**     - Adapted DenseNet with dilated convolution
    * **BLLSTMDenseNet**    - Adapted DenseNet with LSTM (Long short-term memory)
* **DeepSNP**
    * **V1**: with dilated convolution layers
        * **DeepSNP_V1_noAtt**      - no attention unit
        * **DeepSNP_V1_finalAtt**   - with attention unit
    * **V2**: without dilation, but with conventional convolution layers
        * **DeepSNP_V2_noAtt**      - no attention unit
        * **DeepSNP_V2_finalAtt**   - with attention unit

## 3. Train a model
Each model has its own configuration file located in [/configs](configs), which inherits the ConfigFlags class ([config.py](configs/config.py) where you can set a multitude of parameters. (see [config.py](configs/config.py) for details)
