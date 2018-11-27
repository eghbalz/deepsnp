# DeepSNP

This repository contains the code for the paper

**Deep SNP: An End-to-end Deep Neural Network with Attention-based Localization for Break-point Detection in SNP Array Genomic Data**
<br>
[Hamid Eghbal-zadeh](https://www.jku.at/en/institute-of-computational-perception/about-us/people/hamid-eghbal-zadeh/), [Lukas Fischer](https://www.scch.at/en/team/person_id/207), [Niko Popitsch](http://science.ccri.at/contact-us/contact-details/), [Florian Kromp](http://science.ccri.at/contact-us/contact-details/), [Sabine Taschner-Mandl](http://science.ccri.at/contact-us/contact-details/), [Teresa Gerber](http://science.ccri.at/contact-us/contact-details/), [Eva Bozsaky](http://science.ccri.at/contact-us/contact-details/), [Peter F. Ambros](http://science.ccri.at/contact-us/contact-details/), [Inge M. Ambros](http://science.ccri.at/contact-us/contact-details/), [Gerhard Widmer](https://www.jku.at/en/institute-of-computational-perception/about-us/people/gerhard-widmer/), [Bernhard A. Moser](https://www.scch.at/en/team/person_id/90)

To be published in [JCB](https://home.liebertpub.com/publications/journal-of-computational-biology/31/overview)

## Citing DeepSNP

If you use DeepSNP in your research, please cite the following BibTeX entry:

```
@article{eghbal2018deepsnp,
    author={Eghbal-zadeh, Hamid and Fischer, Lukas and Popitsch, Niko and Kromp, Florian and Taschner-Mandl, Sabine and Gerber, Teresa and Bozsaky, Eva and Ambros, Peter and Ambros, Inge and Widmer, Gerhard and Moser, Bernhard},
    title={Deep SNP: An End-to-end Deep Neural Network with Attention-based Localization for Break-point Detection in SNP Array Genomic Data},
    journal = {Journal of Computational Biology},
    volume = {},
    number = {},
    pages = {},
    year = {2019},
    doi = {},
    note ={PMID: },
    URL = {https://doi.org/}
}
```

## Setup
DeepSNP was developed using Python 3.6 (Anaconda), Keras 2.2.2 and Tensorflow 1.9 and was tested under Windows 10 and Ubuntu 16.04.

```
pip install -r requirements.txt
```

## Data preparation
DeepSNP expects windowed data from SNPa sequences. To prepare your data follow the instructions described in [TRAINING](TRAINING.md).

## Configuration
Experiment ID and run mode (training, inference (default), data_from_raw, eval_rawcopy) can be passed to the main.py function.
<br>
Each experiment ID needs a configuration file (inheriting the default configuration: [configs/config.py](configs/config.py)) 

For example to run inference with DeepSNPv1 with no attention use:
```
python main.py --exp_id DeepSNP_V1_noAtt' --mode inference
```
This wil load the configuration defined in: 
```
configs/config_DeepSNP_V1_noAtt.py
```

## Training from scratch
To train the provided models from scratch please refer to [TRAINING](TRAINING.md).

## Evaluate Rawcopy breakpoint predictions
To evaluate the breakpoint prediction performance of [Rawcopy](http://rawcopy.org/) on a certain data set use:
```
python main.py --mode eval_rawcopy
```

## Acknowledgements
This work was carried out within the [Austrian Research Promotion Agency (FFG)](https://www.ffg.at/en) COIN "Networks" project VISIOMICS together with [St. Anna Kinderkrebsforschung](http://science.ccri.at/) and additionally supported by the Austrian Ministry for Transport, Innovation and Technology, the Federal Ministry of Science, Research and Economy, and the Province of Upper Austria in the frame of the COMET center [SCCH](https://www.scch.at/en/news). The authors gratefully acknowledge the support of NVIDIA Corporation with the donation of a Titan~X GPU used for this research.