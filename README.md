# g-Inspector
*g-Inspector* is a recurrent attention model for graph objects classification, which applies the attention mechanism to investigate the significance of each region to classification.

This repository provides a reference implementation of *g-Inspector*.

## FILES
*g-Inspector* source code package contains following files and folders.

* gInspector.py
* loader.py
* datasets/

## DATASETS

In this example, mutag dataset is provided which has been embedded with [DeepWalk](https://github.com/phanein/deepwalk). You can also use other graph embedding methods such as [node2vec](https://github.com/aditya-grover/node2vec). In order to use your own data, you have to provide

- a graphs' label txt file, eg. `dataset/mutag_labels.txt`
- a graphs' embedding folder, eg. `dataset/mutag/`

You can specify a dataset as follows:

```bash
python gInspector.py --data mutag
```

## USAGE

##### Example

To run *g-Inspector* on mutag, execute the following command from the project home directory:

```bash
python gInspector.py
```

##### Options

You can check out the other options available to use with *g-Inspector* using:

```bash
python gInspector.py --help 
```

## Prerequisites

Python 2.7 is required for *g-Inspector*.

Besides, following libs are needed:

* tensorflow
* numpy
* scikit-learn

These codes are tested on Mac OS and ubuntu. 

## contacts
Dr. Zhiling Luo luozhiling@zju.edu.cn http://www.bruceluo.net
Yinghua Cui yhcui@zju.edu.cn
