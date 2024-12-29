# MARE: Multi-Apect Rationale Extractor for Unsupervised Rationale Extraction

This repository contains the source code associated with the paper titled "[MARE: Multi-Apect Rationale Extractor for Unsupervised Rationale Extraction](http://www.baidu.com)" accepted at EMNLP 2024. 
If you use the code or the models in this repository, please cite the following paper:


## Folder Structure
```
mare-master
    \-data
        \-beer
        \-hotel
    \-scripts
        run_classification.sh
        eval_classification.sh
    \-src
        \-module
        run_classification.py
    README.md
    requirements.txt
```


## Setup

Before you begin, ensure you have met the following requirements:

1. You have installed Python 3.9.11.
2. You have installed `pip`.
3. You have installed the necessary dependencies listed in `requirements.txt`.

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Quick Start
To get started, follow these steps:

Download the dataset and place it in the `data/` directory.

Train the model:
```bash
sh scripts/run_classification.sh
```
The trained model will be saved in the `model/` directory

Test the model:
Set the `OUTPUT_DIR` in `scripts/eval_classification.sh` to the model path.
```bash
sh scripts/eval_classification.sh
```

## Result
You can get similar results as follows:
```
##################### beer0 #####################
05/20/2024 08:55:54 - INFO - root -   tag_p: [0.95410772], tag_r: [0.89549672], tag_f1: [0.92387358]
05/20/2024 08:55:54 - INFO - __main__ -   evaluation, loss: 1.1810768842697144
05/20/2024 08:55:54 - INFO - __main__ -   acc: 0.8557692307692307
05/20/2024 08:55:54 - INFO - __main__ -   precision: 1.0, recall: 0.8537378114842904, f1: 0.9210987726475746
05/20/2024 08:55:54 - INFO - __main__ -   tokens_remained: 0.7946670651435852
05/20/2024 08:55:54 - INFO - __main__ -   mare_loss: 0.01414343062788248
05/20/2024 08:55:54 - INFO - __main__ -   layer 0, precision: 0.1843, recall: 1.0000, f1: 0.3112, sparsity: 0.1843, cur_sparsity: 0.9999
05/20/2024 08:55:54 - INFO - __main__ -   layer 1, precision: 0.1843, recall: 1.0000, f1: 0.3112, sparsity: 0.1843, cur_sparsity: 0.9997
05/20/2024 08:55:54 - INFO - __main__ -   layer 2, precision: 0.1843, recall: 1.0000, f1: 0.3113, sparsity: 0.1843, cur_sparsity: 0.9996
05/20/2024 08:55:54 - INFO - __main__ -   layer 3, precision: 0.1843, recall: 1.0000, f1: 0.3113, sparsity: 0.1843, cur_sparsity: 0.9996
05/20/2024 08:55:54 - INFO - __main__ -   layer 4, precision: 0.1843, recall: 1.0000, f1: 0.3113, sparsity: 0.1843, cur_sparsity: 0.9996
05/20/2024 08:55:54 - INFO - __main__ -   layer 5, precision: 0.1843, recall: 1.0000, f1: 0.3113, sparsity: 0.1843, cur_sparsity: 0.9996
05/20/2024 08:55:54 - INFO - __main__ -   layer 6, precision: 0.1843, recall: 1.0000, f1: 0.3113, sparsity: 0.1843, cur_sparsity: 0.9996
05/20/2024 08:55:54 - INFO - __main__ -   layer 7, precision: 0.1843, recall: 1.0000, f1: 0.3113, sparsity: 0.1843, cur_sparsity: 0.9996
05/20/2024 08:55:54 - INFO - __main__ -   layer 8, precision: 0.1843, recall: 1.0000, f1: 0.3113, sparsity: 0.1843, cur_sparsity: 0.9996
05/20/2024 08:55:54 - INFO - __main__ -   layer 9, precision: 0.9329, recall: 0.9169, f1: 0.9248, sparsity: 0.1843, cur_sparsity: 0.1811
05/20/2024 08:55:54 - INFO - __main__ -   layer 10, precision: 0.9494, recall: 0.9044, f1: 0.9263, sparsity: 0.1843, cur_sparsity: 0.1755
05/20/2024 08:55:54 - INFO - __main__ -   layer 11, precision: 0.9541, recall: 0.8955, f1: 0.9239, sparsity: 0.1843, cur_sparsity: 0.1730

##################### beer1 #####################
05/20/2024 08:56:00 - INFO - root -   tag_p: [0.93791382], tag_r: [0.90185611], tag_f1: [0.91953162]
05/20/2024 08:56:00 - INFO - __main__ -   evaluation, loss: 1.2223132848739624
05/20/2024 08:56:00 - INFO - __main__ -   acc: 0.8620296465222349
05/20/2024 08:56:00 - INFO - __main__ -   precision: 0.9959072305593452, recall: 0.8608490566037735, f1: 0.9234661606578115
05/20/2024 08:56:00 - INFO - __main__ -   tokens_remained: 0.7893163561820984
05/20/2024 08:56:00 - INFO - __main__ -   mare_loss: 0.05671863257884979
05/20/2024 08:56:00 - INFO - __main__ -   layer 0, precision: 0.1597, recall: 1.0000, f1: 0.2755, sparsity: 0.1597, cur_sparsity: 1.0000
05/20/2024 08:56:00 - INFO - __main__ -   layer 1, precision: 0.1597, recall: 1.0000, f1: 0.2755, sparsity: 0.1597, cur_sparsity: 0.9998
05/20/2024 08:56:00 - INFO - __main__ -   layer 2, precision: 0.1597, recall: 1.0000, f1: 0.2755, sparsity: 0.1597, cur_sparsity: 0.9998
05/20/2024 08:56:00 - INFO - __main__ -   layer 3, precision: 0.1598, recall: 1.0000, f1: 0.2755, sparsity: 0.1597, cur_sparsity: 0.9997
05/20/2024 08:56:00 - INFO - __main__ -   layer 4, precision: 0.1598, recall: 1.0000, f1: 0.2755, sparsity: 0.1597, cur_sparsity: 0.9997
05/20/2024 08:56:00 - INFO - __main__ -   layer 5, precision: 0.1598, recall: 1.0000, f1: 0.2755, sparsity: 0.1597, cur_sparsity: 0.9997
05/20/2024 08:56:00 - INFO - __main__ -   layer 6, precision: 0.1598, recall: 1.0000, f1: 0.2755, sparsity: 0.1597, cur_sparsity: 0.9997
05/20/2024 08:56:00 - INFO - __main__ -   layer 7, precision: 0.1598, recall: 1.0000, f1: 0.2755, sparsity: 0.1597, cur_sparsity: 0.9997
05/20/2024 08:56:00 - INFO - __main__ -   layer 8, precision: 0.1598, recall: 1.0000, f1: 0.2755, sparsity: 0.1597, cur_sparsity: 0.9997
05/20/2024 08:56:00 - INFO - __main__ -   layer 9, precision: 0.9170, recall: 0.9154, f1: 0.9162, sparsity: 0.1597, cur_sparsity: 0.1594
05/20/2024 08:56:00 - INFO - __main__ -   layer 10, precision: 0.9313, recall: 0.9065, f1: 0.9187, sparsity: 0.1597, cur_sparsity: 0.1555
05/20/2024 08:56:00 - INFO - __main__ -   layer 11, precision: 0.9379, recall: 0.9019, f1: 0.9195, sparsity: 0.1597, cur_sparsity: 0.1536

##################### beer2 #####################
05/20/2024 08:56:07 - INFO - root -   tag_p: [0.82200453], tag_r: [0.81883962], tag_f1: [0.82041902]
05/20/2024 08:56:07 - INFO - __main__ -   evaluation, loss: 0.9887437224388123
05/20/2024 08:56:07 - INFO - __main__ -   acc: 0.8770186335403727
05/20/2024 08:56:07 - INFO - __main__ -   precision: 0.9971014492753624, recall: 0.8764331210191083, f1: 0.9328813559322033
05/20/2024 08:56:07 - INFO - __main__ -   tokens_remained: 0.7830137610435486
05/20/2024 08:56:07 - INFO - __main__ -   mare_loss: 0.05881810560822487
05/20/2024 08:56:07 - INFO - __main__ -   layer 0, precision: 0.1272, recall: 1.0000, f1: 0.2257, sparsity: 0.1272, cur_sparsity: 1.0000
05/20/2024 08:56:07 - INFO - __main__ -   layer 1, precision: 0.1272, recall: 1.0000, f1: 0.2257, sparsity: 0.1272, cur_sparsity: 0.9999
05/20/2024 08:56:07 - INFO - __main__ -   layer 2, precision: 0.1272, recall: 1.0000, f1: 0.2257, sparsity: 0.1272, cur_sparsity: 0.9997
05/20/2024 08:56:07 - INFO - __main__ -   layer 3, precision: 0.1272, recall: 1.0000, f1: 0.2257, sparsity: 0.1272, cur_sparsity: 0.9997
05/20/2024 08:56:07 - INFO - __main__ -   layer 4, precision: 0.1272, recall: 1.0000, f1: 0.2257, sparsity: 0.1272, cur_sparsity: 0.9996
05/20/2024 08:56:07 - INFO - __main__ -   layer 5, precision: 0.1272, recall: 1.0000, f1: 0.2257, sparsity: 0.1272, cur_sparsity: 0.9996
05/20/2024 08:56:07 - INFO - __main__ -   layer 6, precision: 0.1272, recall: 1.0000, f1: 0.2257, sparsity: 0.1272, cur_sparsity: 0.9996
05/20/2024 08:56:07 - INFO - __main__ -   layer 7, precision: 0.1272, recall: 1.0000, f1: 0.2257, sparsity: 0.1272, cur_sparsity: 0.9996
05/20/2024 08:56:07 - INFO - __main__ -   layer 8, precision: 0.1272, recall: 1.0000, f1: 0.2257, sparsity: 0.1272, cur_sparsity: 0.9996
05/20/2024 08:56:07 - INFO - __main__ -   layer 9, precision: 0.7989, recall: 0.8292, f1: 0.8138, sparsity: 0.1272, cur_sparsity: 0.1320
05/20/2024 08:56:07 - INFO - __main__ -   layer 10, precision: 0.8164, recall: 0.8218, f1: 0.8191, sparsity: 0.1272, cur_sparsity: 0.1280
05/20/2024 08:56:07 - INFO - __main__ -   layer 11, precision: 0.8220, recall: 0.8188, f1: 0.8204, sparsity: 0.1272, cur_sparsity: 0.1267
```