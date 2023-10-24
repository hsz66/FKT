# FKT
The code is related to the paper [Response Speed Enhanced Fine-grained Knowledge Tracing: A Multi-task Learning Perspective](https://www.sciencedirect.com/science/article/abs/pii/S095741742302609X).
# Datasets
We cut thousands of lines from the complete Junyi data set to create the Junyi_for_testing dataset for code testing.
The complete dataset can be downloaded from the link below.
[Ednet](http://ednet-leaderboard.s3-website-ap-northeast-1.amazonaws.com/)
[Junyi](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198/)
[ASSIST2012](https://sites.google.com/site/assistmentsdata/2012-13-school-data-withaffect)
[ASSISTChall](https://sites.google.com/view/assistmentsdatamining/)
# Examples to run the model
'''
python -u train.py --dataset junyi_for_testing --gpu 0
'''
# Citation
If our code is helpful to your work, please cite:
'''
@article{HUANG2024122107,
title = {Response speed enhanced fine-grained knowledge tracing: A multi-task learning perspective},
journal = {Expert Systems with Applications},
volume = {238},
pages = {122107},
year = {2024},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.122107},
author = {Tao Huang and Shengze Hu and Huali Yang and Jing Geng and Zhifei Li and Zhuoran Xu and Xinjia Ou},
keywords = {Knowledge tracing, Learning trajectory, Multi-task learning, Transformer, Response speed},
}
'''
