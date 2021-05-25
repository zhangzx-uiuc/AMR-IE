# AMR-IE
The code repository for AMR guided joint information extraction model (NAACL-2021). Our code is built on the top of [OneIE](https://blender.cs.illinois.edu/software/oneie/).

## AMR Parser
We adopt the transformer-based AMR parser [(Astudillo et al.)](https://www.aclweb.org/anthology/2020.findings-emnlp.89/). The code of the AMR parser is derived from the original [Github Repo](https://github.com/IBM/transition-amr-parser). We make some slight modifications and shared the code at [this link](https://drive.google.com/file/d/1SB36NyEaRd740rGTjD_8ga7l5NGeRlkR/). We train this AMR parser on LDC AMR 3.0 data, and the pretrained model is available at [this link](https://drive.google.com/file/d/1LRJuOwHQ6EWmzRBpYpWwsr_5m2kO-IP7). Our great thanks to Astudillo et al. for publicizing their AMR parser.

## Datasets
The preprocessed GENIA 2011 and 2013 datasets for joint information extraction are available at [this link](https://drive.google.com/file/d/1tnGyyJo7Enesqv8R1Mpng7c1U5lEzLqm/view?usp=sharing). Please unzip the file and put all the folders into `./data/`. The ACE2005 and ERE datasets are available at the [LDC website](https://catalog.ldc.upenn.edu/LDC2006T06). Please use the following script to preprocessing these datasets.

## Train the model
`python train.py -c config/genia_2011.json -g 0 -n NAME`

## Acknowledgement
If you have any problems on this code, feel free to contact zhangzx369@gmail.com.
If you use this code as part of your research, please cite the following paper:
```
@inproceedings{amrie2021,
  author    = {Zixuan Zhang and Heng Ji},
  title     = {Abstract Meaning Representation Guided Graph Encoding and Decodingfor Joint Information Extraction},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)},
  year      = {2021},
  pages     = {39--49},
  url       = {https://www.aclweb.org/anthology/2021.naacl-main.4/}
  }
```
