# High-resolution extracellular pH imaging of liver cancer with multiparametric MR using Deep Image Prior (NMR in Biomedicine)

Siyuan Dong, Annabella Shewarega, Julius Chapiro, Zhuotong Cai, Fahmeed Hyder, Daniel Coman, James S. Duncan

[[Paper Link](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/nbm.5145)]

### Citation
If you use this code please cite:

    @article{dong2024high,
      title={High-resolution extracellular pH imaging of liver cancer with multiparametric MR using Deep Image Prior},
      author={Dong, Siyuan and Shewarega, Annabella and Chapiro, Julius and Cai, Zhuotong and Hyder, Fahmeed and Coman, Daniel and Duncan, James S},
      journal={NMR in Biomedicine},
      pages={e5145},
      year={2024},
      publisher={Wiley Online Library}
    }
   
### Environment and Dependencies
 Requirements:
 * python 3.7.11
 * pytorch 1.1.0
 * torchvision 0.3.0
 * numpy 1.19.2
 * h5py 3.8.0

### Directory
    main.py                             # main file for any rabbit
    main_Rabbit1.py                     # main file for evaluating Rabbit1 with HighRes ground truth
    main_Rabbit2.py                     # main file for evaluating Rabbit2 with HighRes ground truth
    loader
    └──  dataloader.py                  # dataloader
    utils
    └──logs.py                          # logging
    models
    └──UNet_T1_T2_DWI.py                # UNet model with T1, T2 and DWI as inputs

