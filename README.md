
![python](https://img.shields.io/badge/Python-3.8-blue)
![tf](https://img.shields.io/badge/Tensorflow--GPU-2.5-orange)  

## Table of contents
* [Introduction](#introduction)  
* [Background Knowledge](#background-knowledge)  
* [Environment](#environment)  
* [File Guide](#file-guide)  
* [Reference](#reference)  

## Introduction  
This project used a 1D CNN model to compare two different kinds of noise generated [light curves](https://exoplanets.nasa.gov/resources/280/light-curve-of-a-planet-transiting-its-star/).  
The methods of generated light curves are following:  
1. original noise data generated from dataset light curves in [Kepler](https://exoplanetarchive.ipac.caltech.edu/bulk_data_download/) mission  
2. noise data generated from quasi-period system by [Pearson et al. (2019)](https://arxiv.org/abs/1706.04319)
  
After  constructing several training datasets by using those two methods with theoretic light curve formula in [Mandel & Agol (2002)](https://arxiv.org/abs/astro-ph/0210099), we trained our CNN model with K-fold Cross-Validation. Then we selected the best method to search possible transit light curves for Kepler  dataset.

## Background Knowledge  
A planet that orbits a star outside the solar system is called an exoplanet.  
There are many of methods to search exoplanets. The common methods are including: Doppler effect method, [transit method](https://exoplanets.nasa.gov/faq/31/whats-a-transit/), astrometry method and so on.  
In this project, we used transit method and 1D CNN model to search whether there exists exoplanet candidates in Kepler Q1 dataset.  
* Transit method:  
  Because stars can emit stable light, planets cannot. Due to this property, when a plenet passes between a star and its observer, the bright emitted by this star drops. This is called a transit event.  
[Transit event video source from NASA](https://exoplanets.nasa.gov/alien-worlds/ways-to-find-a-planet/)  

## Environment  
This project is created with:  
* Python version: 3.8  
* Tensorflow-gpu version: 2.5.0  
* Numpy version: 1.20.3  
* Matplotlib version: 3.4.2  
* astropy version: 4.3.1  

To install quickly:
```
pip install -r requirements.txt
```
  
Notice:  
If you want to use my code, please ensure package version especially tensorflow version. Sometimes when you used different version, it could be incompatible or some module have be deleted.

## File Guide  
* To preprocess kepler datas.  
`Kplr/compare_KM.py` - to record and compare magnitude of datas we had chosen from Kepler mission.  
`Kplr/dev_analysis.py` - This file is to analyze sigma of our kepler datas.We divided them into 4 group and hoped to have at least 50 in each group.  
`Kplr/look_star.py` - pre-processing datas from selected kepler mission.  
`Kplr/read_fits.py` - To convert .fits to .txt  
* To generate our training datasets.  
`generate_data_k.py` - To generate training dataset using selected Kepler datas.  
`generate_data_p.py` - To generate training dataset using quasi-period system.  
* CNN  
`model_structure.py` - Our 1D CNN model and training method(K-fold, early stop training)  
`train.py` - To train our model for each sample size of dataset and record training time.  
* important and useful tools.  
`mangol.py` - the model to simulate transit event.  
`tool.py` - some common tools we often used.  

## Reference  
* Thesis  
[1] 郭芷綺(2020)。以機器學習法搜尋系外行星的研究。國立清華大學碩士論文。  
[2] [Mandel, K., & Agol, E. (2002). Analytic light curves for planetary transit searches. The Astrophysical Journal, 580:L171–L175, 2002 December 1](https://exoplanetarchive.ipac.caltech.edu/bulk_data_download/)  
[3] [Pearson, K.A., Palafox, L., & Griffith, C.A. (2018). Searching for exoplanets using artificial intelligence. Monthly Notices of the Royal Astronomical Society, 474(1), 478-491.](https://arxiv.org/abs/1706.04319)  
[4] [Yeh, Li-Chin & Jiang, Ing-Guey (2021). Searching for Possible Exoplanet Transits from BRITE Data through a Machine Learning Technique. Publications of the Astronomical Society of the Pacific, 133:014401 (12pp), 2021 January.](https://arxiv.org/abs/2012.10035)  
* Repo  
[Pearson KA, Palafox L., Griffith CA, 2018, MNRAS, 474, 478](https://github.com/pearsonkyle/Exoplanet-Artificial-Intelligence)
