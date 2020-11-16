## This is not a md file 

## Author of the Project: Cunwei Fan
## Date                 : Nov 15, 2020

Files:

    Result Report:
        RESULT_REPORT.pdf

    Ipython-Notebook:
        1_Data_Vsiualization.ipynb  : initial notebook for data visualization and feature engineering
        2_Classification.ipynb      : second notebook for zero/non_zero classification for Order Arrivals
        3_Regression.ipynb.         : third notebook for order number regression 
        
Folder:

    DeepNets:
       Architectures.py      : DataSet class and CNN as well as experimental LSTM models
       train.py.             : The code to do the training and evaluation for models
       output                : folders containing training history and final weights of the models
       
       
    figs: Figures used in the report. 
    
    latex: The files to generate the final report.
    
 
NOTE:

    1. The training of the deep net is on a computer with GPUs and if there is any attempt to reproduce the result, please use a computer with GPUs or specify the device argument in the train.py as "torch.device("cpu")". 

    2. However, the deep net evaluation is done on a computer without cpu and thus there should be no trouble of doing that. 
    
    3. THE TIME IS UP and thus the regression problem is not done compeletly and we expect to improve the result with more sophisticated models. 



RESULT on TEST DATA:

    CLASSIFICATION:
    
                   &  Bid        & Ask     
        f1 score   & 0.6990      & 0.6843   
        accuracy   & $81.78%     & 81.47% 

    REGRESSION:
    
                   &  Bid        & Ask test      
        R2 score   & 32.12%      & 26.69%
        RMSE       & 0.955       & 1.078

    The result of regression is not very good since the model only explains around 30% of the variation. We expect to do better if we have more time to implement a more complicated model. 