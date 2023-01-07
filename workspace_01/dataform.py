
from datetime import date, datetime
from dataclasses import dataclass, field

@dataclass
class args:
    is_training : bool   # or int  
    model_id : str       
    model : str          

    # data loader
    data : str           
    root_path : str      
    data_path : str
    meta_path : str      
    features : str        
    target : str          
    freq : str            
    checkpoints : str 

    # forecasting task
    seq_len : int              # 56
    label_len : int            # 1 
    pred_len : int             # 5 
    # DLinear
    moving_avg : int           # 10
    # moving_avg = [3,5,10,15,20,25,30] #,35,40,45]
    conv_kernal : int          # 1
    conv1d : bool              # True
    RIN : bool                 # True
    combination : bool         # True
    embed_type : int           # 0
    enc_in : int               # 3
    c_out : int                # 1 
    embed : str                # 'timeF'
    do_predict : bool          # False

    # optimization
    num_workers : int          # 10
    itr : int                  # 1 
    train_epochs : int         # 100
    batch_size : int           # 5  
    patience : int             # 7  
    learning_rate : float      # 0.01
    des : str                  # 'Exp'
    loss : str                 # 'mse'
    lradj : str                # 'type1'
    use_amp : bool             # False  

    # GPU
    use_gpu : bool             # True   
    gpu : int                  # 0      
    use_multi_gpu : bool       # False  
    devices : str              # '0,1,2,3'
    test_flop : bool           # False    

@dataclass(frozen=True)
class TS_DATA:
    # base info
    Id : int
    ob_measure_id : str
    ob_division : str
    ob_reservoir :str
    ob_name : str             # ob name
    ob_target : str           # target value
    ob_input : str
    Modi_Date : date
    FileName : str
    FilePath : str
    Row_count : int
    MAE : float
    MSE : float
    RMSE : float
    # normed_MAPE : float
    MAPE : float
    MSPE : float
    RSE : float
    CORR : float
    plot : bool = False
