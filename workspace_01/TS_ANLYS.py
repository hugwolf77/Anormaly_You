from datetime import date, datetime
import torch
import pandas as pd
import numpy as np

from SFW_exp.exp_main import Exp_Main
from dataform import args, TS_DATA
from Toolkit import LOAD_TS_FILE #, REPORT

from scipy.stats import pearsonr
from sklearn.metrics import *

class TS_ANLYS:
    def __init__(self,file_path, file_format, meta_file_format, file_ix, target_ix, tg) -> None:
        self.LOAD_TS_FILE = LOAD_TS_FILE(file_path, file_format, meta_file_format, file_ix, target_ix)
        self.file_ix = file_ix
        self.tg = tg
        self.Row_count, self.ob_measure_id, self.ob_division, self.ob_reservoir, self.ob_name, self.ob_target, self.ob_input,self.MAE, self.MSE, self.RMSE, self.MAPE, self.MSPE, self.RSE, self.CORR = self.ANLYS()
        # self.ColumnNames, self.Row_NA_count, self.columns_NA, 

    def __repr__(self) -> str:
        return "TS_ANLYS is called"

    def ANLYS(self):
        tsFile = self.LOAD_TS_FILE.File_loader()
        _, _, _, _, MetaInfo = self.LOAD_TS_FILE.Meta_file_loader()
        args = self.Set_args()
        Exp = Exp_Main
        
        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
        if args.use_gpu and args.use_multi_gpu:
            args.dvices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]    

        Row_count = tsFile.shape[0]
        ob_measure_id = str(MetaInfo['measure_id'])
        ob_division = str(MetaInfo['division'])
        ob_reservoir = str(MetaInfo['reservoir'])
        ob_name = str(MetaInfo['name'])
        ob_target = str(self.tg)
        ob_input = str(MetaInfo['input_serial'])

        if args.is_training:
          for ii in range(args.itr):
              # setting record of experiments
              setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_eb{}_{}_{}'.format(
                                                                    args.model_id, args.model, args.data, args.features,
                                                                    args.seq_len, args.label_len, args.pred_len, args.embed, 
                                                                    args.des, ii)
              exp = Exp(args)  # set experiments
              print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
              exp.train(setting)

              print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
              preds, trues, inputx, mae,mse,rmse,mape,mspe,rse,corr = exp.test(setting)
              MAE, MSE, RMSE, norm_MAPE, MAPE = self.Matric(preds, trues)

              if args.do_predict:
                  print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                  exp.predict(setting, True)
              torch.cuda.empty_cache()
        else:
          ii = 0
          setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_eb{}_{}_{}'.format(
                                                                    args.model_id, args.model, args.data, args.features,
                                                                    args.seq_len, args.label_len, args.pred_len, args.embed,
                                                                    args.des, ii)
          exp = Exp(args)  # set experiments
          print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
          preds, trues, inputx, mae,mse,rmse,mape,mspe,rse,corr = exp.test(setting, test=1)
          torch.cuda.empty_cache()
          MAE, MSE, RMSE, norm_MAPE, MAPE = self.Matric(preds, trues)

        return Row_count, ob_measure_id, ob_division, ob_reservoir, ob_name, ob_target, ob_input, mae,mse,rmse,MAPE,mspe,rse,corr

    def Set_args(self):
        tsFile = self.LOAD_TS_FILE.File_loader()
        _, _, _, _, MetaInfo = self.LOAD_TS_FILE.Meta_file_loader()
        model_args = args(
            is_training = True, 
            model_id = '(SFW)_'+ MetaInfo['division'] +'_' + MetaInfo['reservoir'] + '_' + MetaInfo['name'] + '_' + self.tg,
            model = 'DLinear',
            # data loader
            data = 'custom_hour',
            root_path = self.LOAD_TS_FILE.file_path,  
            data_path = self.LOAD_TS_FILE.fileName,
            meta_path = self.LOAD_TS_FILE.Metafile_name,
            features = 'MS',                             
            target =  str(self.tg),                             
            freq = 'h',                                  
            checkpoints = './(SFW_DLinear)_checkpoints/',
            # forecasting task
            seq_len = 72, 
            label_len = 1,
            pred_len = 24, 
            # DLinear
            moving_avg = 10,  
            conv_kernal = 1,
            conv1d = True,
            RIN = True,
            combination = True,
            embed_type = 0,   
            enc_in =  MetaInfo['input_count'],       
            c_out = 1,        
            embed = 'timeF',  
            do_predict = False,      
            # optimization
            num_workers =5,         
            itr = 1,                 
            train_epochs = 50,      
            batch_size = 20,          
            patience = 5,            
            learning_rate = 0.1,
            des = 'Exp',             
            loss = 'mse',            
            lradj = 'type1',         
            use_amp = False ,        
            # GPU
            use_gpu = True,          
            gpu = 0,                 
            use_multi_gpu = False,   
            devices = '0,1,2,3',     
            test_flop = False,
        )
        return model_args 

    def Set_TS_Data(self):
        TSDATA = TS_DATA(       Id = self.file_ix+1,
                                ob_measure_id = self.ob_measure_id,
                                ob_division = self.ob_division,
                                ob_reservoir = self.ob_reservoir,
                                ob_name = self.ob_name,       
                                ob_target = self.ob_target,
                                ob_input = self.ob_input,         
                                Modi_Date = datetime.today(),
                                FileName = self.LOAD_TS_FILE.fileName,
                                FilePath = self.LOAD_TS_FILE.file_path,
                                Row_count = self.Row_count,
                                MAE = self.MAE,
                                MSE =  self.MSE,
                                RMSE = self.RMSE,
                                # norm_MAPE = self.norm_MAPE,
                                MAPE = self.MAPE,
                                MSPE = self.MSPE,
                                RSE = self.RSE,
                                CORR =  self.CORR,
                                # ColumnNames = self.ColumnNames,
                                # Row_NA_count = self.Row_NA_count,
                                plot = True
                                )
        return TSDATA

    def Matric(self,pred, true_y):
        args = self.Set_args()
        t_df_rs = self.LOAD_TS_FILE.File_loader().copy()
        # print(f"t_df_rs.columns : {t_df_rs.columns}")

        t_df_rs['date'] = pd.to_datetime(t_df_rs['measure_date'])
        t_df_rs = t_df_rs.set_index('date')
        t_df_rs = t_df_rs.resample(rule = 'D', kind='timestamp', origin='start').first()

        num_train = int(len(t_df_rs) * 0.8)
        num_test = int(len(t_df_rs) * 0.1)
        num_vali = len(t_df_rs) - num_train - num_test
        border1 = [0, num_train - args.seq_len, len(t_df_rs) - num_test - args.seq_len]
        border2 = [num_train, num_train + num_vali, len(t_df_rs)]

        train_data = t_df_rs[border1[0]:border2[0]]
        true_step = t_df_rs[border1[2]:border2[2]]
        train_step = train_data
        self.train_mean = train_step.mean()
        self.train_std = train_step.std() + 0.00000001
        
        #prepare
        result = pred[:].squeeze()
        label = true_y[:].squeeze()
        # print(result.shape, label.shape)

        result_scale = (result*self.train_std[1]) + self.train_mean[1]
        label_scale = (label*self.train_std[1]) + self.train_mean[1]

        bk = []
        diff = 0

        #MAE
        # MAE = mean_absolute_error(label, result)
        sumout = 0
        for id in range(result.shape[0]-diff):
          if id in bk:
            pass
          else:
            out = mean_absolute_error(label[id,:],result[id,:])
            sumout += out
        MAE = sumout/(result.shape[0]-diff)
        # print(f"MAE : {MAE}")

        #MSE
        sumout = 0
        for id in range(result.shape[0]-diff):
          if id in bk:
            pass
          else:
            out = mean_squared_error(label[id,:],result[id,:])
            sumout += out
        MSE =  sumout/(result.shape[0]-diff)
        # print(f"MSE : {MSE}")

        # RMSE (Root Mean Squared Error)
        sumout = 0
        for id in range(result.shape[0]-diff):
          if id in bk:
            pass
          else:
            out = mean_squared_error(label[id,:],result[id,:])
            sumout += np.sqrt(out)
        RMSE =  sumout/(result.shape[0]-diff)
        # print(f"RMSE : {RMSE}")

        #MAPE (Mean Absolute Percentage Error)
        sumout = 0
        for id in range(result.shape[0]-diff):
          if id in bk:
            pass
          else:  
            out = np.mean(np.abs((label[id,:] - result[id,:]) / label[id,:])) * 100 
            sumout += out
        norm_MAPE = sumout/(result.shape[0]-diff)
        # print(f"MAPE : {MAPE}")

        #MAPE_scale (Mean Absolute Percentage Error)
        sumout = 0
        for id in range(result_scale.shape[0]-diff):
          if id in bk:
            pass
          else:  
            out = np.mean(np.abs((label_scale[id,:] - result_scale[id,:]) / label_scale[id,:])) * 100 
            sumout += out
        MAPE = sumout/(result_scale.shape[0]-diff)
        # print(f"denorm_MAPE : {MAPE}")
        # print(f"error-count : {diff}")
        return MAE, MSE, RMSE, norm_MAPE, MAPE
