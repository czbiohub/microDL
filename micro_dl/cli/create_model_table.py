import os
import glob
import pandas as pd

def GetSubDirName(ImgPath):
    assert os.path.exists(ImgPath), 'Input folder does not exist!'
    subDirPath = glob.glob(os.path.join(ImgPath, '*/'))
    subDirName = [os.path.split(subdir[:-1])[1] for subdir in subDirPath]
#    assert subDirName, 'No sub directories found'
    return subDirName

model_path = r'D:\Box Sync\Processed\2018_07_03_KidneyTissueSection\SMS_2018_0703_1835_1_BG_2018_0703_1829_1'
model_summary_fname = 'model_summary.csv'
model_dir_list = GetSubDirName(model_path)
df_model_metric = pd.DataFrame()
for model_dir in model_dir_list:
    df_history = pd.read_csv(os.path.join(model_path, model_dir, 'history.csv'))
    df_best_metric = df_history[['coeff_determination', 'val_coeff_determination']].agg(['max'])
    df_best_metric.rename({'max': model_dir}, axis='index', inplace=True)
    df_best_loss = df_history[['loss', 'val_loss']].agg(['min'])
    df_best_loss.rename({'min': model_dir}, axis='index', inplace=True)
    df_best = pd.concat([df_best_metric, df_best_loss], axis=1)
    df_model_metric = df_model_metric.append(df_best)
df_model_metric.to_csv(os.path.join(model_path, model_summary_fname), sep=',')


