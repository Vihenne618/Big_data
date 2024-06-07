from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StringType,StructField,StructType,DoubleType,IntegerType
from scipy.io.arff import loadarff
from pyspark.sql.functions import col
import pandas as pd
from itertools import islice
import io
import gc
import time


# Parameter column names for data
attributes = ['separation real',
 'propensity real',
 'length real',
 'PredSS_r1_-4 {H,E,C,X}',
 'PredSS_r1_-3 {H,E,C,X}',
 'PredSS_r1_-2 {H,E,C,X}',
 'PredSS_r1_-1 {H,E,C,X}',
 'PredSS_r1 {H,E,C}',
 'PredSS_r1_1 {H,E,C,X}',
 'PredSS_r1_2 {H,E,C,X}',
 'PredSS_r1_3 {H,E,C,X}',
 'PredSS_r1_4 {H,E,C,X}',
 'PredSS_r2_-4 {H,E,C,X}',
 'PredSS_r2_-3 {H,E,C,X}',
 'PredSS_r2_-2 {H,E,C,X}',
 'PredSS_r2_-1 {H,E,C,X}',
 'PredSS_r2 {H,E,C}',
 'PredSS_r2_1 {H,E,C,X}',
 'PredSS_r2_2 {H,E,C,X}',
 'PredSS_r2_3 {H,E,C,X}',
 'PredSS_r2_4 {H,E,C,X}',
 'PredSS_freq_central_H real',
 'PredSS_freq_central_E real',
 'PredSS_freq_central_C real',
 'PredCN_freq_central_0 real',
 'PredCN_freq_central_1 real',
 'PredCN_freq_central_2 real',
 'PredCN_freq_central_3 real',
 'PredCN_freq_central_4 real',
 'PredRCH_freq_central_0 real',
 'PredRCH_freq_central_1 real',
 'PredRCH_freq_central_2 real',
 'PredRCH_freq_central_3 real',
 'PredRCH_freq_central_4 real',
 'PredSA_freq_central_0 real',
 'PredSA_freq_central_1 real',
 'PredSA_freq_central_2 real',
 'PredSA_freq_central_3 real',
 'PredSA_freq_central_4 real',
 'PredRCH_r1_-4 {0,1,2,3,4,X}',
 'PredRCH_r1_-3 {0,1,2,3,4,X}',
 'PredRCH_r1_-2 {0,1,2,3,4,X}',
 'PredRCH_r1_-1 {0,1,2,3,4,X}',
 'PredRCH_r1 {0,1,2,3,4}',
 'PredRCH_r1_1 {0,1,2,3,4,X}',
 'PredRCH_r1_2 {0,1,2,3,4,X}',
 'PredRCH_r1_3 {0,1,2,3,4,X}',
 'PredRCH_r1_4 {0,1,2,3,4,X}',
 'PredRCH_r2_-4 {0,1,2,3,4,X}',
 'PredRCH_r2_-3 {0,1,2,3,4,X}',
 'PredRCH_r2_-2 {0,1,2,3,4,X}',
 'PredRCH_r2_-1 {0,1,2,3,4,X}',
 'PredRCH_r2 {0,1,2,3,4}',
 'PredRCH_r2_1 {0,1,2,3,4,X}',
 'PredRCH_r2_2 {0,1,2,3,4,X}',
 'PredRCH_r2_3 {0,1,2,3,4,X}',
 'PredRCH_r2_4 {0,1,2,3,4,X}',
 'PredCN_r1_-4 {0,1,2,3,4,X}',
 'PredCN_r1_-3 {0,1,2,3,4,X}',
 'PredCN_r1_-2 {0,1,2,3,4,X}',
 'PredCN_r1_-1 {0,1,2,3,4,X}',
 'PredCN_r1 {0,1,2,3,4}',
 'PredCN_r1_1 {0,1,2,3,4,X}',
 'PredCN_r1_2 {0,1,2,3,4,X}',
 'PredCN_r1_3 {0,1,2,3,4,X}',
 'PredCN_r1_4 {0,1,2,3,4,X}',
 'PredCN_r2_-4 {0,1,2,3,4,X}',
 'PredCN_r2_-3 {0,1,2,3,4,X}',
 'PredCN_r2_-2 {0,1,2,3,4,X}',
 'PredCN_r2_-1 {0,1,2,3,4,X}',
 'PredCN_r2 {0,1,2,3,4}',
 'PredCN_r2_1 {0,1,2,3,4,X}',
 'PredCN_r2_2 {0,1,2,3,4,X}',
 'PredCN_r2_3 {0,1,2,3,4,X}',
 'PredCN_r2_4 {0,1,2,3,4,X}',
 'PredSA_r1_-4 {0,1,2,3,4,X}',
 'PredSA_r1_-3 {0,1,2,3,4,X}',
 'PredSA_r1_-2 {0,1,2,3,4,X}',
 'PredSA_r1_-1 {0,1,2,3,4,X}',
 'PredSA_r1 {0,1,2,3,4}',
 'PredSA_r1_1 {0,1,2,3,4,X}',
 'PredSA_r1_2 {0,1,2,3,4,X}',
 'PredSA_r1_3 {0,1,2,3,4,X}',
 'PredSA_r1_4 {0,1,2,3,4,X}',
 'PredSA_r2_-4 {0,1,2,3,4,X}',
 'PredSA_r2_-3 {0,1,2,3,4,X}',
 'PredSA_r2_-2 {0,1,2,3,4,X}',
 'PredSA_r2_-1 {0,1,2,3,4,X}',
 'PredSA_r2 {0,1,2,3,4}',
 'PredSA_r2_1 {0,1,2,3,4,X}',
 'PredSA_r2_2 {0,1,2,3,4,X}',
 'PredSA_r2_3 {0,1,2,3,4,X}',
 'PredSA_r2_4 {0,1,2,3,4,X}',
 'PredSS_freq_global_H real',
 'PredSS_freq_global_E real',
 'PredSS_freq_global_C real',
 'PredCN_freq_global_0 real',
 'PredCN_freq_global_1 real',
 'PredCN_freq_global_2 real',
 'PredCN_freq_global_3 real',
 'PredCN_freq_global_4 real',
 'PredRCH_freq_global_0 real',
 'PredRCH_freq_global_1 real',
 'PredRCH_freq_global_2 real',
 'PredRCH_freq_global_3 real',
 'PredRCH_freq_global_4 real',
 'PredSA_freq_global_0 real',
 'PredSA_freq_global_1 real',
 'PredSA_freq_global_2 real',
 'PredSA_freq_global_3 real',
 'PredSA_freq_global_4 real',
 'AA_freq_central_A real',
 'AA_freq_central_R real',
 'AA_freq_central_N real',
 'AA_freq_central_D real',
 'AA_freq_central_C real',
 'AA_freq_central_Q real',
 'AA_freq_central_E real',
 'AA_freq_central_G real',
 'AA_freq_central_H real',
 'AA_freq_central_I real',
 'AA_freq_central_L real',
 'AA_freq_central_K real',
 'AA_freq_central_M real',
 'AA_freq_central_F real',
 'AA_freq_central_P real',
 'AA_freq_central_S real',
 'AA_freq_central_T real',
 'AA_freq_central_W real',
 'AA_freq_central_Y real',
 'AA_freq_central_V real',
 'PredSS_central_-2 {H,E,C}',
 'PredSS_central_-1 {H,E,C}',
 'PredSS_central {H,E,C}',
 'PredSS_central_1 {H,E,C}',
 'PredSS_central_2 {H,E,C}',
 'PredCN_central_-2 {0,1,2,3,4}',
 'PredCN_central_-1 {0,1,2,3,4}',
 'PredCN_central {0,1,2,3,4}',
 'PredCN_central_1 {0,1,2,3,4}',
 'PredCN_central_2 {0,1,2,3,4}',
 'PredRCH_central_-2 {0,1,2,3,4}',
 'PredRCH_central_-1 {0,1,2,3,4}',
 'PredRCH_central {0,1,2,3,4}',
 'PredRCH_central_1 {0,1,2,3,4}',
 'PredRCH_central_2 {0,1,2,3,4}',
 'PredSA_central_-2 {0,1,2,3,4}',
 'PredSA_central_-1 {0,1,2,3,4}',
 'PredSA_central {0,1,2,3,4}',
 'PredSA_central_1 {0,1,2,3,4}',
 'PredSA_central_2 {0,1,2,3,4}',
 'AA_freq_global_A real',
 'AA_freq_global_R real',
 'AA_freq_global_N real',
 'AA_freq_global_D real',
 'AA_freq_global_C real',
 'AA_freq_global_Q real',
 'AA_freq_global_E real',
 'AA_freq_global_G real',
 'AA_freq_global_H real',
 'AA_freq_global_I real',
 'AA_freq_global_L real',
 'AA_freq_global_K real',
 'AA_freq_global_M real',
 'AA_freq_global_F real',
 'AA_freq_global_P real',
 'AA_freq_global_S real',
 'AA_freq_global_T real',
 'AA_freq_global_W real',
 'AA_freq_global_Y real',
 'AA_freq_global_V real',
 'PSSM_r1_-4_A real',
 'PSSM_r1_-4_R real',
 'PSSM_r1_-4_N real',
 'PSSM_r1_-4_D real',
 'PSSM_r1_-4_C real',
 'PSSM_r1_-4_Q real',
 'PSSM_r1_-4_E real',
 'PSSM_r1_-4_G real',
 'PSSM_r1_-4_H real',
 'PSSM_r1_-4_I real',
 'PSSM_r1_-4_L real',
 'PSSM_r1_-4_K real',
 'PSSM_r1_-4_M real',
 'PSSM_r1_-4_F real',
 'PSSM_r1_-4_P real',
 'PSSM_r1_-4_S real',
 'PSSM_r1_-4_T real',
 'PSSM_r1_-4_W real',
 'PSSM_r1_-4_Y real',
 'PSSM_r1_-4_V real',
 'PSSM_r1_-3_A real',
 'PSSM_r1_-3_R real',
 'PSSM_r1_-3_N real',
 'PSSM_r1_-3_D real',
 'PSSM_r1_-3_C real',
 'PSSM_r1_-3_Q real',
 'PSSM_r1_-3_E real',
 'PSSM_r1_-3_G real',
 'PSSM_r1_-3_H real',
 'PSSM_r1_-3_I real',
 'PSSM_r1_-3_L real',
 'PSSM_r1_-3_K real',
 'PSSM_r1_-3_M real',
 'PSSM_r1_-3_F real',
 'PSSM_r1_-3_P real',
 'PSSM_r1_-3_S real',
 'PSSM_r1_-3_T real',
 'PSSM_r1_-3_W real',
 'PSSM_r1_-3_Y real',
 'PSSM_r1_-3_V real',
 'PSSM_r1_-2_A real',
 'PSSM_r1_-2_R real',
 'PSSM_r1_-2_N real',
 'PSSM_r1_-2_D real',
 'PSSM_r1_-2_C real',
 'PSSM_r1_-2_Q real',
 'PSSM_r1_-2_E real',
 'PSSM_r1_-2_G real',
 'PSSM_r1_-2_H real',
 'PSSM_r1_-2_I real',
 'PSSM_r1_-2_L real',
 'PSSM_r1_-2_K real',
 'PSSM_r1_-2_M real',
 'PSSM_r1_-2_F real',
 'PSSM_r1_-2_P real',
 'PSSM_r1_-2_S real',
 'PSSM_r1_-2_T real',
 'PSSM_r1_-2_W real',
 'PSSM_r1_-2_Y real',
 'PSSM_r1_-2_V real',
 'PSSM_r1_-1_A real',
 'PSSM_r1_-1_R real',
 'PSSM_r1_-1_N real',
 'PSSM_r1_-1_D real',
 'PSSM_r1_-1_C real',
 'PSSM_r1_-1_Q real',
 'PSSM_r1_-1_E real',
 'PSSM_r1_-1_G real',
 'PSSM_r1_-1_H real',
 'PSSM_r1_-1_I real',
 'PSSM_r1_-1_L real',
 'PSSM_r1_-1_K real',
 'PSSM_r1_-1_M real',
 'PSSM_r1_-1_F real',
 'PSSM_r1_-1_P real',
 'PSSM_r1_-1_S real',
 'PSSM_r1_-1_T real',
 'PSSM_r1_-1_W real',
 'PSSM_r1_-1_Y real',
 'PSSM_r1_-1_V real',
 'PSSM_r1_0_A real',
 'PSSM_r1_0_R real',
 'PSSM_r1_0_N real',
 'PSSM_r1_0_D real',
 'PSSM_r1_0_C real',
 'PSSM_r1_0_Q real',
 'PSSM_r1_0_E real',
 'PSSM_r1_0_G real',
 'PSSM_r1_0_H real',
 'PSSM_r1_0_I real',
 'PSSM_r1_0_L real',
 'PSSM_r1_0_K real',
 'PSSM_r1_0_M real',
 'PSSM_r1_0_F real',
 'PSSM_r1_0_P real',
 'PSSM_r1_0_S real',
 'PSSM_r1_0_T real',
 'PSSM_r1_0_W real',
 'PSSM_r1_0_Y real',
 'PSSM_r1_0_V real',
 'PSSM_r1_1_A real',
 'PSSM_r1_1_R real',
 'PSSM_r1_1_N real',
 'PSSM_r1_1_D real',
 'PSSM_r1_1_C real',
 'PSSM_r1_1_Q real',
 'PSSM_r1_1_E real',
 'PSSM_r1_1_G real',
 'PSSM_r1_1_H real',
 'PSSM_r1_1_I real',
 'PSSM_r1_1_L real',
 'PSSM_r1_1_K real',
 'PSSM_r1_1_M real',
 'PSSM_r1_1_F real',
 'PSSM_r1_1_P real',
 'PSSM_r1_1_S real',
 'PSSM_r1_1_T real',
 'PSSM_r1_1_W real',
 'PSSM_r1_1_Y real',
 'PSSM_r1_1_V real',
 'PSSM_r1_2_A real',
 'PSSM_r1_2_R real',
 'PSSM_r1_2_N real',
 'PSSM_r1_2_D real',
 'PSSM_r1_2_C real',
 'PSSM_r1_2_Q real',
 'PSSM_r1_2_E real',
 'PSSM_r1_2_G real',
 'PSSM_r1_2_H real',
 'PSSM_r1_2_I real',
 'PSSM_r1_2_L real',
 'PSSM_r1_2_K real',
 'PSSM_r1_2_M real',
 'PSSM_r1_2_F real',
 'PSSM_r1_2_P real',
 'PSSM_r1_2_S real',
 'PSSM_r1_2_T real',
 'PSSM_r1_2_W real',
 'PSSM_r1_2_Y real',
 'PSSM_r1_2_V real',
 'PSSM_r1_3_A real',
 'PSSM_r1_3_R real',
 'PSSM_r1_3_N real',
 'PSSM_r1_3_D real',
 'PSSM_r1_3_C real',
 'PSSM_r1_3_Q real',
 'PSSM_r1_3_E real',
 'PSSM_r1_3_G real',
 'PSSM_r1_3_H real',
 'PSSM_r1_3_I real',
 'PSSM_r1_3_L real',
 'PSSM_r1_3_K real',
 'PSSM_r1_3_M real',
 'PSSM_r1_3_F real',
 'PSSM_r1_3_P real',
 'PSSM_r1_3_S real',
 'PSSM_r1_3_T real',
 'PSSM_r1_3_W real',
 'PSSM_r1_3_Y real',
 'PSSM_r1_3_V real',
 'PSSM_r1_4_A real',
 'PSSM_r1_4_R real',
 'PSSM_r1_4_N real',
 'PSSM_r1_4_D real',
 'PSSM_r1_4_C real',
 'PSSM_r1_4_Q real',
 'PSSM_r1_4_E real',
 'PSSM_r1_4_G real',
 'PSSM_r1_4_H real',
 'PSSM_r1_4_I real',
 'PSSM_r1_4_L real',
 'PSSM_r1_4_K real',
 'PSSM_r1_4_M real',
 'PSSM_r1_4_F real',
 'PSSM_r1_4_P real',
 'PSSM_r1_4_S real',
 'PSSM_r1_4_T real',
 'PSSM_r1_4_W real',
 'PSSM_r1_4_Y real',
 'PSSM_r1_4_V real',
 'PSSM_r2_-4_A real',
 'PSSM_r2_-4_R real',
 'PSSM_r2_-4_N real',
 'PSSM_r2_-4_D real',
 'PSSM_r2_-4_C real',
 'PSSM_r2_-4_Q real',
 'PSSM_r2_-4_E real',
 'PSSM_r2_-4_G real',
 'PSSM_r2_-4_H real',
 'PSSM_r2_-4_I real',
 'PSSM_r2_-4_L real',
 'PSSM_r2_-4_K real',
 'PSSM_r2_-4_M real',
 'PSSM_r2_-4_F real',
 'PSSM_r2_-4_P real',
 'PSSM_r2_-4_S real',
 'PSSM_r2_-4_T real',
 'PSSM_r2_-4_W real',
 'PSSM_r2_-4_Y real',
 'PSSM_r2_-4_V real',
 'PSSM_r2_-3_A real',
 'PSSM_r2_-3_R real',
 'PSSM_r2_-3_N real',
 'PSSM_r2_-3_D real',
 'PSSM_r2_-3_C real',
 'PSSM_r2_-3_Q real',
 'PSSM_r2_-3_E real',
 'PSSM_r2_-3_G real',
 'PSSM_r2_-3_H real',
 'PSSM_r2_-3_I real',
 'PSSM_r2_-3_L real',
 'PSSM_r2_-3_K real',
 'PSSM_r2_-3_M real',
 'PSSM_r2_-3_F real',
 'PSSM_r2_-3_P real',
 'PSSM_r2_-3_S real',
 'PSSM_r2_-3_T real',
 'PSSM_r2_-3_W real',
 'PSSM_r2_-3_Y real',
 'PSSM_r2_-3_V real',
 'PSSM_r2_-2_A real',
 'PSSM_r2_-2_R real',
 'PSSM_r2_-2_N real',
 'PSSM_r2_-2_D real',
 'PSSM_r2_-2_C real',
 'PSSM_r2_-2_Q real',
 'PSSM_r2_-2_E real',
 'PSSM_r2_-2_G real',
 'PSSM_r2_-2_H real',
 'PSSM_r2_-2_I real',
 'PSSM_r2_-2_L real',
 'PSSM_r2_-2_K real',
 'PSSM_r2_-2_M real',
 'PSSM_r2_-2_F real',
 'PSSM_r2_-2_P real',
 'PSSM_r2_-2_S real',
 'PSSM_r2_-2_T real',
 'PSSM_r2_-2_W real',
 'PSSM_r2_-2_Y real',
 'PSSM_r2_-2_V real',
 'PSSM_r2_-1_A real',
 'PSSM_r2_-1_R real',
 'PSSM_r2_-1_N real',
 'PSSM_r2_-1_D real',
 'PSSM_r2_-1_C real',
 'PSSM_r2_-1_Q real',
 'PSSM_r2_-1_E real',
 'PSSM_r2_-1_G real',
 'PSSM_r2_-1_H real',
 'PSSM_r2_-1_I real',
 'PSSM_r2_-1_L real',
 'PSSM_r2_-1_K real',
 'PSSM_r2_-1_M real',
 'PSSM_r2_-1_F real',
 'PSSM_r2_-1_P real',
 'PSSM_r2_-1_S real',
 'PSSM_r2_-1_T real',
 'PSSM_r2_-1_W real',
 'PSSM_r2_-1_Y real',
 'PSSM_r2_-1_V real',
 'PSSM_r2_0_A real',
 'PSSM_r2_0_R real',
 'PSSM_r2_0_N real',
 'PSSM_r2_0_D real',
 'PSSM_r2_0_C real',
 'PSSM_r2_0_Q real',
 'PSSM_r2_0_E real',
 'PSSM_r2_0_G real',
 'PSSM_r2_0_H real',
 'PSSM_r2_0_I real',
 'PSSM_r2_0_L real',
 'PSSM_r2_0_K real',
 'PSSM_r2_0_M real',
 'PSSM_r2_0_F real',
 'PSSM_r2_0_P real',
 'PSSM_r2_0_S real',
 'PSSM_r2_0_T real',
 'PSSM_r2_0_W real',
 'PSSM_r2_0_Y real',
 'PSSM_r2_0_V real',
 'PSSM_r2_1_A real',
 'PSSM_r2_1_R real',
 'PSSM_r2_1_N real',
 'PSSM_r2_1_D real',
 'PSSM_r2_1_C real',
 'PSSM_r2_1_Q real',
 'PSSM_r2_1_E real',
 'PSSM_r2_1_G real',
 'PSSM_r2_1_H real',
 'PSSM_r2_1_I real',
 'PSSM_r2_1_L real',
 'PSSM_r2_1_K real',
 'PSSM_r2_1_M real',
 'PSSM_r2_1_F real',
 'PSSM_r2_1_P real',
 'PSSM_r2_1_S real',
 'PSSM_r2_1_T real',
 'PSSM_r2_1_W real',
 'PSSM_r2_1_Y real',
 'PSSM_r2_1_V real',
 'PSSM_r2_2_A real',
 'PSSM_r2_2_R real',
 'PSSM_r2_2_N real',
 'PSSM_r2_2_D real',
 'PSSM_r2_2_C real',
 'PSSM_r2_2_Q real',
 'PSSM_r2_2_E real',
 'PSSM_r2_2_G real',
 'PSSM_r2_2_H real',
 'PSSM_r2_2_I real',
 'PSSM_r2_2_L real',
 'PSSM_r2_2_K real',
 'PSSM_r2_2_M real',
 'PSSM_r2_2_F real',
 'PSSM_r2_2_P real',
 'PSSM_r2_2_S real',
 'PSSM_r2_2_T real',
 'PSSM_r2_2_W real',
 'PSSM_r2_2_Y real',
 'PSSM_r2_2_V real',
 'PSSM_r2_3_A real',
 'PSSM_r2_3_R real',
 'PSSM_r2_3_N real',
 'PSSM_r2_3_D real',
 'PSSM_r2_3_C real',
 'PSSM_r2_3_Q real',
 'PSSM_r2_3_E real',
 'PSSM_r2_3_G real',
 'PSSM_r2_3_H real',
 'PSSM_r2_3_I real',
 'PSSM_r2_3_L real',
 'PSSM_r2_3_K real',
 'PSSM_r2_3_M real',
 'PSSM_r2_3_F real',
 'PSSM_r2_3_P real',
 'PSSM_r2_3_S real',
 'PSSM_r2_3_T real',
 'PSSM_r2_3_W real',
 'PSSM_r2_3_Y real',
 'PSSM_r2_3_V real',
 'PSSM_r2_4_A real',
 'PSSM_r2_4_R real',
 'PSSM_r2_4_N real',
 'PSSM_r2_4_D real',
 'PSSM_r2_4_C real',
 'PSSM_r2_4_Q real',
 'PSSM_r2_4_E real',
 'PSSM_r2_4_G real',
 'PSSM_r2_4_H real',
 'PSSM_r2_4_I real',
 'PSSM_r2_4_L real',
 'PSSM_r2_4_K real',
 'PSSM_r2_4_M real',
 'PSSM_r2_4_F real',
 'PSSM_r2_4_P real',
 'PSSM_r2_4_S real',
 'PSSM_r2_4_T real',
 'PSSM_r2_4_W real',
 'PSSM_r2_4_Y real',
 'PSSM_r2_4_V real',
 'PSSM_central_-2_A real',
 'PSSM_central_-2_R real',
 'PSSM_central_-2_N real',
 'PSSM_central_-2_D real',
 'PSSM_central_-2_C real',
 'PSSM_central_-2_Q real',
 'PSSM_central_-2_E real',
 'PSSM_central_-2_G real',
 'PSSM_central_-2_H real',
 'PSSM_central_-2_I real',
 'PSSM_central_-2_L real',
 'PSSM_central_-2_K real',
 'PSSM_central_-2_M real',
 'PSSM_central_-2_F real',
 'PSSM_central_-2_P real',
 'PSSM_central_-2_S real',
 'PSSM_central_-2_T real',
 'PSSM_central_-2_W real',
 'PSSM_central_-2_Y real',
 'PSSM_central_-2_V real',
 'PSSM_central_-1_A real',
 'PSSM_central_-1_R real',
 'PSSM_central_-1_N real',
 'PSSM_central_-1_D real',
 'PSSM_central_-1_C real',
 'PSSM_central_-1_Q real',
 'PSSM_central_-1_E real',
 'PSSM_central_-1_G real',
 'PSSM_central_-1_H real',
 'PSSM_central_-1_I real',
 'PSSM_central_-1_L real',
 'PSSM_central_-1_K real',
 'PSSM_central_-1_M real',
 'PSSM_central_-1_F real',
 'PSSM_central_-1_P real',
 'PSSM_central_-1_S real',
 'PSSM_central_-1_T real',
 'PSSM_central_-1_W real',
 'PSSM_central_-1_Y real',
 'PSSM_central_-1_V real',
 'PSSM_central_0_A real',
 'PSSM_central_0_R real',
 'PSSM_central_0_N real',
 'PSSM_central_0_D real',
 'PSSM_central_0_C real',
 'PSSM_central_0_Q real',
 'PSSM_central_0_E real',
 'PSSM_central_0_G real',
 'PSSM_central_0_H real',
 'PSSM_central_0_I real',
 'PSSM_central_0_L real',
 'PSSM_central_0_K real',
 'PSSM_central_0_M real',
 'PSSM_central_0_F real',
 'PSSM_central_0_P real',
 'PSSM_central_0_S real',
 'PSSM_central_0_T real',
 'PSSM_central_0_W real',
 'PSSM_central_0_Y real',
 'PSSM_central_0_V real',
 'PSSM_central_1_A real',
 'PSSM_central_1_R real',
 'PSSM_central_1_N real',
 'PSSM_central_1_D real',
 'PSSM_central_1_C real',
 'PSSM_central_1_Q real',
 'PSSM_central_1_E real',
 'PSSM_central_1_G real',
 'PSSM_central_1_H real',
 'PSSM_central_1_I real',
 'PSSM_central_1_L real',
 'PSSM_central_1_K real',
 'PSSM_central_1_M real',
 'PSSM_central_1_F real',
 'PSSM_central_1_P real',
 'PSSM_central_1_S real',
 'PSSM_central_1_T real',
 'PSSM_central_1_W real',
 'PSSM_central_1_Y real',
 'PSSM_central_1_V real',
 'PSSM_central_2_A real',
 'PSSM_central_2_R real',
 'PSSM_central_2_N real',
 'PSSM_central_2_D real',
 'PSSM_central_2_C real',
 'PSSM_central_2_Q real',
 'PSSM_central_2_E real',
 'PSSM_central_2_G real',
 'PSSM_central_2_H real',
 'PSSM_central_2_I real',
 'PSSM_central_2_L real',
 'PSSM_central_2_K real',
 'PSSM_central_2_M real',
 'PSSM_central_2_F real',
 'PSSM_central_2_P real',
 'PSSM_central_2_S real',
 'PSSM_central_2_T real',
 'PSSM_central_2_W real',
 'PSSM_central_2_Y real',
 'PSSM_central_2_V real',
 'class {0,1}']

# Load data for data processing from csv
# @ Parameters:
#     spark: (SparkSession)
#     train_data_path: Path to the train data file
#     test_data_path: Path to the test data file
# @ return: (DataFrame) Processed data
#     
def operate(spark, train_data_path, test_data_path):
   # constructed data structure
   struct_fields = []
   string_col = []
   for attribute in attributes:
      if "class {0,1}" in attribute:
         struct_fields.append(StructField(attribute, IntegerType(), True))
      elif "real" not in attribute:
         string_col.append(attribute)
         struct_fields.append(StructField(attribute, StringType(), True))
      else:
         struct_fields.append(StructField(attribute, DoubleType(), True))
   schema = StructType(struct_fields)
  
  # Load data from file
   df_traindata = spark.read.csv(train_data_path, header=True, inferSchema=False, schema = schema)
   df_testdata = spark.read.csv(test_data_path, header=True, inferSchema=False, schema = schema)
   df_traindata.cache()
   df_testdata.cache()

   indexers = [StringIndexer(inputCol=col, outputCol=col+"_indexed").fit(df_traindata) for col in string_col]
   for indexer in indexers:
      df_traindata_p1 = indexer.transform(df_traindata)  
      df_testdata_p1 = indexer.transform(df_testdata)
   df_traindata_p1 = df_traindata_p1.drop(*string_col)
   df_testdata_p1 = df_testdata_p1.drop(*string_col) 
   return df_traindata_p1, df_testdata_p1



# Load data for data processing from arff
#     - Block read
# @ Parameters:
#     spark: (SparkSession)
#     file_path: Path to arff file
#     chunksize: chunk size
#     dataschema, datatype: data struct
# @ return: (DataFrame) Processed data
#    
def read_data_in_chunks(spark, file_path, chunksize, dataschema, datatype):
      pd.set_option('future.no_silent_downcasting', True)
      # open file
      with open(file_path, 'r') as f:
         first_iteration = True
         chunks_data = []
         load_lines = 0
         # replace rule
         while True:
               if first_iteration:
                  # Skip head
                  chunk = list(islice(f, len(attributes) + 1))
                  first_iteration = False
                  next(f)
               else:
                  # load the chunk finish
                  if load_lines >= chunksize:
                     # Process character variables and save row data
                     chunk_str = "".join(chunks_data).replace("X", "-999").replace("H", "2").replace("E", "1").replace("C", "0")
                     df = pd.read_csv(io.StringIO(chunk_str), header=None, names=attributes, dtype=datatype)
                     df_spark = spark.createDataFrame(data = df, schema = dataschema)
                     df_spark.cache()
                     chunks_data = []
                     load_lines = 0
                     # Release memory
                     chunk_str, df = None, None
                     gc.collect()
                     yield df_spark

                  # load a line data
                  chunk = list(islice(f, 20000))
                  # All files are loaded
                  if not chunk:
                     raise RuntimeError("Load data complete.")
                  # Balanced sampling 
                  class_0_list = [x for x in chunk if x[-2] == '0']
                  class_1_list = [x for x in chunk if x[-2] == '1']
                  min_class = min(len(class_0_list),len(class_1_list))
                  chunks_data.extend(class_0_list[:min_class])
                  chunks_data.extend(class_1_list[:min_class])
                  load_lines += min_class*2
                  # Release memory
                  chunk, class_0_list, class_1_list, min_class = None, None, None, None
                  gc.collect()


# Load test data for data processing from arff
#     - Block read
# @ Parameters:
#     spark: (SparkSession)
#     file_path: Path to arff file
#     chunksize: chunk size
#     dataschema, datatype: data struct
# @ return: (DataFrame) Processed data
#    
# def read_test_data_in_chunks(spark, file_path, chunksize, dataschema, datatype):
#       pd.set_option('future.no_silent_downcasting', True)
#       # open file
#       with open(file_path, 'r') as f:
#          first_iteration = True
#          while True:
#                if first_iteration:
#                   # Skip head
#                   chunk = list(islice(f, len(attributes) + 1))
#                   first_iteration = False
#                   next(f)
#                else:
#                   # load a chunk data
#                   chunk = list(islice(f, chunksize))
#                   # All files are loaded
#                   if not chunk:
#                      raise RuntimeError("Load data complete.")
#                   # Transform data
#                   chunk_str = "".join(chunk).replace("X", "3").replace("H", "2").replace("E", "1").replace("C", "0")
#                   df = pd.read_csv(io.StringIO(chunk_str), header=None, names=attributes, dtype=datatype)
#                   df_spark = spark.createDataFrame(data = df, schema = dataschema)
#                   df_spark.cache()
#                   # Release memory
#                   chunk,chunk_str,df = None,None,None
#                   gc.collect()
#                   yield df_spark



# get df data schema
def get_data_schema():
   data_type = {}
   struct_fields = []
   string_col = []
   for attribute in attributes:
      if "class {0,1}" in attribute:
         data_type[attribute] = int
         struct_fields.append(StructField(attribute, IntegerType(), True))
      else:
         data_type[attribute] = float
         struct_fields.append(StructField(attribute, DoubleType(), True))
   schema = StructType(struct_fields)
   return schema, data_type, string_col


