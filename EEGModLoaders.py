from tqdm.autonotebook import tqdm
import pickle
from joblib import dump, load
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from cwt import ComplexMorletCWT
tf.compat.v1.disable_eager_execution()
#%env XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
import mne


channel_spec={0:'np_O1',1:'np_P3',2:'np_C3',3:'np_F3',4:'np_F4',5:'np_C4',6:'np_P4',7:'np_O2',
              8:'sf_ch1',9:'sf_ch2',10:'sf_ch3',11:'sf_ch4',12:'sf_ch5',13:'sf_ch6',14:'sf_ch7',15:'sf_ch8',16:'sf_enc'}

import numpy as np
from scipy.signal import resample
class SklearnEEGModelReader():
    def __init__(self,modelpath,statspath, inputdescr, apply_chans_sep=True, voting='max', predtype='optithresh', customthresh=0.5, chnames=list(channel_spec.values())): #predtype can be optithresh, customthresh, defaultthresh, proba
        self.mpath=modelpath #model name structure ex: CLASSIFIER_LogisticRegression_l1_haagladen_20_slowwaves_rawframes_min_<built-in function min>_max_<built-in function max>_wi_150_wa_150_ns_500_tstroc_0.73178616_thresh_0.70
        self.modeln=self.mpath.split('/')[-1].split('.j')[0]
        self.spath=statspath
        self.inputdescr=inputdescr #input descriptor structure: 'CHANNELS_C3_SF_100_WIN_150_DATA_SCALO_scalomagn_scaloreal_scaloimag_lf1_uf40_nscales32_ww1'
        self.apply_chans_sep=apply_chans_sep
        self.voting=voting
        self.predtype=predtype
        self.customthresh=customthresh
        self.mtype=self.modeln.split('_')[0]
        self.algo=self.modeln.split('_')[1]
        self.window=int(self.modeln.split('_wa_')[1].split('_')[0])
        self.thresh=float(self.modeln.split('_thresh_')[1])
        self.channels=inputdescr.split('CHANNELS_')[1].split('_SF_')[0].split(',')
        self.sfrequency=int(inputdescr.split('_SF_')[1].split('_WIN_')[0])
        #self.window=int(inputdescr.split('_WIN_')[1].split('_DATA_')[0])
        self.dtype=inputdescr.split('_DATA_')[1].split('_')[0]
        self.all_channels=chnames
        self.chofis=[self.all_channels.index(i) for i in self.channels]
        if self.dtype=='SCALO':
            self.dstypes=inputdescr.split('_SCALO_')[1].split('_lf')[0].split('_')
            self.lf=float(inputdescr.split('_lf')[1].split('_')[0])
            self.uf=float(inputdescr.split('_uf')[1].split('_')[0])
            self.nscales=int(inputdescr.split('nscales')[1].split('_')[0])
            self.ww=int(inputdescr.split('ww')[1]) #.split('.')[0])
        self.model_data=load(self.mpath)
        if len(self.model_data)>1:
            self.transforms=self.model_data[0]
            self.model=self.model_data[1]
            print('Preprocessing objects found and loaded with the model')
        else:
            self.transforms=[]
            self.model=self.model_data[0]


        with open(self.spath, 'rb') as f:
            self.stats=pickle.load(f)
        print('model data processed')
    def print_model_stats(self):
        print(self.stats['roc_auc'])
        print(self.stats['clsrep'])
        print(self.stats['cm'])
        if 'optimal_threshold' in self.stats.keys():
            print(self.stats['optimal_threshold'])
    def predict(self, raw_data, sampling_rate=125, verb=False): #default neuroplay saampling rate
        preds=[]
        for i in self.chofis:
            self.cdata=raw_data.T[[i]]
            if verb==True:
                print(self.cdata.shape)
                print(self.cdata)
            self.cdata=self.undersample_eeg(data=self.cdata, original_sampling_rate=sampling_rate)
            if verb==True:
                print(self.cdata.shape)
                print(self.cdata)
            self.cdata=self.cdata[:,-self.window:]
            if verb==True:
                print(self.cdata.shape)
                print(self.cdata)
            if len(self.cdata.shape)>1:
                self.cdata=self.cdata[0]
            if self.dtype=='SCALO':
                res=self.get_scalogram(sigprobe=self.cdata)
                self.scalo=res[2]
                self.cdata=self.preddata_fromscalo(scalodata=self.scalo)
                if verb==True:
                    print(self.cdata.shape)
                    print(self.cdata)
            if len(self.cdata.shape)==1:
                self.cdata=self.cdata.reshape(1,-1)
            if len(self.transforms)>0:
                for scalr in self.transforms:
                    self.cdata=scalr.transform(self.cdata)
            if self.predtype in ['optithresh','customthresh','proba']:
                prediction=self.model.predict_proba(self.cdata)[:,1]
                #if verb==True:
                #    print(prediction)
                #prediction=prediction[1]
                if self.predtype=='optithresh':
                    prediction=int(prediction>self.thresh)
                if self.predtype=='customthresh':
                    prediction=int(prediction>self.customthresh)
                if self.predtype=='proba':
                    prediction=prediction[0]
            if self.predtype=='defaultthresh':
                prediction=self.model.predict(self.cdata)
                prediction=prediction[0]

            preds.append(prediction)
        if self.voting=='max':
            self.allcurpreds=preds
            self.current_prediction=np.max(preds)

        print('predicted')
        return self.current_prediction

    def preddata_fromscalo(self, scalodata):
        minp=[]
        dstypes=self.dstypes
        if 'scalomagn' in dstypes:
            minp.append(scalodata[3].reshape(-1))
        if'scaloreal' in dstypes:
            minp.append(scalodata[1].reshape(-1))
        if 'scaloimag' in dstypes:
            minp.append(scalodata[2].reshape(-1))
        minp=np.hstack(minp)
        return minp
    def undersample_eeg(self, data, original_sampling_rate, target_sampling_rate=None):
        if str(target_sampling_rate)!='None':
            target_sampling_rate=int(target_sampling_rate)
        else:
            target_sampling_rate=self.sfrequency
        downsampling_factor = original_sampling_rate / target_sampling_rate
        #downsampled = resample(data, int(len(data) / downsampling_factor))
        downsampled=mne.filter.resample(data, down=downsampling_factor)
        return downsampled


    def get_scalogram(self, sigprobe): #, sf): #, lower_freq = 1, upper_freq = 40, n_scales = 32, wavelet_width = 1):
        sf=self.sfrequency
        lower_freq=self.lf
        upper_freq=self.uf
        n_scales=self.nscales
        wavelet_width=self.ww
        try:
            tf.keras.backend.clear_session()
            cwt = ComplexMorletCWT(wavelet_width, sf, lower_freq, upper_freq, n_scales)
            freqs=cwt.frequencies
            # reshape and change dtype
            prepared_signal = sigprobe.reshape((1, -1, 1)).astype(np.float32)
            # compute the scalogram with the created cwt layer
            scalogram = cwt(prepared_signal)
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())  # In case it wasn't run before
                np_scalogram = sess.run(scalogram)
            scalogram_real = np_scalogram[0, :, :, 0]
            scalogram_imag = np_scalogram[0, :, :, 1]
            scalogram_magn = np.sqrt(scalogram_real ** 2 + scalogram_imag ** 2)
            return cwt, np_scalogram, [np_scalogram, scalogram_real, scalogram_imag, scalogram_magn]
        except Exception as e:
            print(e)
            return None, None, None
        