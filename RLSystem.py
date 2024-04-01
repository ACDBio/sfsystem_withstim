import websocket
import json
import gym
from gym import spaces
import numpy as np
import re
import numexpr as ne
import time
from collections import OrderedDict
import plotly.graph_objects as go
import plotly.subplots as sp
import plotly
import json
import os
import webcolors
import random
import threading
import shutil
plotly.io.json.config.default_engine = 'orjson'
websocket.enableTrace(False)
#if smaple_obsevarions() stuck - restart the esp32 (not by button, but by getting usb oord in-out)

# use_neuroplay=True
# if use_neuroplay:
#     neuroplay_loc='/home/biorp/NeuroPlayPro/NeuroPlayPro.sh'
#     def run_neuroplay():
#         neuroplay_loc = '/home/biorp/NeuroPlayPro/NeuroPlayPro.sh'
#         os.system(f'bash {neuroplay_loc}')
#     neuroplay_thread = threading.Thread(target=run_neuroplay)
#     neuroplay_thread.daemon = True
#     neuroplay_thread.start()

def get_random_css_color_names(n, seed=333):
    """
    Returns a list of n different CSS color names.
    
    Args:
    - n: The number of different color names to retrieve.
    
    Returns:
    - A list of n different CSS color names.
    """
    # Get the list of all CSS color names
    css_color_names = list(webcolors.CSS3_HEX_TO_NAMES.values())
    
    random.seed(seed)
    # Randomly select n different color names
    random_color_names = random.sample(css_color_names, n)
    
    return random_color_names

# Example usage:
# Get a list of 10 different CSS color names
color_names = get_random_css_color_names(50)

#steps are not implemented for now
out_dict={'leddelay':{'names':['leddelay'], 'value_range':{'min':1, 'max':10001, 'step':100}, 'init_val':{'leddelay':10}},
          'ledcontrols':{'names':['lv1r','lv1g','lv1b','lv2r','lv2g','lv2b','lv3r','lv3g','lv3b','lv4r','lv4g','lv4b', 'lv5r','lv5g','lv5b','lv6r','lv6g',
          'lv6b',
          'lv7r',
          'lv7g',
          'lv7b',
          'lv8r',
          'lv8g',
          'lv8b'], 'value_range':{'min':10, 'max':255, 'step':10}, 'init_val':{'lv1r':255, 'lv1g':0, 'lv1b':0, 
                                                                               'lv2r':0, 'lv2b':255, 'lv2b':0, 
                                                                               'lv3r':0, 'lv3g':0, 'lv3b':255, 
                                                                               'lv4r':255, 'lv4g':255, 'lv4b':255, 
                                                                               'lv5r':0,   'lv5g':255, 'lv5b':255, 
                                                                               'lv6r':255, 'lv6g':0,   'lv6b':255, 
                                                                               'lv7r':255, 'lv7b':255, 'lv7b':0, 
                                                                               'lv8r':255, 'lv8g':0, 'lv8b':0}},
          'sound_wave_frequencies':{'names':['wave_1_freq','wave_2_freq'], 'value_range':{'min':1, 'max':30000, 'step':100}, 'init_val':{'wave_1_freq':440, 
                                                                                                                                         'wave_2_freq':440}},
          'panner_phasor_frequencies':{'names':['panner_freq', 'phasor_1_freq', 'phasor_2_freq','phasor_1_min',  'phasor_2_min', 'phasor_1_dif', 'phasor_2_dif'],  'value_range':{'min':1, 'max':50, 'step':1},
                                       'init_val':{'panner_freq':1,
                                                    'phasor_1_freq':10,
                                                    'phasor_2_freq':10,
                                                    'phasor_1_min':1,
                                                    'phasor_2_min':1,
                                                    'phasor_1_dif':30,
                                                    'phasor_2_dif':30}},
          'panner_div':{'names':['panner_div'], 'value_range':{'min':1, 'max':5, 'step':1}, 'init_val':{'panner_div':2}},
          'sound_wave_shapes':{'names':['wave_1_type', 'wave_2_type'], 'value_range':{'min':0, 'max':3, 'step':1}, 
                               'init_val':{'wave_1_type':0,
                                           'wave_2_type':0}},
          'maxivolume':{'names':['maxivolume'], 'value_range':{'min':0, 'max':50, 'step':10}, 
                        'init_val':{'maxivolume':10}}
}
out_order=['lv1r','lv1g','lv1b','lv2r','lv2g','lv2b','lv3r','lv3g','lv3b','lv4r','lv4g','lv4b','lv5r','lv5g','lv5b','lv6r','lv6g','lv6b',
          'lv7r','lv7g','lv7b','lv8r','lv8g','lv8b','leddelay','wave_1_freq','wave_2_freq','panner_freq','panner_div','phasor_1_freq',
          'phasor_1_min','phasor_1_dif','phasor_2_freq','phasor_2_min','phasor_2_dif','maxivolume','wave_1_type','wave_2_type']


channel_spec={0:'np_O1',1:'np_P3',2:'np_C3',3:'np_F3',4:'np_F4',6:'np_C4',7:'np_P4',8:'np_O2',
              9:'sf_ch1',10:'sf_ch2',11:'sf_ch3',12:'sf_ch4',13:'sf_ch5',14:'sf_ch6',15:'sf_ch7',16:'sf_ch8',17:'sf_enc'}



class CustomExceptionWithDetails(Exception):
    """A custom exception class with additional details."""
    def __init__(self, message, detail):
        super().__init__(message)
        self.detail = detail

class SFSystemCommunicator(gym.Env):
    def __init__(self, out_dict=out_dict, out_order=out_order,input_channels=['np_O1','np_P3','np_C3','np_F3','np_F4','np_C4','np_P4','np_O2','sf_enc'], 
                 n_timepoints_per_sample=100, 

                 max_sfsystem_output=9000000, #for 24 bit ADS
                 min_sfsystem_output=-9000000,
                 max_fft_output=10000000,
                 min_fft_output=0,
                 max_bin_output=100000000,
                 min_bin_output=0,

                 reward_formula_string='(fbin_1_4_ch0+freq_30_ch0)/fbin_12_30_ch0', 
                 fbins=[(0,1), (1,4), (4,8), (8,12), (12,30)], delay=10,
                 use_raw_in_os_def=False, use_freq_in_os_def=False, use_fbins_in_os_def=False, device_address="ws://10.42.0.231:80/",
                 step_stim_length_millis=10000, episode_time_seconds=60, render_data=True, return_plotly_figs=False,
                 logfn='current_training.log', log_steps=True, log_episodes=True, log_best_actions_final=True, signal_plot_width=2000, signal_plot_height=1500, training_plot_width=2000, training_plot_height=500, 
                 write_raw=True,
                 write_fft=True,
                 write_bins=True,
                 log_best_actions_every_episode=True,
                 log_actions_every_step=True,
                 render_each_step=True,
                 colors=color_names, 
                 stim_length_on_reset=10,
                 use_abs_values_for_raw_data_in_reward=False,
                 only_pos_encoder_mode=True,
                 log_actions_on_hold=True,
                 channel_spec=channel_spec,
                 use_unfiltered_np_data=True,
                 write_edf_ann=False,
                 edf_ann_fn='default_edf',
                 edf_step_annotation=False,
                 edf_rf_annotation=False,
                 edf_rf_annotation_threshold=1,
                 send_reward_to_display=False,
                 text_size=1,
                 reward_np_sigqual_thresh=90,
                 basic_np_sigqual_thresh=80,
                 send_np_signal_to_display=True):
        

        self.reward_np_sigqual_thresh=reward_np_sigqual_thresh
        self.basic_np_sigqual_thresh=basic_np_sigqual_thresh
        self.send_np_signal_to_display=send_np_signal_to_display


        self.reward_cur=None
        self.send_reward_to_display=send_reward_to_display
        self.text_size=text_size

        self.current_actions=None
        self.ws_sf=None
        self.ws_neuroplay=None
        self.channel_spec=channel_spec
        self.sel_input_channels=input_channels
        self.all_input_channels=list(self.channel_spec.values())
        self.use_neuroplay=False
        self.use_sf=False
        for i in self.sel_input_channels:
            if 'np' in i:
                self.use_neuroplay=True
            if 'sf' in i:
                self.use_sf=True
        if self.use_neuroplay==True:
            self.start_neuroplay()
            time.sleep(10)
        self.use_unfiltered_np_data=use_unfiltered_np_data
 
    


        #edf writing
        self.write_edf_ann=write_edf_ann
        self.edf_ann_fn=edf_ann_fn
        self.edf_step_annotation=edf_step_annotation
        self.edffolder=f'{os.getcwd()}/{edf_ann_fn}'
        self.edfpath=f'{os.getcwd()}/{edf_ann_fn}/{edf_ann_fn}'
        self.edf_rf_annotation=edf_rf_annotation
        self.edf_rf_annotation_threshold=edf_rf_annotation_threshold


        self.enc_is_clicked=0
        self.enc_is_holded=0
        self.log_actions_on_hold=log_actions_on_hold

        self.use_abs_values_for_raw_data_in_reward=use_abs_values_for_raw_data_in_reward
        self.only_pos_encoder_mode=only_pos_encoder_mode
        self.colors=colors
        self.stim_length_on_reset=stim_length_on_reset


        self.device_address=device_address
        self.step_stim_length_millis=step_stim_length_millis
        self.step_stim_length=step_stim_length_millis/1000
        self.episode_time_seconds=episode_time_seconds
        self.n_steps_per_episode=int(self.episode_time_seconds/self.step_stim_length)
        self.cur_step=0
        self.render_data=render_data
        self.return_plotly_figs=return_plotly_figs
        
        self.out_dict=out_dict
        self.out_order=out_order

        self.max_sfsystem_output=max_sfsystem_output
        self.min_sfsystem_output=min_sfsystem_output
        self.max_fft_output=max_fft_output
        self.min_fft_output=min_fft_output
        self.max_bin_output=max_bin_output
        self.min_bin_output=min_bin_output



        self.n_timepoints_per_sample=n_timepoints_per_sample
        self.n_input_channels=len(self.sel_input_channels)
        self.reward_formula_string=reward_formula_string
        self.reward=None

        self.delay=delay

        self.record_raw=use_raw_in_os_def

        self.do_fft=use_freq_in_os_def
        self.record_fft=use_freq_in_os_def

        self.do_fbins=use_fbins_in_os_def
        self.record_fbins=use_fbins_in_os_def
        
        binlist=fbins
        fbins=[]
        for b in binlist:
            bin=tuple(b)
            fbins.append(bin)
        self.fbins=fbins
        self.n_fbins=len(self.fbins)

        self.channels_of_interest_inds=[self.all_input_channels.index(i) for i in self.sel_input_channels]
        self.channels_of_interest_inds.sort()
        self.n_channels_of_interest=len(self.channels_of_interest_inds)

       # self.set_fft_params()

        self.timesleep_period=0.1

        if 'raw' in self.reward_formula_string:
            self.record_raw=True
        if 'freq' in self.reward_formula_string:
            self.record_fft=True
        if 'fbin' in self.reward_formula_string:
            self.record_fbins=True

        if self.record_fbins:
            self.do_fft=True
            self.do_fbins=True
        
        #self.init_action_space()
        #self.init_observation_space()
        #self.set_value_dict_for_reward_function()
        self.connect()
        self.set_delay_and_data_transfer_buffer_size()
        self.set_fft_params()
        self.init_action_space()
        self.init_observation_space()
        self.set_value_dict_for_reward_function()
        if self.only_pos_encoder_mode:
            self.set_pos_encoder_mode()
        print('Delay and data transfer buffer size are set up.')
        self.set_default_actions()
        print('Default actions are set.')


        #logging part
        self.logfn=logfn
        if self.render_data==True:
            self.collect_data_toplot=True

        self.write_raw=write_raw
        self.write_fft=write_fft
        self.write_bins=write_bins


        self.log_steps=log_steps
        self.log_episodes=log_episodes
        
 
        self.clear_reward_buffers()   
        self.clear_reward_stats()
        self.current_episode=0
        self.set_fbin_x_axis_labels()
        
        self.create_log()
        self.signal_plot_width=signal_plot_width
        self.signal_plot_height=signal_plot_height
        self.training_plot_width=training_plot_width
        self.training_plot_height=training_plot_height
        #action logging
        self.best_action_overall=None
        self.best_action_episode=None
        self.log_actions_every_step=log_actions_every_step
        self.log_best_actions_every_episode=log_best_actions_every_episode
        self.log_best_actions_final=log_best_actions_final

        self.render_each_step=render_each_step

        self.done=False
        self.training_completed=False
        self.figures={'signal_fig':[],'training_fig':[]}
        self.colornames=color_names
        self.cur_action_log_no=0
        self.get_np_sig_qual()

    def get_np_sig_qual(self):
        print('Sampling signal quality')
        self.ws_np.send('CurrentDeviceInfo')
        devinfo=json.loads(self.ws_np.recv())
        chnames=devinfo["currentChannelsNames"]
        quals=devinfo["quality"]
        res={}
        for i in range(len(chnames)):
             res[chnames[i]]=quals[i]
        self.chquals=res

    def np_reward_sig_qual_filter(self, thresh=None):
        if str(thresh)=='None':
            if str(self.reward_np_sigqual_thresh)!='None':
                thresh=self.reward_np_sigqual_thresh
        res=True
        if str(thresh)!='None':
            for ch in self.rewardchnames:
                if self.chquals[ch]<thresh:
                    res=False
        self.reward_sig_qual_filter_status=res
        return res
    
    def np_basic_sig_qual_filter(self, thresh=None):
        if str(thresh)=='None':
            if str(self.reward_np_sigqual_thresh)!='None':
                thresh=self.basic_np_sigqual_thresh
        res=True
        if str(thresh)!='None':
            for ch in self.chquals.keys():
                if self.chquals[ch]<thresh:
                    res=False
        self.basic_sig_qual_filter_status=res
        return res

    def run_neuroplay(self):
        neuroplay_loc = '/home/biorp/NeuroPlayPro/NeuroPlayPro.sh'
        os.system(f'bash {neuroplay_loc}')
    def start_neuroplay(self):
        neuroplay_thread = threading.Thread(target=self.run_neuroplay)
        neuroplay_thread.daemon = True
        neuroplay_thread.start()

    def start_edf_log(self):
        self.ws_np.send(f'StartRecord?path={self.edfpath}')
        self.ws_np.recv()
    
    def stop_edf_log(self):
        self.ws_np.send('StopRecord')
        self.ws_np.recv()

    def pause_edf_log(self):
        self.ws_np.send('PauseRecord')
        self.ws_np.recv()
    
    def continue_edf_log(self):
        self.ws_np.send('ContinueRecord')
        self.ws_np.recv()

    def write_edf_annotation_fn(self, ann_text, ann_duration_ms):
        self.ws_np.send(f'AddEDFAnnotation?duration={ann_duration_ms}&text={ann_text}')
        self.ws_np.recv()

    def create_log(self):
        if not os.path.isfile(self.logfn):
            open(self.logfn, 'a').close()
    
    def log_actions(self):
        actionstring=self.get_json_string_from_ordered_dict(self.current_actions)
        actname=f'action_{self.cur_action_log_no}.log'
        with open(actname, 'w') as f:
            actstring=json.dumps(actionstring)
            f.write(actstring)
        self.cur_action_log_no+=1
        return
    def read_and_launch_logged_actions(self, actionlogfn):
        with open(actionlogfn)  as f:
            lines=f.read()
        try:
            linesd=json.loads(lines)
            print(linesd.keys()) 
        except:
            linesd=json.loads(json.loads(lines))
        act=OrderedDict(linesd)
        for key, value in act.items():
            act[key]=np.array(value)
        self.step(act)
    def launch_action_from_json_string(self, actionstring):
        try:
            linesd=json.loads(actionstring)
            print(linesd.keys()) 
        except:
            linesd=json.loads(actionstring)
        act=OrderedDict(linesd)
        for key, value in act.items():
            act[key]=np.array(value)
        self.step(act)

    def write_tolog(self, string):
        with open(self.logfn, 'a') as log_file:
            log_file.write(string + '\n')
    def help(self):
        print('Reward formula can use the following operators: //, *, **, -, +')
        print('It can refer to channels using "ch" prefix followed by an index (starting with 0) e.g. ch0')
        print('It can refer to values of frequency bins in specific channels e.g. fbin_10_50_ch0')
        print('The corresponding freqency bins must be present among fbins passed at the initialization step')
        print('fbins should be passed in the form of [(b1 min, b1max),...(bn min, bn max)]')
        print('It can refer to specific frequencies from fft e.g. freq_50_ch0')
        print('Use integers for frequencies, fractions are not supported for now')
        print('Some examples:')
        print('1. (freq_50_ch0+fbin_0_10_ch0)/(fbin_20_30_ch0)')
        print('2. freq_5_ch0/freq_10_ch0')
        print('3. fbin_05_5_ch0')
    def clear_log(self):
        if os.path.isfile(self.logfn):
            os.remove(self.logfn)
        if self.write_edf_ann:
            if os.path.isdir(self.edffolder):
                shutil.rmtree(self.edffolder)
    def set_value_dict_for_reward_function(self):
        ftokens=re.split(r'[+/)(*]+',self.reward_formula_string)
        self.rewarddict={}
        self.tokendict={}
        self.rewardchnames=[]
        for token in ftokens:
            if 'ch' in token:
                self.rewarddict[token]=None
                self.tokendict[token]={}
                subtokens=token.split('_')
                self.tokendict[token]['datatype']=subtokens[0]
                for subtoken in subtokens:
                    if 'ch' in subtoken:
                        self.tokendict[token]['channelindex']=int(subtoken.split('h')[1])
                        self.rewardchannels=self.all_input_channels[self.tokendict[token]['channelindex']]
                    if subtoken=='freq':
                        tfreq=float(subtokens[1])
                        self.tokendict[token]['freqdata']=tfreq
                        closestind=np.argmin(np.abs(self.f_plot - tfreq))
                        self.tokendict[token]['closest_fft_ind']=closestind
                        print(f'Token {token}:')
                        print(f'Closest fft frequency {self.f_plot[closestind]}')
                    if subtoken=='fbin':
                        bin_lst=[subtokens[1],subtokens[2]]
                        for i in range(2):
                            val=bin_lst[i]
                            if val.startswith('0'):
                                val=float('0.'+val[1:])
                            else:
                                val=float(val)
                            bin_lst[i]=val
                        self.tokendict[token]['freqdata']=tuple(bin_lst)
                        self.tokendict[token]['fbin_idx']=self.fbins.index(self.tokendict[token]['freqdata'])
        self.rewardchnames=list(set(self.rewardchnames))
    def set_default_actions(self):
        action_space_sample=self.action_space.sample()
        for key1 in action_space_sample:
            for key2 in self.out_dict:
                if key1 in self.out_dict[key2]['init_val']:
                    val=self.out_dict[key2]['init_val'][key1]
                    if str(type(val))=="<class 'int'>":
                        val=np.array([val]) 
                    action_space_sample[key1]=val

        self.default_actions=action_space_sample
    def calculate_rms_amplitude(self,  signal):
        return np.sqrt(np.mean(np.square(signal)))
    def calculate_max_amplitude(self, signal):
        return np.max(np.abs(signal))
    def calculate_peak_to_peak_amplitude(self, signal):
        return np.ptp(signal)
    def populate_rewarddict(self, observations):
        for token, tokendata in self.tokendict.items():
            tartype=tokendata['datatype']
            tarchannelidx=tokendata['channelindex']
            if tartype=='raw':
                tarobs=observations['raw_data']
                tarobs=tarobs[:,tarchannelidx]
                if self.use_abs_values_for_raw_data_in_reward==True:
                    tarobs=np.abs(tarobs)
                res=np.sum(tarobs) #here we use the sum, but this may be changed
            if tartype=='rmsamp':
                tarobs=observations['raw_data']
                tarobs=tarobs[:,tarchannelidx]
                res=self.calculate_rms_amplitude(tarobs)
            if tartype=='maxamp':
                tarobs=observations['raw_data']
                tarobs=tarobs[:,tarchannelidx]
                res=self.calculate_max_amplitude(tarobs)
            if tartype=='ptpamp':
                tarobs=observations['raw_data']
                tarobs=tarobs[:,tarchannelidx]
                res=self.calculate_peak_to_peak_amplitude(tarobs)
            if tartype=='freq':
                #tarfreq=tokendata['freqdata']
                taridx=tokendata['closest_fft_ind']
                tarobs=observations['fft'][tarchannelidx]
                res=tarobs[taridx]
            if tartype=='fbin':
                tarobs=observations['fbins'][tarchannelidx]
                res=tarobs[tokendata['fbin_idx']]
            self.rewarddict[token]=res
    def get_reward(self, observations=None, toreturn=False):
        if str(observations) == "None":
            observations=self.observation_space.sample() #if no observations are given extrernally, sample from the observation space
        self.populate_rewarddict(observations)
        self.reward=ne.evaluate(self.reward_formula_string, local_dict=self.rewarddict)
        if toreturn==True:
            return self.reward
    def init_action_space(self):
        self.action_space=spaces.Dict({})
        for key, val in self.out_dict.items():
            spacesnames=val['names']
            spacesrange=val['value_range']
            for spacename in spacesnames:
                self.action_space.spaces[spacename]=spaces.Box(low=spacesrange['min'], high=spacesrange['max'], shape=(1,), dtype=int)
    def init_observation_space(self):
        self.observation_space=spaces.Dict({})
        self.observation_space['raw_data']=spaces.Box(low=self.min_sfsystem_output, high=self.max_sfsystem_output, shape=(self.n_timepoints_per_sample, self.n_channels_of_interest), dtype=int) #n timepoints per sample rows, n input channels columns, signals should be normalized
        if self.record_fft:
            self.observation_space['fft']=spaces.Box(low=self.min_fft_output, high=self.max_fft_output, shape=(self.n_channels_of_interest, self.n_fft_values))
        if self.record_fbins:
            self.observation_space['fbins']=spaces.Box(low=self.min_bin_output, high=self.max_bin_output, shape=(self.n_channels_of_interest, self.n_fbins))
    def set_fft_params(self):
        self.sampling_frequency=int(1000/self.delay)
        self.max_possible_fft_frequency=self.sampling_frequency/2
        self.sampling_period=int(self.delay*self.n_timepoints_per_sample)
        self.tstep=1/self.sampling_frequency
        self.timesteps=np.linspace(0, (self.n_timepoints_per_sample-1)*self.tstep, self.n_timepoints_per_sample)
        self.fstep=self.sampling_frequency/self.n_timepoints_per_sample
        self.f=np.linspace(0, (self.n_timepoints_per_sample-1)*self.fstep, self.n_timepoints_per_sample)
        self.f_plot=self.f[0:int(self.n_timepoints_per_sample/2 + 1)]
        self.n_fft_values=len(self.f_plot)

    def connect(self):
        websocket.enableTrace(False)
        self.ws_sf=websocket.WebSocket()
        self.ws_sf.connect(self.device_address)
        self.sf_connection_status=self.ws_sf.recv()
        if self.sf_connection_status=='Connected':
            print('SF system connected')
        if self.use_neuroplay==True:
            self.ws_np=websocket.WebSocket()
            self.ws_np.connect("ws://localhost:1336/")
            self.ws_connection_status=self.ws_np.connected
            if self.ws_connection_status==True:
                devcount=0
                self.ws_np.send('StartSearch')
                print(self.ws_np.recv())
                while devcount==0:
                    self.ws_np.send('DeviceCount')
                    devcount=json.loads(self.ws_np.recv())["deviceCount"]
                self.ws_np.send('startDevice?id=0')
                print('Neuroplay connected.')
                print(self.ws_np.recv())
                self.ws_np.send('StopSearch')
                print(self.ws_np.recv())
                self.ws_np.send('EnableDataGrabMode')
                print(self.ws_np.recv())
                self.ws_np.send('CurrentDeviceInfo')
                self.np_info=json.loads(self.ws_np.recv())
                self.np_freq=self.np_info['currentFrequency']
                print(self.np_info)
                print('Neuroplay device data received')
        else:
            self.ws_np=None

        
    def set_pos_encoder_mode(self):
        self.ws_sf.send('use_only_pos_enc_mode')
        msg=self.ws_sf.recv()
        print(msg)
    
    def get_fft_fromsignal(self, raw_singlech):
        X=np.fft.fft(raw_singlech)
        X_mag=np.abs(X)/self.n_timepoints_per_sample
        X_mag_plot=2*X_mag[0:int(self.n_timepoints_per_sample/2 + 1)]
        return X_mag_plot

    def get_fft_allchannels(self, raw_data):
        if str(raw_data) == "None":
            raw_data=self.observation_space.sample()['raw_data']

        fft_data=[]
        for chindex in range(raw_data.shape[1]):
            chraw=raw_data[:,chindex]
            chfft=self.get_fft_fromsignal(chraw)
            fft_data.append(chfft)
        fft_data=np.array(fft_data)
        return fft_data
    
    def get_bin_values_from_signal(self, fft_signlech):
        fpl=np.array(self.f_plot)
        xmp=np.array(fft_signlech)
        magnitudes=[]
        for low, high in self.fbins:
            mask = (fpl >= low) & (fpl < high)
            magnitude = np.abs(xmp[mask]).sum() #here can be other functions
            magnitudes.append(magnitude)
        magnitudes=np.array(magnitudes)
        if True in np.isnan(magnitudes):
            print('Warning: nan values among bin values detected!')
        return magnitudes
    def set_fbin_x_axis_labels(self):
        self.fbin_axis_labels=[]
        for low, high in self.fbins:
            binname=f'{low}-{high} Hz'
            self.fbin_axis_labels.append(binname)
    def get_bin_values_allchannels(self, fft=None):
        if str(fft) == "None":
            fft=self.observation_space.sample()['fft']
        fbins_data=[]
        for chindex in range(fft.shape[0]):
            chfft=fft[chindex]
            chbins=self.get_bin_values_from_signal(chfft)
            fbins_data.append(chbins)
        fbins_data=np.array(fbins_data)
        return fbins_data

    def set_delay_and_data_transfer_buffer_size(self):
        self.ws_sf.send("set_delay_and_data_transfer_buffer_size")
        time.sleep(self.timesleep_period)
        setup=False
        if self.use_neuroplay==True:
            self.np_delay=int(1000/self.np_freq)
            if self.delay!=self.np_delay:
                print(f'Setting delay to {self.np_delay} to match neuroplay sampling frequency')
                self.delay=self.np_delay

        while setup==False:
            try:
                device_msg=self.ws_sf.recv()
                if device_msg == "Awaiting delay and data transfer buffer size in shape with space separator":
                    self.ws_sf.send(f'{self.delay},{self.n_timepoints_per_sample}')
                time.sleep(self.timesleep_period)
                device_msg=self.ws_sf.recv()
                if device_msg == "Delay and data transfer buffer size set up":
                    setup = True 
                    break;
            except:
                pass

    # def start_data_transfer_from_device(self):
    #     self.ws_sf.send("start_data_transfer_from_ads")
    # def stop_data_transfer_from_device(self):
    #     self.ws_sf.send("stop_data_transfer_from_ads")
    def stop_audiovis_feedback(self):
        self.ws_sf.send("stop_led_cycle")
    
    def update_audiovis_feedback(self, update_dict=None, print_msg=True):
        self.ws_sf.send("receive_output_control_data")
        outmsg_vals=[]
        for controlnm in self.out_order:
            outmsg_vals.append(update_dict[controlnm][0])
        if print_msg==True:
            print(outmsg_vals)
        outmsg_vals=list(map(int, outmsg_vals))
        self.current_control_msg=','.join(list(map(str,outmsg_vals)))
        if print_msg==True:
            print(self.current_control_msg)
        self.ws_sf.send(self.current_control_msg)

    def synth_data(self, signal_freq=30):
        y=1*np.sin(2*np.pi*signal_freq*self.timesteps)
        return y
    
    def sample_fromsf(self):
        #try:
        if self.ws_sf.connected:
            self.current_sample_sf=None
            self.ws_sf.send("start_data_transfer_from_ads")
            self.current_sample_sf=json.loads(self.ws_sf.recv())
            self.ws_sf.send("stop_data_transfer_from_ads")
            return True
        else:
            print('No connection')
            return True
            self.close()
                #raise CustomExceptionWithDetails("Unable to sample", "No connection")
                
        #except:
            #raise CustomExceptionWithDetails("Unable to sample", "")
    
    def sample_observations(self, use_synth_data=False): #True for testing of fft etc., False - for actual application

        if self.ws_np is not None:
            self.get_np_sig_qual()
            self.np_reward_sig_qual_filter()
            self.np_basic_sig_qual_filter()


        if self.use_sf==True:
            sf_thread = threading.Thread(target=self.sample_fromsf)
            sf_thread.daemon = True
            sf_thread.start()
            sf_thread.join()
           # self.ws_sf.send("start_data_transfer_from_ads")
        self.raw_data=[]
        # np_unsampled=True
        if self.use_neuroplay==True:
        #     while np_unsampled:
            try:
                if self.use_unfiltered_np_data==True:
                    self.ws_np.send('grabRawData')
                else:
                    self.ws_np.send('grabFilteredData')
                self.current_sample_np=np.array(json.loads(self.ws_np.recv())['data'])
                #np_unsampled=False
            except Exception as e:
                print(e)
                return False

        else:
            self.current_sample_np=np.zeros(shape=(8,1250))
        for i in self.current_sample_np:
                self.raw_data.append(i[-self.n_timepoints_per_sample:])   
        print('sampled from np')
        if self.use_sf==True:
            #print("HERE")
            #self.current_sample_sf=json.loads(self.ws_sf.recv())
            #self.ws_sf.send("stop_data_transfer_from_ads")
            for key, value in self.current_sample_sf.items():
                if use_synth_data==False:
                    if key not in ['enc_is_clicked', "enc_is_holded"]:
                        self.raw_data.append(np.array(value))
                    else:
                        if key=='enc_is_clicked':
                            self.enc_is_clicked=value[0]
                            if self.enc_is_clicked==1:
                                if  self.use_neuroplay==True:
                                    if self.write_edf_ann==True:
                                        self.write_edf_annotation_fn(ann_text='enc_click',ann_duration_ms=self.delay)
                        if key=='enc_is_holded':
                            self.enc_is_holded=value[0]
                            if self.enc_is_holded==1:
                                if self.use_neuroplay==True:
                                    if self.write_edf_ann==True:
                                        self.write_edf_annotation_fn(ann_text='enc_holded',ann_duration_ms=self.delay)

                else:
                    self.raw_data.append(self.synth_data())
            #print('HERE in sampling')
            #print(np.array(self.raw_data).shape)
            #self.raw_data_sf=np.array(self.raw_data).transpose()
        else:
            self.current_sample_sf=np.zeros(shape=(9, self.n_timepoints_per_sample))
            for i in self.current_sample_sf:
                    self.raw_data.append(i)         
        self.raw_data=np.array(self.raw_data, dtype=float).transpose() 
        #print(np.array(self.raw_data).shape)   
        #elf.raw_data_all=self.raw_data
        #self.raw_data=self.raw_data[:,self.channels_of_interest_inds]
        #print(np.array(self.raw_data).shape)
        print('sampled from sf')
        return True
    def sample_and_process_observations_from_device(self):
        self.correct_observations=False
        new_observations=dict()
        self.new_observations_tarchs=dict()
        #while self.correct_observations==False:
        #    try:
        self.correct_observations=self.sample_observations()
        #    except:
        #        raise ConnectionAbortedError
        new_observations['raw_data']=self.raw_data
        self.new_observations_tarchs['raw_data']=new_observations['raw_data'][:,self.channels_of_interest_inds]
        if self.do_fft:
         self.fft=self.get_fft_allchannels(raw_data=self.raw_data)
         if self.record_fft:
            new_observations['fft']=self.fft
            self.new_observations_tarchs['fft']=new_observations['fft'][self.channels_of_interest_inds,:]
        if self.do_fbins:
            self.fbins_data=self.get_bin_values_allchannels(fft=self.fft)
            if self.record_fbins:
                new_observations['fbins']=self.fbins_data 
                self.new_observations_tarchs['fbins']=new_observations['fbins'][self.channels_of_interest_inds,:]
        new_observations=OrderedDict(new_observations)
        return new_observations
    def write_signal_logs(self):
        if self.record_fft==True:
            if self.write_fft==True:
                self.write_tolog(json.dumps({'fft':self.cur_observations['fft'].tolist()}))
        if self.record_fbins==True:
            if self.write_bins==True:
                self.write_tolog(json.dumps({'fbins':self.cur_observations['fbins'].tolist()}))
        if self.record_raw==True:
            if self.write_raw==True:
                self.write_tolog(json.dumps({'raw_data':self.cur_observations['raw_data'].tolist()}))
    def step(self, action):
        #print(action)
        try:
            self.current_actions=action
            actionstring=self.get_json_string_from_ordered_dict(action)
            if self.ws_sf is not None:
                if self.ws_np is not None:
                    if self.write_edf_ann==True:
                        if self.edf_step_annotation==True:
                            self.write_edf_annotation_fn(ann_text=f'episode_{self.current_episode}_step_{self.cur_step}', ann_duration_ms=self.step_stim_length_millis)

                self.done=False
                self.best_overall_reward_now=False
                self.best_episode_reward_now=False
                self.best_total_episode_reward_now=False


                self.update_audiovis_feedback(update_dict=action)
                time.sleep(self.step_stim_length)
                # if self.ws_np is not None:
                #     self.get_np_sig_qual()
                #     self.np_reward_sig_qual_filter()
                #     self.np_basic_sig_qual_filter()
                new_observations=self.sample_and_process_observations_from_device()
                self.cur_observations=new_observations
                reward=self.get_reward(observations=new_observations, toreturn=True)
                reward_val=reward.tolist()
                #if action!=self.default_actions:
                self.reward_cur=reward
                self.total_cur_episode_reward+=reward_val
                if self.ws_np is not None:
                    if self.write_edf_ann==True:
                        if self.edf_step_annotation==True:
                            anntxt=f'sr_{np.round(reward_val,4)}_tcer_{np.round(self.total_cur_episode_reward,4)}' #f'step_reward_{reward_val}_total_current_episode_reward_{self.total_cur_episode_reward}_action_{actionstring}'
                            self.write_edf_annotation_fn(ann_text=anntxt, ann_duration_ms=self.delay)


                if self.ws_np is not None:
                    if self.write_edf_ann:
                        if self.edf_rf_annotation:
                            if reward_val>self.edf_rf_annotation_threshold:
                                self.write_edf_annotation_fn(ann_text=f'r_{self.reward_formula_string}_t_{self.edf_rf_annotation_threshold}', ann_duration_ms=self.delay)

                if reward_val>=self.episode_max_reward:
                        self.episode_max_reward=reward_val
                        self.best_episode_reward_now=True
                        self.best_action_episode=action

                if reward_val>=self.overall_max_reward:
                        #print('setting best overall reward')
                        self.overall_max_reward=reward_val
                        self.best_overall_reward_now=True
                        self.best_action_overall=action
                        if self.ws_np is not None:
                            if self.write_edf_ann==True:
                                anntxt=f'comr_{np.round(reward_val,4)}' #f'current_overall_max_reward_{reward_val}_action_{actionstring}_inprev_{self.step_stim_length}_s'
                                self.write_edf_annotation_fn(ann_text=anntxt, ann_duration_ms=self.delay)


                if self.total_cur_episode_reward>=self.total_episode_max_reward:
                    self.total_episode_max_reward=self.total_cur_episode_reward
                    self.best_total_episode_reward_now=True
                    if self.ws_np is not None:
                        if self.write_edf_ann==True:
                            anntxt=f'temr_{np.round(self.total_cur_episode_reward,4)}'#f'current_overall_max_reward_{reward_val}_action_{actionstring}_inprev_{self.step_stim_length}_s'
                            self.write_edf_annotation_fn(ann_text=anntxt, ann_duration_ms=self.delay)


                if self.collect_data_toplot:
                    self.cur_episode_rewards.append(reward_val)
                if self.log_steps:
                    self.write_tolog(json.dumps({'Episode':self.current_episode, 'Step': self.cur_step, 'Step reward': reward_val}))
                    if self.ws_np is not None:
                        self.write_tolog(json.dumps({'NP chqual':self.chquals, 'NP reward qual filter status':self.reward_sig_qual_filter_status, 'NP reward sig qual filter thresh': self.reward_np_sigqual_thresh}))


                if self.log_actions_every_step:
                    self.write_tolog(json.dumps({'Action:':reward_val}))
                    self.write_tolog(json.dumps({'Action reward':reward_val}))
                    self.write_tolog(actionstring)
                self.write_signal_logs()        
                    


                if self.cur_step<self.n_steps_per_episode:
                    self.done=False
                    self.cur_step+=1
                else:
                    self.done=True
                if self.render_each_step==True:
                    self.render()
                #print('step_done')
                if self.log_actions_on_hold==True:
                    
                    #print(self.enc_is_holded)
                    #print(self.current_sample)
                    if self.enc_is_holded:
                        self.log_actions()
                if self.send_reward_to_display and not self.send_np_signal_to_display:
                    msg='display_text:'+str(self.text_size)+':'+str(np.round(reward,3))
                    self.ws_sf.send(msg)
                if self.send_np_signal_to_display and not self.send_reward_to_display:
                    sigquals=' '.join([f'{ch} {i}' for ch,i in self.chquals.items()])
                    msg='display_text:'+'1'+':'+str(sigquals)
                    self.ws_sf.send(msg)
                if self.send_reward_to_display and self.send_np_signal_to_display:
                    sigquals=' '.join([f'{ch} {i}' for ch,i in self.chquals.items()])
                    msg='display_text:'+'1'+':'+f'R {np.round(reward,3)} Quals {sigquals}'
                    self.ws_sf.send(msg)

                return self.new_observations_tarchs, reward, self.done, {} #False
            else:
                print('No connection')
                return False, False, False, False #False
        except:
            return False, False, False, False
            #raise CustomExceptionWithDetails("Step unfinished", 'No connection')
    def clear_all_stats(self):
        self.episode_max_reward=0
        self.total_cur_episode_reward=0
        self.best_episode_reward_now=False #just in case
        self.best_overall_reward_now=False #just in case
        self.best_total_episode_reward_now=False #just in case

    def reset(self):
        #print('resetting')
        if self.cur_step>0:
            if self.done:
                if self.collect_data_toplot:
                    self.previous_episodes_total_rewards.append(self.total_cur_episode_reward)
                    self.previous_episodes_max_rewards.append(self.episode_max_reward)
                if self.log_episodes:
                    self.write_tolog(json.dumps({'Episode':self.current_episode, 'Episode total reward': self.total_cur_episode_reward, 'Episode max reward': self.episode_max_reward}))
                if self.log_best_actions_every_episode:
                    actionstring=self.get_json_string_from_ordered_dict(self.best_action_episode)
                    self.write_tolog(json.dumps({'Best action in the episode reward':self.episode_max_reward}))
                    self.write_tolog(actionstring)
        self.stop_audiovis_feedback()
        self.cur_step=0
        self.done=False
        self.update_audiovis_feedback(update_dict=self.default_actions) #update back to default actions
        
        if self.collect_data_toplot==True:
            self.previous_episodes_max_rewards.append(self.episode_max_reward)
            self.cur_episode_rewards=[]
            
        time.sleep(self.stim_length_on_reset) #prepare the brain for the next episode 
        new_observations=self.sample_and_process_observations_from_device() #sampling new observations after reset
        self.cur_observations=new_observations
        return self.new_observations_tarchs, {}

    def get_json_string_from_ordered_dict(self, od):
        od=dict(od)
        for key, value in od.items():
            od[key]=value.tolist()
        return json.dumps(od)

    def render(self, elems=['reward_lineplots', 'current_fft', 'current_fbins', 'current_raw'], return_figs=False, jnb=False, colors=None):
        if str(colors)=='None':
            colors=self.colors
        if jnb:
            clear_output(wait=True)
        if return_figs==None:
            return_figs=self.return_plotly_figs
        if 'reward_lineplots' in elems:
            training_fig=sp.make_subplots(rows=2, cols=2)
            training_fig.update_layout(width=self.training_plot_width, height = self.training_plot_height)
            training_fig.add_trace(sp.go.Scatter(x=list(range(len(self.cur_episode_rewards))), y=self.cur_episode_rewards, mode='lines+markers', name='Current episode rewards'), row=1, col=1)
            training_fig.add_trace(sp.go.Scatter(x=list(range(len(self.previous_episodes_max_rewards))), y=self.previous_episodes_max_rewards, mode='lines+markers', name='Previous episode max rewards'), row=1, col=2)
            training_fig.add_trace(sp.go.Scatter(x=list(range(len(self.previous_episodes_total_rewards))), y=self.previous_episodes_total_rewards, mode='lines+markers', name='Previous episode total rewards'), row=2, col=1)           
            self.figures['training_fig']=training_fig
            if self.render_data:
                if jnb==True:
                    training_fig.show()
        if ('current_fft' in elems) or ('current_fbins' in elems):
            if self.record_fft or self.record_fbins:
                signal_fig=sp.make_subplots(rows=self.n_channels_of_interest, cols=3)
                signal_fig.update_layout(width=self.signal_plot_width, height = self.signal_plot_height)
                if 'current_fft' in elems and self.record_fft:
                    for chidx in range(self.n_channels_of_interest):
                        color=colors[chidx]
                        orig_chidx=self.channels_of_interest_inds[chidx]
                        chname=self.sel_input_channels[chidx]
                        chfft=self.cur_observations['fft'][chidx]
                        signal_fig.add_trace(sp.go.Scatter(x=self.f_plot[1:], y=chfft[1:], mode='lines', name=f'Channel {chname} spectrum', line=dict(color=color)), row=chidx+1, col=1)
                if 'current_fbins' in elems and self.record_fbins:
                    for chidx in range(self.n_channels_of_interest):
                        color=colors[chidx]
                        orig_chidx=self.channels_of_interest_inds[chidx]
                        chname=self.sel_input_channels[chidx]
                        chbins=self.cur_observations['fbins'][chidx]
                        signal_fig.add_trace(sp.go.Bar(x=self.fbin_axis_labels, y=chbins, name=f'Channel {chname} frequency bins', marker=dict(color=color)), row=chidx+1, col=2)
                if 'current_raw' in elems and self.record_raw:
                    for chidx in range(self.n_channels_of_interest):
                        color=colors[chidx]
                        orig_chidx=self.channels_of_interest_inds[chidx]
                        chname=self.sel_input_channels[chidx]
                        chraw=self.cur_observations['raw_data'][:,chidx]
                        signal_fig.add_trace(sp.go.Scatter(x=list(range(len(chraw))), y=chraw, mode='lines', name=f'Channel {chname} raw signal', line=dict(color=color)), row=chidx+1, col=3)                
                
                
                if self.render_data:
                    if jnb==True:
                        signal_fig.show()
                self.figures['signal_fig']=signal_fig

        if return_figs:
            return self.figures
    def clear_reward_buffers(self):
        self.cur_episode_rewards=[]
        self.previous_episodes_max_rewards=[]
        self.previous_episodes_total_rewards=[]
    def clear_reward_stats(self):
        self.episode_max_reward=0
        self.overall_max_reward=0
        self.total_cur_episode_reward=0
        self.total_episode_max_reward=0
        self.best_episode_reward_now=False
        self.best_overall_reward_now=False
        self.best_total_episode_reward_now=False
    def close(self, clear_log=False):
        if self.ws_sf.connected:
            self.ws_sf.send("turn_off_display")
            self.stop_audiovis_feedback() #just in case
            self.ws_sf.send("stop_data_transfer_from_ads") #just in case
        if self.log_best_actions_final:
            if str(self.best_action_overall) != 'None':
                actionstring=self.get_json_string_from_ordered_dict(self.best_action_overall)
                self.write_tolog(json.dumps({'Best action across episodes reward':self.overall_max_reward}))
                self.write_tolog(actionstring)
        if self.ws_np.connected:
            if self.use_neuroplay:
                self.ws_np.send("stopSearch")
                self.ws_np.send("close")
        self.cur_step=0 #just in case
        self.current_episode=0

        self.clear_reward_buffers()
        self.clear_reward_stats()
        #try:
        if self.ws_np.connected:
            self.stop_edf_log()
        #except Exception as e:
        #    print(f'On env close received {e}')
            #return True
        
        if clear_log:
            self.clear_log()
        if self.ws_sf.sock is not None:
            self.ws_sf.close()
        if self.use_neuroplay:
            if self.ws_np is not None:
                self.ws_np.close()
        
               


class FlattenActionSpaceWrapper(gym.Wrapper):
    def __init__(self, env, min_value=-1.0, max_value=1.0):
        super().__init__(env)
        # Flatten the action space
        self.action_space = self._flatten_action_space(env.action_space, min_value, max_value)
        # Store the normalization valuesNo
        self.min_value = min_value
        self.max_value = max_value

    def _flatten_action_space(self, original_space, min_value, max_value):
        # Calculate the total size of the flattened action space
        total_size = sum(space.shape[0] for space in original_space.spaces.values())
        # Create a new Box space with the total size and normalized bounds
        return spaces.Box(low=np.float32(np.full(total_size, min_value)),
                            high=np.float32(np.full(total_size, max_value)),
                            shape=(total_size,),
                            dtype=np.float32) # Assuming the action space is continuous

    def _unflatten_action(self, action):
        # Split the flattened action into the original spaces
        original_action = {}
        start = 0
        for key, space in self.env.action_space.spaces.items():
            end = start + space.shape[0]
            original_action[key] = action[start:end]
            start = end
        return original_action

    def unnormalize_action(self, normalized_action):
        """
        Unnormalizes an action from the range [min_value, max_value] back to its original scale.
        
        Args:
        - normalized_action: A dictionary representing the normalized action.
        
        Returns:
        - The unnormalized action.
        """
        unnormalized_action = {}
        for key, space in self.env.action_space.spaces.items():
            unnormalized_action[key] = space.low + (normalized_action[key] - self.min_value) * (space.high - space.low) / (self.max_value - self.min_value)
        return unnormalized_action

    def step(self, action):
        # Convert the flattened action back to the original action space format
        original_action = self._unflatten_action(action)
        # Unnormalize the action
        unnormalized_action = self.unnormalize_action(original_action)
        return super().step(unnormalized_action)


    def flatten_and_normalize_action(self, action_space, action, min_value=None, max_value=None):
        """
        Flattens and normalizes a given unflattened and unnormalized action for a gym environment with a dictionary action space.
        
        Parameters:
        - action_space: A dictionary where keys represent different action components and values are Box objects from gym.
        - action: The action to be flattened and normalized.
        
        Returns:
        - The flattened and normalized action.
        """
        if str(min_value)=='None':
            min_value=self.min_value
        if str(max_value)=='None':
            max_value=self.max_value
        # Initialize an empty list to store the normalized action
        normalized_action = []
        
        # Iterate over each component in the action space
        for component in action_space:
            space=action_space[component]
            # Extract the low and high bounds of the Box
            low, high = space.low[0], space.high[0]
            
            # Normalize the action component based on its own range
            normalized_value = min_value + (action[component] - low) * (max_value - min_value) / (high - low)
            normalized_value=normalized_value[0]
            normalized_value = max(min(normalized_value, max_value), min_value)
            # Append the normalized value to the normalized action list
            normalized_action.append(normalized_value)
        
        return np.array(normalized_action)


    
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from gym.wrappers import FlattenObservation
from stable_baselines3 import PPO, SAC, DDPG, TD3, A2C, DQN
from IPython.display import clear_output

class stable_baselines_model_trainer():
    def __init__(self, initialized_environment, algorithm='A2C', policy='MlpPolicy', logfn='model_stats.log', n_steps_per_timestep=1,
                    start_on_click=False,
                    pause_learning_if_reward_sig_qual_false=False,
                    start_on_reward_sig_qual=False):
        self.start_on_click=start_on_click
        self.pause_learning_if_reward_sig_qual_false=pause_learning_if_reward_sig_qual_false
        self.start_on_reward_sig_qual=start_on_reward_sig_qual



        self.env=initialized_environment
        self.orig_env=initialized_environment
        self.env = FlattenObservation(self.env)
        self.env = FlattenActionSpaceWrapper(self.env)
        self.algorithm=algorithm
        self.policy=policy
        self.n_steps_per_timestep=n_steps_per_timestep #for PPO and A2C
        self.set_model()
        self.max_test_reward=0
        self.logfn=logfn
        self.collect_environment_data()
        self.cur_episode_no=None
        self.stat1=None
        self.stat2=None
        self.n_total_timesteps=None
        self.num_episodes=None
        self.training_completed=False
        if not os.path.isfile(self.logfn):
            open(self.logfn, 'a').close()
        with open(self.logfn, 'a') as log_file:
            log_file.write(json.dumps(self.env_data) + '\n')

        self.default_env_actions=self.env.flatten_and_normalize_action(self.orig_env.action_space, self.env.default_actions)
        if  self.env.use_neuroplay==True:
            if self.env.write_edf_ann==True:
                self.env.start_edf_log()
                print('EDF log started:')
                print(self.env.edfpath)


    def direct_feedback_run(self, reward_mapping_min, reward_mapping_max, overlay_random, mapped_outputs, min_space_value=-1, max_space_value=1):
        if  self.env.use_neuroplay==True:
            if self.env.write_edf_ann==True:
                self.env.write_edf_annotation_fn('started_direct_feedback_run', self.env.delay)
        
        #dont forget to run at least one iteration of the environment for reward setup
        reward=self.env.reward
        reward_scaled=(reward-reward_mapping_min)/(reward_mapping_max-reward_mapping_min)
        action_modifier=min_space_value+(max_space_value-min_space_value)*reward_scaled
        if action_modifier>1: #just in case
            action_modifier=1

        if overlay_random==False:
            new_actions=self.default_env_actions
        else:
            new_actions=self.env.action_space.sample()
        rgbidxs=list(range(1,25))
        r_idxs=list(range(1,25,3))
        g_idxs=list(range(2,25,3))
        b_idxs=list(range(3,25,3))
        #rgbkeys=['lv1r','lv1g','lv1b','lv2r','lv2g','lv2b','lv3r','lv3g','lv3b','lv4r','lv4g','lv4b','lv5r','lv5g','lv5b','lv6r','lv6g','lv6b','lv7r','lv7g','lv7b','lv8r','lv8g','lv8b']
        for ma in mapped_outputs:
            if ma!='Flash frequency':
                action_modifier=min_space_value+(max_space_value-min_space_value)*reward_scaled
            else:
                action_modifier=max_space_value-(max_space_value-min_space_value)*reward_scaled
            if ma=='Flash frequency':
                new_actions[0]=action_modifier
            if ma=='LED intensity':
                for idx in rgbidxs:
                    new_actions[idx]=action_modifier
            if ma=='R':
                for idx in r_idxs:
                    new_actions[idx]=action_modifier
            if ma=='G':
                for idx in g_idxs:
                    new_actions[idx]=action_modifier
            if ma=='B':
                for idx in b_idxs:
                    new_actions[idx]=action_modifier
            if ma=='Wave 1 frequency':
                new_actions[25]=action_modifier   
            if ma=='Wave 2 frequency':
                new_actions[26]=action_modifier
            if ma=='Panner freq':
                new_actions[27] = action_modifier
            if ma=='Sound volume':
                new_actions[37] = action_modifier
        
        self.env.step(new_actions)
        return

    def static_launch(self):
        try:
            if  self.env.use_neuroplay==True:
                if self.env.write_edf_ann==True:
                    self.env.write_edf_annotation_fn('started_static_run', self.env.delay)
            r1, r2, r3, r4 = self.orig_env.step(self.orig_env.default_actions)
            if str(r4)=='False':
                return False
        except:
            return True
    def dynamic_launch(self):
        if self.env.use_neuroplay==True:
            if self.env.write_edf_ann==True:
                self.env.write_edf_annotation_fn('started_dynamic_run', self.env.delay)
        try:
            r1, r2, r3, r4 = self.env.step(self.env.action_space.sample())
            if str(r4)=='False':
                return False
        except:
            return True
    def set_model(self):
        if self.algorithm=='PPO':
            self.model = PPO(self.policy, self.env, n_steps=self.n_steps_per_timestep)
            self.model_blank=PPO
        if self.algorithm=='SAC':
            self.model = SAC(self.policy, self.env)
            self.model_blank=SAC
        if self.algorithm=='DDPG':
            self.model = DDPG(self.policy, self.env)
            self.model_blank=DDPG
        if self.algorithm=='TD3':
            self.model = TD3(self.policy, self.env)
            self.model_blank=TD3
        if self.algorithm=='A2C':
            self.model = A2C(self.policy, self.env, n_steps=self.n_steps_per_timestep)
            self.model_blank=A2C
        if self.algorithm=='DQN':
            self.model = DQN(self.policy, self.env)
            self.model_blank=DQN
    
    def close_env(self):
        self.env.close()

    def train(self, num_episodes=5, log_model=True, get_plots=False, render_plots=False,n_total_timesteps=1, log_or_plot_every_n_timesteps=1, 
              jnb=False,  pause_on_click=False):
        print(self.env.edfpath)
        self.n_total_timesteps=n_total_timesteps
        self.num_episodes=num_episodes
        env_paused=False

        start=True
        if  self.start_on_click==True:
            start=False
            while start==False:
                self.env.sample_observations()
                if self.env.is_clicked:
                    start=True
                    self.env.ws_sf.send('display_text:2:STARTED')
                    break
                else:
                    self.env.ws_sf.send('display_text:2:AWAITING ENC CLICK')
        if  self.start_on_reward_sig_qual == True:
            start=False
            while start==False:
                self.env.sample_observations()
                if self.env.reward_sig_qual_filter_status:
                    start=True
                    self.env.ws_sf.send('display_text:2:STARTED')
                    break
                else:
                    self.env.ws_sf.send('display_text:2:AWAITING QUALITY SIGNAL')

        if self.env.use_neuroplay==True:
            if self.env.write_edf_ann==True:
                self.env.write_edf_annotation_fn('started_training_run', self.env.delay)
        if start:
            if self.env.ws_sf.sock is not None:
                self.training_completed=False
                #if n_total_timesteps=='episode':
                #    n_total_timesteps=int(self.env.n_steps_per_episode/self.n_steps_per_timestep) #we run one episode + 1 step before resetting, episode 
                for i in range(num_episodes):
                    self.cur_episode_no=i
                    self.cur_n_timesteps=0
                    while self.cur_n_timesteps<int(n_total_timesteps): #here-for A2C
                        if self.env.ws_sf.sock is not None:
                            if env_paused==False:
                                if self.env.use_neuroplay==True:
                                    if self.env.write_edf_ann==True:
                                        self.env.write_edf_annotation_fn(f't_episode_{i}', self.env.delay)
                                #print('h1')
                                tocontinue=True
                                if self.pause_learning_if_reward_sig_qual_false==True:
                                    tocontinue=False
                                    while tocontinue==False:
                                        self.env.sample_observations()
                                        if self.env.reward_sig_qual_filter_status:
                                            self.env.ws_sf.send('display_text:1:SIGNAL OK')
                                            tocontinue=True
                                            break
                                        else:
                                            self.env.ws_sf.send('display_text:1:AWAITING QUALITY SIGNAL')
                                            
                                            

                                self.model.learn(total_timesteps=log_or_plot_every_n_timesteps)
                                #print('h2')
                                #print(self.env.enc_is_clicked)
                                #print(self.env.current_sample)
                                self.cur_n_timesteps+=log_or_plot_every_n_timesteps
                                if render_plots:
                                    if get_plots:
                                        self.figs=self.env.render(return_figs=True)
                                    else:
                                        self.env.render()
                                    if jnb:
                                        clear_output(wait=True)
                                if log_model:
                                    self.model.save("last_model")
                                    with open(self.logfn, 'a') as log_file:
                                            self.stat0=f'target {self.env.reward_formula_string}, current last_model reward {self.env.reward_cur}, file best_overall_reward_model'
                                            log_file.write(self.stat0 + '\n')                                
                                    if self.env.best_overall_reward_now:
                                        self.model.save("best_overall_reward_model")
                                        with open(self.logfn, 'a') as log_file:
                                            self.stat1=f'target {self.env.reward_formula_string}, current best_overall_reward_model reward {self.env.overall_max_reward}, file best_overall_reward_model'
                                            log_file.write(self.stat1 + '\n')
                                    if self.env.best_total_episode_reward_now:
                                        self.model.save("best_total_episode_reward_model")
                                        with open(self.logfn, 'a') as log_file:
                                            self.stat2=f'target {self.env.reward_formula_string}, current best_total_episode_reward_model reward {self.env.total_episode_max_reward}, file best_total_episode_reward_model'
                                            log_file.write(self.stat2 + '\n')
                                if pause_on_click==True:
                                    if self.env.enc_is_clicked==1:
                                        env_paused=True
                            else:
                                self.env.step(self.env.flatten_and_normalize_action(self.orig_env.action_space, self.env.current_actions))
                                self.env.sample_observations()
                                if self.env.enc_is_clicked==1:
                                    env_paused=False                            
                        
                            self.env.clear_all_stats()
                        else:
                            print('Connection stopped.')
                            break
                
                self.env.stop_audiovis_feedback()
                self.training_completed=True 
                if self.env.use_neuroplay==True:
                    self.env.pause_edf_log()   
                return     
            else:
                print('No connection.')
                return  
    def collect_environment_data(self):
        env_data=dict()
        env_data['out_dict']=self.env.out_dict
        env_data['session_settings']=dict()
        env_data['channels_of_interest']=self.env.sel_input_channels
      #  env_data['session_settings']['n_input_channels']=self.env.n_input_channels #should correct this!
      #  env_data['session_settings']['channels_of_interest_inds']=self.env.channels_of_interest_inds
        env_data['session_settings']['n_timepoints_per_sample']=self.env.n_timepoints_per_sample
      #  env_data['session_settings']['max_sfsystem_output']=self.env.max_sfsystem_output
        env_data['session_settings']['reward_formula_string']=self.env.reward_formula_string
        env_data['session_settings']['fbins']=self.env.fbins
        env_data['session_settings']['delay']=self.env.delay
        env_data['session_settings']['use_raw_in_os_def']=self.env.record_raw
        env_data['session_settings']['use_freq_in_os_def']=self.env.record_fft
        env_data['session_settings']['use_fbins_in_os_def']=self.env.record_fbins
        env_data['session_settings']['device_address']=self.env.device_address
        env_data['session_settings']['step_stim_length_millis']=self.env.step_stim_length_millis
        env_data['session_settings']['episode_time_seconds']=self.env.episode_time_seconds
        env_data['session_settings']['logfn']=self.env.logfn
        env_data['session_settings']['algorithm']=self.algorithm
        env_data['session_settings']['policy']=self.policy
        env_data['session_settings']['n_steps_per_timestep']=self.n_steps_per_timestep
        self.env_data=env_data
    def get_state_fromlogfile(self, logfile=None):
        if str(logfile)=='None':
            logfile=self.logfn
        with open(logfile, 'r') as f:
            data = f.read()
        envdata=json.loads(data.split('\n')[0])
        return envdata
    def load_model(self, model_path):
        self.model=self.model_blank.load(model_path)
    def set_model_environment(self):
        self.model.set_env(self.env)
    def no_training_model_run(self):
        obs=self.env.reset()[0]
        done=False
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)


  

        
        





        
