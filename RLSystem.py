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
plotly.io.json.config.default_engine = 'orjson'
websocket.enableTrace(False)
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


class SFSystemCommunicator(gym.Env):
    def __init__(self, out_dict=out_dict, out_order=out_order,n_input_channels=8, channels_of_interest_inds=list(range(8)), n_timepoints_per_sample=100, max_sfsystem_output=1023,reward_formula_string='(fbin_1_4_ch0+freq_30_ch0)/fbin_12_30_ch0', 
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
                 log_actions_on_hold=True):
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
        self.n_timepoints_per_sample=n_timepoints_per_sample
        self.n_input_channels=n_input_channels
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

        self.channels_of_interest_inds=channels_of_interest_inds
        self.channels_of_interest_inds.sort()
        self.n_channels_of_interest=len(self.channels_of_interest_inds)

        self.set_fft_params()

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
        
        self.init_action_space()
        self.init_observation_space()
        self.set_value_dict_for_reward_function()
        self.connect()
        print(self.connection_status)
        self.set_delay_and_data_transfer_buffer_size()
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
    def set_value_dict_for_reward_function(self):
        ftokens=re.split(r'[+/)(*]+',self.reward_formula_string)
        self.rewarddict={}
        self.tokendict={}
        for token in ftokens:
            if 'ch' in token:
                self.rewarddict[token]=None
                self.tokendict[token]={}
                subtokens=token.split('_')
                self.tokendict[token]['datatype']=subtokens[0]
                for subtoken in subtokens:
                    if 'ch' in subtoken:
                        self.tokendict[token]['channelindex']=int(subtoken.split('h')[1])
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
        self.observation_space['raw_data']=spaces.Box(low=0, high=self.max_sfsystem_output, shape=(self.n_timepoints_per_sample, self.n_channels_of_interest), dtype=int) #n timepoints per sample rows, n input channels columns, signals should be normalized
        if self.record_fft:
            self.observation_space['fft']=spaces.Box(low=0.0, high=1.0, shape=(self.n_channels_of_interest, self.n_fft_values))
        if self.record_fbins:
            self.observation_space['fbins']=spaces.Box(low=0.0, high=1.0, shape=(self.n_channels_of_interest, self.n_fbins))
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
        self.ws=websocket.WebSocket()
        self.ws.connect(self.device_address)
        self.connection_status=self.ws.recv()
    def set_pos_encoder_mode(self):
        self.ws.send('use_only_pos_enc_mode')
        msg=self.ws.recv()
        print(msg)
    def set_use_directional_enc_mode(self):
        self.ws.send('use_directional_enc_mode')
        msg=self.ws.recv()
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
            magnitude = np.abs(xmp[mask]).mean() #here can be other functions
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
        self.ws.send("set_delay_and_data_transfer_buffer_size")
        time.sleep(self.timesleep_period)
        setup=False
        while setup==False:
            try:
                device_msg=self.ws.recv()
                if device_msg == "Awaiting delay and data transfer buffer size in shape with space separator":
                    self.ws.send(f'{self.delay},{self.n_timepoints_per_sample}')
                time.sleep(self.timesleep_period)
                device_msg=self.ws.recv()
                if device_msg == "Delay and data transfer buffer size set up":
                    setup = True 
                    break;
            except:
                pass
    def start_data_transfer_from_device(self):
        self.ws.send("start_data_transfer_from_ads")
    def stop_data_transfer_from_device(self):
        self.ws.send("stop_data_transfer_from_ads")
    def stop_audiovis_feedback(self):
        self.ws.send("stop_led_cycle")
    
    def update_audiovis_feedback(self, update_dict=None, print_msg=True):
        self.ws.send("receive_output_control_data")
        outmsg_vals=[]
        for controlnm in self.out_order:
            outmsg_vals.append(update_dict[controlnm][0])
        if print_msg==True:
            print(outmsg_vals)
        outmsg_vals=list(map(int, outmsg_vals))
        self.current_control_msg=','.join(list(map(str,outmsg_vals)))
        if print_msg==True:
            print(self.current_control_msg)
        self.ws.send(self.current_control_msg)

    def synth_data(self, signal_freq=30):
        y=1*np.sin(2*np.pi*signal_freq*self.timesteps)
        return y
    def sample_observations(self, use_synth_data=False): #True for testing of fft etc., False - for actual application
        self.ws.send("start_data_transfer_from_ads")
        self.current_sample=json.loads(self.ws.recv())
        self.raw_data=[]
        for key, value in self.current_sample.items():
            if use_synth_data==False:
                if key not in ['enc_is_clicked', "enc_is_holded"]:
                    self.raw_data.append(value)
                else:
                    if key=='enc_is_clicked':
                        self.enc_is_clicked=value[0]
                    if key=='enc_is_holded':
                        self.enc_is_holded=value[0]
            else:
                self.raw_data.append(self.synth_data())
        self.raw_data=np.array(self.raw_data).transpose()
        self.raw_data=self.raw_data[:,self.channels_of_interest_inds]
        self.ws.send("stop_data_transfer_from_ads")
    def sample_and_process_observations_from_device(self):
        new_observations=dict()
        self.sample_observations()
        new_observations['raw_data']=self.raw_data
        if self.do_fft:
         self.fft=self.get_fft_allchannels(raw_data=self.raw_data)
         if self.record_fft:
            new_observations['fft']=self.fft
        if self.do_fbins:
            self.fbins_data=self.get_bin_values_allchannels(fft=self.fft)
            if self.record_fbins:
                new_observations['fbins']=self.fbins_data 
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
        if self.ws.sock is not None:
            self.done=False
            self.best_overall_reward_now=False
            self.best_episode_reward_now=False
            self.best_total_episode_reward_now=False


            self.update_audiovis_feedback(update_dict=action)
            self.current_actions=action
            time.sleep(self.step_stim_length)
            
            new_observations=self.sample_and_process_observations_from_device()
            self.cur_observations=new_observations
            reward=self.get_reward(observations=new_observations, toreturn=True)
            reward_val=reward.tolist()
            self.reward_cur=reward
            self.total_cur_episode_reward+=reward_val
            if reward_val>=self.episode_max_reward:
                    self.episode_max_reward=reward_val
                    self.best_episode_reward_now=True
                    self.best_action_episode=action
            if reward_val>=self.overall_max_reward:
                    #print('setting best overall reward')
                    self.overall_max_reward=reward_val
                    self.best_overall_reward_now=True
                    self.best_action_overall=action
            if self.total_cur_episode_reward>=self.total_episode_max_reward:
                self.total_episode_max_reward=self.total_cur_episode_reward
                self.best_total_episode_reward_now=True

            if self.collect_data_toplot:
                self.cur_episode_rewards.append(reward_val)
            if self.log_steps:
                self.write_tolog(json.dumps({'Episode':self.current_episode, 'Step': self.cur_step, 'Step reward': reward_val}))
            if self.log_actions_every_step:
                self.write_tolog(json.dumps({'Action reward':reward_val}))
                actionstring=self.get_json_string_from_ordered_dict(action)
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
            return new_observations, reward, self.done, {} #False
        else:
            print('No connection')
            return
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
        return new_observations, {}

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
                        chfft=self.cur_observations['fft'][chidx]
                        signal_fig.add_trace(sp.go.Scatter(x=self.f_plot, y=chfft, mode='lines+markers', name=f'Channel {orig_chidx} spectrum', line=dict(color=color)), row=chidx+1, col=1)
                if 'current_fbins' in elems and self.record_fbins:
                    for chidx in range(self.n_channels_of_interest):
                        color=colors[chidx]
                        orig_chidx=self.channels_of_interest_inds[chidx]
                        chbins=self.cur_observations['fbins'][chidx]
                        signal_fig.add_trace(sp.go.Bar(x=self.fbin_axis_labels, y=chbins, name=f'Channel {orig_chidx} frequency bins', marker=dict(color=color)), row=chidx+1, col=2)
                if 'current_raw' in elems and self.record_raw:
                    for chidx in range(self.n_channels_of_interest):
                        color=colors[chidx]
                        orig_chidx=self.channels_of_interest_inds[chidx]
                        chraw=self.cur_observations['raw_data'][:,chidx]
                        signal_fig.add_trace(sp.go.Scatter(x=list(range(len(chraw))), y=chraw, mode='lines+markers', name=f'Channel {orig_chidx} raw signal', line=dict(color=color)), row=chidx+1, col=3)                
                
                
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
        if self.log_best_actions_final:
            if str(self.best_action_overall) != 'None':
                actionstring=self.get_json_string_from_ordered_dict(self.best_action_overall)
                self.write_tolog(json.dumps({'Best action across episodes reward':self.overall_max_reward}))
                self.write_tolog(actionstring)
        self.stop_audiovis_feedback() #just in case
        self.stop_data_transfer_from_device() #just in case
        self.cur_step=0 #just in case
        self.current_episode=0

        self.clear_reward_buffers()
        self.clear_reward_stats()
        if clear_log:
            self.clear_log()
        if self.ws.sock is not None:
            self.ws.close()
        
               


class FlattenActionSpaceWrapper(gym.Wrapper):
    def __init__(self, env, min_value=-1.0, max_value=1.0):
        super().__init__(env)
        # Flatten the action space
        self.action_space = self._flatten_action_space(env.action_space, min_value, max_value)
        # Store the normalization values
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
    def __init__(self, initialized_environment, algorithm='A2C', policy='MlpPolicy', logfn='model_stats.log', n_steps_per_timestep=1):
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

    def direct_feedback_run(self, reward_mapping_min, reward_mapping_max, overlay_random, mapped_outputs, min_space_value=-1, max_space_value=1):
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
        self.orig_env.step(self.orig_env.default_actions)
    def dynamic_launch(self):
        self.env.step(self.env.action_space.sample())
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

    def train(self, num_episodes=5, log_model=True, get_plots=False, render_plots=False,n_total_timesteps=1, log_or_plot_every_n_timesteps=1, jnb=False,  pause_on_click=False):
        self.n_total_timesteps=n_total_timesteps
        self.num_episodes=num_episodes
        env_paused=False
        if self.env.ws.sock is not None:
            self.training_completed=False
            #if n_total_timesteps=='episode':
            #    n_total_timesteps=int(self.env.n_steps_per_episode/self.n_steps_per_timestep) #we run one episode + 1 step before resetting, episode 
            for i in range(num_episodes):
                self.cur_episode_no=i
                self.cur_n_timesteps=0
                while self.cur_n_timesteps<int(n_total_timesteps): #here-for A2C
                    if self.env.ws.sock is not None:
                        if env_paused==False:
                            self.model.learn(total_timesteps=log_or_plot_every_n_timesteps)
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
            return     
        else:
            print('No connection.')
            return  
    def collect_environment_data(self):
        env_data=dict()
        env_data['out_dict']=self.env.out_dict
        env_data['session_settings']=dict()
        env_data['session_settings']['n_input_channels']=self.env.n_input_channels
        env_data['session_settings']['channels_of_interest_inds']=self.env.channels_of_interest_inds
        env_data['session_settings']['n_timepoints_per_sample']=self.env.n_timepoints_per_sample
        env_data['session_settings']['max_sfsystem_output']=self.env.max_sfsystem_output
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


  

        
        





        
