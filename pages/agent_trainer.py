import dash
import dash_bootstrap_components as dbc
from RLSystem import SFSystemCommunicator, stable_baselines_model_trainer
from dash_extensions.enrich import html, dcc, Input, Output, State, ctx
from dash import callback
import threading
import webcolors
import random
import os
import json
from dash.exceptions import PreventUpdate
import shutil
from datetime import datetime
import time
import pandas as pd
import  numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pickle



pio.templates.default = 'simple_white'
reward_model_data=pd.read_csv('./reward_models/model_inputdescrs.txt', sep='\t')
reward_model_data_disp=[]
reward_models=[]
for row in reward_model_data.index.tolist():
    cdt=reward_model_data.iloc[row,:]
    string=f'Model: {cdt["model"]}'
    reward_models.append(cdt["model"])
    reward_model_data_disp.append(string)
    reward_model_data_disp.append(html.Br())
    string=f'Input: {cdt["inputdescr"]}'
    reward_model_data_disp.append(string)
    reward_model_data_disp.append(html.Br())
    print(cdt)
    reward_model_data_disp.append(html.Hr())


channel_spec={0:'np_O1',1:'np_P3',2:'np_C3',3:'np_F3',4:'np_F4',5:'np_C4',6:'np_P4',7:'np_O2',
              8:'sf_ch1',9:'sf_ch2',10:'sf_ch3',11:'sf_ch4',12:'sf_ch5',13:'sf_ch6',14:'sf_ch7',15:'sf_ch8',16:'sf_enc'}


dash.register_page(__name__,'/')

#invis={'display':'none'}
#vis={'display':'inline-block'}
#d_vis={'color': 'Black', 'font-size': 20}
b_vis={"padding": "1rem 1rem", "margin-top": "2rem", "margin-bottom": "1rem", 'display':'inline-block'}
b_invis={"padding": "1rem 1rem", "margin-top": "2rem", "margin-bottom": "1rem", 'display':'none'}
invis={'display':'none'}
vis={'display':'inline-block'}
d_vis={'color': 'Black', 'font-size': 20}

offcanvas_session_lib = html.Div(
    [
        dbc.Button(
            "Show saved session list",
            id="open_session_lib",
            n_clicks=0,
        ),
        dbc.Offcanvas(
            html.P(""),
            id="offcanvas_session_lib",
            scrollable=True,
            title="Session library",
            is_open=False,
        ),
    ]
)

offcanvas_channel_specification = html.Div(
    [
        dbc.Button(
            "Show channel index map",
            id="open_channel_map",
            n_clicks=0,
        ),
        dbc.Offcanvas(
            #html.P(""),
            children=[
                '0 - np_O1',
                html.Br(),
                '1 - np_P3',
                html.Br(),
                '2 - np_C3',
                html.Br(),
                '3 - np_F3',
                html.Br(),
                '4 - np_F4',
                html.Br(),
                '5 - np_C4',
                html.Br(),
                '6 - np_P4',
                html.Br(),
                '7 - np_O2',
                html.Br(),
                '8 - sf_ch1',
                html.Br(),
                '9 - sf_ch2',
                html.Br(),
                '10 - sf_ch3',
                html.Br(),
                '11 - sf_ch4',
                html.Br(),
                '12 - sf_ch5',
                html.Br(),
                '13 - sf_ch6',
                html.Br(),
                '14 - sf_ch7',
                html.Br(),
                '15 - sf_ch8',
                html.Br(),
                '16 - sf_enc'],
            id="offcanvas_channel_map",
            scrollable=True,
            title="Channel map",
            is_open=False,
        ),
    ]
)

def clear_wavs():
    cfiles=os.listdir('./')
    cwd=os.getcwd()+'/'
    for f in cfiles:
        if f.endswith('.wav'):
            os.remove(cwd+f)
def clear_unclicked_wavs():
    cfiles=os.listdir('./')
    cwd=os.getcwd()+'/'
    for f in cfiles:
        if f.endswith('_0.wav'):
            os.remove(cwd+f)

def clear_edfs():
    cfiles=os.listdir('./')
    cwd=os.getcwd()+'/'
    for f in cfiles:
        if f.endswith('_edf'):
            shutil.rmtree(f)

def get_dir_tree(dirloc):
    results=[]
    for dirpath, dirnames, filenames in os.walk(dirloc):
        results.append(f"{dirpath.split('/')[-1]}:")
        results.append(html.Hr())
        for filename in filenames:
            results.append('  '+filename)
        results.append(html.Br())
    return results[2:]
def copy_directory(src_dir, dest_dir):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Copy the directory and all its contents
    shutil.copytree(src_dir, dest_dir,  dirs_exist_ok = True)

signal_log_exploration_controls=dbc.Container(id='signal_log_exploration_controls', children=[



    dbc.Row(children=[html.Div(children=[
    html.Datalist(
            id='logpoints',
            children=[],
        ),
    dcc.Store(id='log_session_data',data=None),
    html.Div(id='msg_logd', children=[]),
    html.Br(),
    dcc.RangeSlider(id='log_range_slider',min=0, max=1, step=1, marks=None, value=[0, 1], tooltip={"placement": "bottom", "always_visible": True, "template": "Timestep {value}"}),
    html.Br(),
    'Interval for history plot calculations: ',
    dcc.Input(type='number', placeholder='1,2,3 etc.', value=500, id='log_intermedcalc_interval'),
    html.Br(),
    'Plot channels: ',
    dcc.Dropdown(options=list(channel_spec.values()), value=list(channel_spec.values()), id='logchs', multi=True), 
    html.Br(),
    'Fft frequency bindings:',
    html.Br(),
    dcc.RangeSlider(id='fft_range_slider',min=0, max=50, step=1, marks=None, value=[0, 1], tooltip={"placement": "bottom", "always_visible": True, "template": "{value} Hz"}),
    html.Br(),
    'Datapoint id for display: ',
    html.Br(),
    dcc.Input(type='text', placeholder='datapoint_id', value='data point', id='query_datapoint', list='logpoints'),
    html.Br(),
    html.Button("Display data point info", id="datapoint_display_btn", style=b_vis, n_clicks=0),
    html.Br(),
    dcc.Textarea(
    id='point_info',
    value=' ',
    style={'width': '100%', 'height': 150, 'font-size': '14px', 'line-height': '1'},
    maxLength=2024, # Adjust the maxLength as needed
    rows=32, # Adjust the number of rows as needed
    cols=128, # Adjust the number of columns as needed
    ),
    html.Br(),
    'New reward formula: ',
    dcc.Input(type='text', placeholder='Formula string', value='raw_ch16', id='new_formula_string', size='50'),
    html.Br(),
    'Frequency bins to plot: ',
    dcc.Input(type='text', placeholder='Bin values, Hz', value='1,4;4,8;8,14;14,35;35,50', id='fbins_toplot'),  
    html.Br(),
    html.Button("Update plots", id="replot_data_btn", style=b_vis, n_clicks=0),],style={'width':'100%'})], justify="start"),
   
   
   
    dbc.Row(
    children=[
    html.Div(id='logplotcontainer', children=[], style={'width':'100%'})])], style=invis, fluid=True)


layout=html.Div(
    [dcc.Store(id='run_type', data=None),
     dcc.Store(id='session_library', data=get_dir_tree('./session_lib')),
     #dbc.Row(justify="start", id='message_row', children=[]),
                 dbc.Row(justify="start", children=[dbc.Col(width=4, children=[
    dcc.Store(id='settings_dictionary',data=None),
    dcc.Interval(id='training_status_update', disabled=True, n_intervals=0, max_intervals=-1),
    dcc.Interval(id='timer_interval', disabled=True, n_intervals=0, max_intervals=-1),
    dcc.Interval(id='data_cleaning_interval', disabled=True, n_intervals=0, max_intervals=-1),
    dbc.Row(justify="start", children=[dcc.Markdown("##### Audiovisual space setup"),
                      html.Hr(),
                      dbc.Col(width='auto',children=[ 

            dcc.RadioItems(options=['Set fully blank defaults','Set no lighting defaults','Set specific defaults'], value='Set specific defaults', id='defaults_mode'),     
            html.Br(),   
            dcc.Markdown("Lighting setup"),
            html.Div('Flash frequency, Hz'),
            html.Br(),
            html.Div('Lower bound'),
            dcc.Input(type='number', value=10, placeholder='Min frequency, Hz', id='flash_frequency_lb'),
            html.Div('Upper bound'),
            dcc.Input(type='number', value=1000, placeholder='Max frequency, Hz', id='flash_frequency_ub'), 
            html.Div('Initial value'),
            dcc.Input(type='number', value=100, placeholder='Initial frequency, Hz', id='flash_frequency_iv')]),

            
            dbc.Col(width='auto',children=['Individual LED controls',
                              html.Br(),
                              'RGB intensity range',
                              dcc.RangeSlider(id='rgb_value_range',min=10, max=255, step=5, marks=None, value=[10, 255], tooltip={"placement": "bottom", "always_visible": True}),
                              'Initial LED RGBs:',
                              dbc.Row(justify='start',children=[
                              dbc.Col(width='auto',
                                  children=[html.Div(['LED 1: ', 
                                            dcc.Input(type='text', placeholder='R,G,B', value='255,0,0', id='l1c')]),

                                            html.Div(['LED 2: ', 
                                            dcc.Input(type='text', placeholder='R,G,B', value='0,255,0', id='l2c')]),

                                            html.Div(['LED 3: ', 
                                            dcc.Input(type='text', placeholder='R,G,B', value='0,0,255', id='l3c')]),

                                            html.Div(['LED 4: ', 
                                            dcc.Input(type='text', placeholder='R,G,B', value='255,255,255', id='l4c'),])]


                              ),
                              dbc.Col(
                                  children=[html.Div(['LED 5: ', 
                                            dcc.Input(type='text', placeholder='R,G,B', value='0,255,255', id='l5c')]),

                                            html.Div(['LED 6: ', 
                                            dcc.Input(type='text', placeholder='R,G,B', value='255,0,255', id='l6c')]),

                                            html.Div(['LED 7: ', 
                                            dcc.Input(type='text', placeholder='R,G,B', value='255,255,0',id='l7c')]),

                                            html.Div(['LED 8: ', 
                                            dcc.Input(type='text', placeholder='R,G,B', value='255,0,0',id='l8c'),])]
                              ),
                            ] )  
                    ],)        
                    ]),
        html.Br(),
        dbc.Row(justify="start", children=[dcc.Markdown("##### Sound space setup"),
                      html.Hr(),
                      dbc.Col(width='auto', children=[
                          'Sound wave frequency range',
                          dcc.RangeSlider(id='sound_wave_frange',min=1, max=50000, step=100, marks=None, value=[1, 30000], tooltip={"placement": "bottom", "always_visible": True}),
                        html.Div(['Wave 1 initial frequency: ', 
                                            dcc.Input(type='number', placeholder='Frequency, Hz', value=440, id='wave_1_freq')]),
                        html.Div(['Wave 2 initial frequency: ', 
                                            dcc.Input(type='number', placeholder='Frequency, Hz', value=440, id='wave_2_freq')]),
                        'Sound wave shapes',
                        #dcc.Checklist(options=['Noise','Sine', 'Square', 'Triangle'], value=['Noise','Sine', 'Square', 'Triangle'], id='sound_wave_shapes'),
                        html.Div(['Wave 1 initial shape: ', 
                                            dcc.RadioItems(['Noise','Sine', 'Square', 'Triangle'], 'Noise', id='w1sh')]),
                        html.Div(['Wave 2 initial shape: ', 
                                            dcc.RadioItems(['Noise','Sine', 'Square', 'Triangle'], 'Noise', id='w2sh')]),
                        html.Hr(),
                        'Volume range',
                        dcc.RangeSlider(min=0, max=50, step=1, marks=None, value=[1, 25], tooltip={"placement": "bottom", "always_visible": True},id='volume_range'),
                        'Initial volume: ',
                        dcc.Input(type='number', placeholder='Volume level', value=5, id='maxivolume')                                            
                                            
                                            
                                            ]
                        ),
                      dbc.Col(width='auto', children=[
                          'Panner and phasor frequency ranges',
                          dcc.RangeSlider(id='panner_phasor_frange',min=1, max=100, step=1, marks=None, value=[1, 50], tooltip={"placement": "bottom", "always_visible": True}),
                        html.Div(['Panner initial frequency: ', 
                                            dcc.Input(type='number', placeholder='Frequency, Hz', value=1, id='panner_freq')]),

                        'Panner denominator range',
                        dcc.RangeSlider(id='panner_div_range',min=1, max=5, step=1, marks=None, value=[1, 5], tooltip={"placement": "bottom", "always_visible": True}),
                        html.Div(['Panner initial denominator: ', 
                                            dcc.Input(type='number', placeholder='Panner denominator', value=2, id='panner_div')]),
                        html.Br(),
                        html.Div(['Phasor 1 initial frequency: ', 
                                            dcc.Input(type='number', placeholder='Frequency, Hz', value=440, id='phasor_1_freq')]),
                        'Phasor 1 initial frequency span',
                        dcc.RangeSlider(min=1, max=30, step=1, marks=None, value=[1, 50], tooltip={"placement": "bottom", "always_visible": True}, id='phasor_1_span'),
                        html.Div(['Phasor 2 initial frequency: ', 
                                            dcc.Input(type='number', placeholder='Frequency, Hz', value=440, id='phasor_2_freq')]),
                        'Phasor 2 initial frequency span',
                        dcc.RangeSlider(min=1, max=30, step=1, marks=None, value=[1, 50], tooltip={"placement": "bottom", "always_visible": True},id='phasor_2_span'),                                                                           
                          ]
                        ),
                        html.Br(),
                        dcc.Markdown("#### Play audio"),
                        dcc.Dropdown(id='audio_recording_path', options=os.listdir('./suggestions'),multi=False),
                        dcc.Checklist(id='audio_options', options=['Auto Play', 'Loop'], value=['Auto Play', 'Loop']),
                        html.Audio(id='audio',src='/suggestions/test_audio.mp3', controls=True, autoPlay=True, loop=True)
                      ]),
          html.Br(),
          dbc.Row(justify='start', children=[
            dcc.Markdown('##### Session settings'),
            html.Hr(),
            dbc.Col(width='auto', children=[
            html.Div(children=[
                #'Total channel count: ',
                #dcc.Input(type='number', placeholder='N channels', value=9, id='n_input_channels'),
                #html.Br(),
                'Channels to observe: ',
                dcc.Input(type='text', placeholder='Channels of interest ch0,...chn', value='np_O1,np_P3,np_C3,np_F3,np_F4,np_C4,np_P4,np_O2,sf_enc', id='channels_of_interest'),
                html.Br(),
                'N timepoints per sample: ',
                dcc.Input(type='number', placeholder='N points', value=500, id='n_timepoints_per_sample'),
                html.Br(),               
                'Delay between datapoints (if Neuroplay is used,   will be set aautomatically to match): ',
                dcc.Input(type='number', placeholder='Delay, ms', value=10, id='delay'), 
                html.Br(), 
               # 'Max ADS output: ',
               # dcc.Input(type='number', placeholder='Value', value=1023, id='max_sfsystem_output'),
               # html.Br(),         
                'Data types to use in observational space: ',
                dcc.Dropdown(options=['Raw signal values','Frequency spectra', 'Frequency bin values'], value=['Raw signal values','Frequency spectra', 'Frequency bin values'], id='obs_space_opts', multi=True), 
                html.Br(),      
                dcc.Checklist(id='use_unfiltered_np_data',options=['Use unfiltered Neuroplay signal'],  value=[]),
                html.Br(),           
                ]
            ),

            html.Div(children=[
                html.Br(),
                dcc.Markdown('Data processing and usage settings'),  
                html.Hr(),              
                'Reward formula: ',
                dcc.Input(type='text', placeholder='Formula string', value='raw_ch16', id='formula_string', size='50'),  
                ' ',
                dbc.Button("Help", id="open_formula_instructions", n_clicks=0),    
                html.Br(),
                offcanvas_channel_specification,
                dcc.Checklist(options=['Use absolute raw values in reward calculation',
                                       'Use directionality-agnostic encoder mode'], value=['Use directionality-agnostic encoder mode'], id='data_proc_options'),
                dbc.Offcanvas(children=[
                        html.P('Reward formula can use the following operators: //, *, **, -, +'),
                        html.P('It can refer to channels using "ch" prefix followed by an index (starting with 0) e.g. ch0'),
                        html.P('It can refer to values of frequency bins in specific channels e.g. fbin_10_50_ch0'),
                        html.P('The corresponding freqency bins must be present among fbins passed at the initialization step'),
                        html.P('fbins should be passed in the form of [(b1 min, b1max),...(bn min, bn max)]'),
                        html.P('It can refer to specific frequencies from fft e.g. freq_50_ch0'),
                        html.P('It can refer to total sum or raw signal values in a sample raw_ch8'),
                        html.P('It can refer to root mean square amplitude e.g. rmsamp_ch8'),
                        html.P('It can refer to peak-to-peak amplitude e.g. ptp_ch8'),
                        html.P('It can refer to max amplitude e.g. maxamp_ch8'),
                        html.P('Use integers for frequencies, fractions are not supported for now'),
                        html.P('Some examples:'),
                        html.P('1. (freq_50_ch0+fbin_0_10_ch0)/(fbin_20_30_ch0)'),
                        html.P('2. freq_5_ch0/freq_10_ch0'),
                        html.P('3. fbin_05_5_ch0'),
                        html.P('4. raw_ch0'),
                        html.P('ch8 is the encoder channel'),
                ],
                  id='formula_instructions',
                  title='Instructions',
                  is_open=False,
                  placement="start",
                  scrollable=True,
                 # style={'width':'95%'},
                ),


                html.Br(),
                'Observational space data types: ',
                dcc.Checklist(options=['Raw signal values','Frequency spectra', 'Frequency bin values'], value=['Raw signal values','Frequency spectra', 'Frequency bin values'], id='observational_space_datatypes'),
                html.Br(),
                'Frequency bins to record: ',
                dcc.Input(type='text', placeholder='Bin values, Hz', value='1,4;4,8;8,14;14,35;35,50', id='fbins'),   
                html.Br(),
                'Device address: ',
                dcc.Input(type='text', placeholder='Bin values, Hz', value='ws://10.42.0.231:80/', id='device_address'),  
                
            ]),
            html.Br(),
            ]),

            dbc.Col(width='auto', children=[
                dcc.Markdown('Session timings'),  
                html.Hr(),
                'Step length, ms (approximate): ',
                dcc.Input(type='number', placeholder='Length, ms', value=10000, id='step_stim_length_millis'),  
                html.Br(),
                'Episode time (approximate): ',
                dcc.Input(type='number', placeholder='Length, s', value=60, id='episode_time_seconds'), 
                html.Br(),
                html.Br(),
                dcc.Markdown('RL training settings'),
                html.Hr(),
                'Rl algorithm: ',
                dcc.Dropdown(id='algorithm',
                options=['PPO','SAC','DDPG','TD3','A2C','DQN'],
                value='A2C',
                multi=False), 
                'N steps per timestep (for PPO, A2C): ',
                dcc.Input(type='number', placeholder='N steps', value=0, id='n_steps_per_timestep'), 
                html.Br(),
                '(to use the step count corresponding to the episode time set 0)',
                html.Br(),
                'N timesteps per algorithm training episode: ',
                dcc.Input(type='number', placeholder='N timesteps', value=1, id='n_total_timesteps'), 
                html.Br(),
                'N episodes: ',
                dcc.Input(type='number', placeholder='N episodes', value=5, id='num_episodes'), 
                html.Br(),
                'Reset stage length: ',
                dcc.Input(type='number', placeholder='Length, s', value=10, id='stim_length_on_reset'), 
                html.Br(),
                dcc.Checklist(options=['Start training using Neuroplay reward signal quality threshold'], value=['Start training using Neuroplay reward signal quality threshold'], id='start_on_reward_sig_qual'),
                dcc.Checklist(options=['Pause training when Neuroplay reward signal quality below threshold'], value=['Pause training when Neuroplay reward signal quality below threshold'], id='pause_learning_if_reward_sig_qual_false'),
                dcc.Checklist(options=['Start training on encoder click'], value=[], id='start_on_click'),
                html.Br(),
                'Reward channel quality threshold for Neuroplay: ',
                dcc.Input(type='number', placeholder='Quality threshold, %', value=90, id='reward_np_sigqual_thresh'),                
            ]),
            html.Br(),
            html.Br(),
            dbc.Row(justify='start', children=[
                dbc.Col(width='auto',children=[
                dcc.Markdown('##### Logging and visualization'),
                html.Hr(),
                'General logging file name: ',
                dcc.Input(type='text', placeholder='filename.log', value='current_training.log', id='logfn'), 
                html.Br(),
                'Logging types: ',
                dcc.Dropdown(options=['Step data','Episode data', 'Best episode actions', 'Each step actions','Final best actions', 'Raw data','FFT results','Bin values','Models'], value=['Step data','Episode data', 'Best episode actions', 'Each step actions','Final best actions', 'Raw data','FFT results','Bin values','Models'], id='logging_plotting_opts', multi=True), 
                html.Br(),
                'Model logging interval in algorithm timesteps: ',
                dcc.Input(type='text', placeholder='Interval, timesteps', value=1, id='log_or_plot_every_n_timesteps'), 
                html.Br(),
                dcc.Checklist(options=['Render session plots'], value=['Render session plots'], id='render_data'),
                html.Br(),                
                'Training plot width: ',
                dcc.Slider(id='training_plot_width',min=500, max=5000, step=100, value=2000, marks=None, tooltip={"placement": "bottom", "always_visible": True, "template": "{value} px"}),
                'Training plot height: ',
                dcc.Slider(id='training_plot_height',min=50, max=5000, step=50, value=500, marks=None, tooltip={"placement": "bottom", "always_visible": True, "template": "{value} px"}),                
                'Signal plot width: ',
                dcc.Slider(id='signal_plot_width',min=500, max=5000, step=100, value=2000, marks=None, tooltip={"placement": "bottom", "always_visible": True, "template": "{value} px"}),  
                'Signal plot height: ',
                dcc.Slider(id='signal_plot_height',min=50, max=5000, step=50, value=1500, marks=None, tooltip={"placement": "bottom", "always_visible": True, "template": "{value} px"}),                  
                html.Br(),
                'Neuroplay EDF logging',
                html.Hr(),
                dcc.Checklist(options=['Write EDF log'], value=['Write EDF log'], id='write_edf_ann'),
                ' ',
                dcc.Checklist(options=['Annotate steps in EDF'], value=['Annotate steps in EDF'], id='edf_step_annotation'),
                ' ',
                dcc.Checklist(options=['Annotate reward above threshold in EDF'], value=['Annotate reward above threshold in EDF'], id='edf_rf_annotation'),
                ' ',
                'Reward threshold: ',
                dcc.Input(type='number', placeholder='Reward value', value=1, id='edf_rf_annotation_threshold'), 
                html.Hr(),
                html.Br(),                          
               
               
                ])             
            ]),
            html.Br(),
            html.Div(children=['Session data update minimal interval: ',
            dcc.Input(type='number', placeholder='interval, ms', value=1000, id='info_upd_interval'),
            html.Br(),
            'Step count for a dynamic notrain session, trained model run, or direct feedback: ',
            dcc.Input(type='number', placeholder='N steps', value=360, id='n_steps_notrain'),
            html.Br(),
            'Model location for upload (for the corresponding regimen): ',
            dcc.Input(type='text', placeholder='session name/model name', value='default_session/best_total_episode_reward_model.zip', id='model_name'),]), 
            offcanvas_session_lib,
            html.Div(children=[
            html.Button("Run training session", id="start_session_train", style=b_vis, n_clicks=0),
            ' ',
            html.Button("Run dynamic session without training", id="start_session_notrain", style=b_vis, n_clicks=0),
            ' ',
            html.Button("Run static session", id="start_session_static", style=b_vis, n_clicks=0),
            ' ',
            html.Button("Stop session", id="stop_session_train", style=b_invis, n_clicks=0),
            ' ',
            html.Button("Run additional episodes", id="additional_session", style=b_invis, n_clicks=0),
            ' ',
            html.Button("Run trained model", id="run_trained", style=b_vis, n_clicks=0),
            html.Br(),
            'Additional options for running the trained model: ',
            dcc.Checklist(options=['Train the logged model with the original settings'], value=[], id='train_logged_orig'),
            dcc.Checklist(options=['Train the logged model with the new settings'], value=[], id='train_logged_new'),
            html.Br(),
            'Action log location to launch: ',
            dcc.Input(type='text', placeholder='session name/model name/action_no.log', value='default_session/action_0.log', id='action_log_name'),
            html.Br(),
            html.Button("Trigger logged action", id="run_action", style=b_vis, n_clicks=0),            
               ]),
             ])
]),
dbc.Offcanvas(children=[html.Br(),
    dbc.Row(justify="start", id='message_row', children=[]),
    html.Div(id='training_figure_container', children=[]),
    html.Br(),
    html.Div(id='signal_figure_container', children=[]),
    html.Br(),
    html.Div(signal_log_exploration_controls, style={'width':'100%'}),],
    id='plot_panel',
    title='Session data',
    is_open=False,
    placement="end",
    scrollable=True,
    style={'width':'95%'},
),




dbc.Offcanvas(children=[html.Br(),
    dbc.Row(justify="start", id='message_row', children=[]),
    html.Div(id='reward_model_list', children=reward_model_data_disp),
    html.Br(),],
    id='r_model_info',
    title='Available reward models',
    is_open=False,
    placement="end",
    scrollable=True,
    style={'width':'95%'},
),

dbc.Col(children=[
                dcc.Markdown("### ML model usage"),
                dcc.Checklist(options=['Use ML model-based reward'], value=['Use ML model-based reward'], id='use_reward_model'),
                dcc.Dropdown(id='model prefix', options=reward_models, value='CLASSIFIER_LogisticRegression_l1_haagladen_20_slowwaves_rawframes_min_100_max_350_wi_150_wa_150_ns_500_tstroc_0.73178616_thresh_0.70.joblib', multi=False, style={'width':'50%'}),
                
                #dcc.Input(type='text', placeholder='model prefix', value='CLASSIFIER_LogisticRegression_l1_haagladen_20_slowwaves_rawframes_min_100_max_350_wi_150_wa_150_ns_500_tstroc_0.73178616_thresh_0.70', id='r_model_path', size= '150'),
                html.Br(),
                dcc.Input(type='text', placeholder='model input descriptor', value='CHANNELS_np_C3,np_P3_SF_100_WIN_150_DATA_SCALO_scalomagn_lf1_uf40_nscales32_ww1', id='r_model_inputdescr', size='150'),
                dcc.Dropdown(id='r_model_predtype',
                options=['optithresh', 'customthresh', 'defaultthresh', 'proba'],
                value='optithresh',
                multi=False,
                style={'width': '50%'}), 
                dcc.Dropdown(id='r_model_voting',
                options=['max'],
                value='max',
                multi=False,
                style={'width': '50%'}), 
                dcc.Input(type='number', placeholder='custom decision threshol (for customthresh regimen)', value=0.5, id='r_model_customthresh', size='150'),
                html.Br(),
                html.Button("Show reward model stats", id="show_r_model_stats", style=b_vis, n_clicks=0),
                html.Br(),
                dbc.Button(
                    "Show reward model list",
                    id="show_r_model_list",
                    n_clicks=0,
                ),
                html.Br(),
                html.Br(),
                html.Br(),
                dcc.Textarea(
                    id='r_model_data',
                    value='Reward model stats',
                    style={'width': '30%', 'height': 300, 'font-size': '14px', 'line-height': '1'},
                    maxLength=1024, # Adjust the maxLength as needed
                    rows=64, # Adjust the number of rows as needed
                    cols=128, # Adjust the number of columns as needed
                ),

                html.Hr(),
                html.Br(),


                 dcc.Markdown("### Session Data"),
                 html.Div(id='plot_style_container', children=[
                'CSS color names for the signal plots: ',
                dcc.Input(type='text', placeholder='CSS color name', value='black', id='signal_plot_color'), 
                html.Br(),
                'Use "random,seed" for a random color sample',
                html.Br(),
                html.Button("Resample colors", id="color_resample", style=b_vis, n_clicks=0),
                ' ',
                html.Button("Clear logs", id="clear_logfiles", style=b_vis, n_clicks=0),
                ' ',
                html.Button("Store session data", id="move_data", style=b_vis, n_clicks=0),
                ' ',
                html.Button("Clear session library", id="clear_session_lib", style=b_vis, n_clicks=0),
                ' ',
                html.Button("Clear trainer results to the last 10 points", id="clear_trainer_data", style=b_vis, n_clicks=0),
                html.Br(),
                'Session name: ',
                dcc.Input(type='text', placeholder='Session name (old data, if present, will be overwritten)', value='default_session', id='session_name'),]),
                html.Button("Set session name to current timestamp", id="set_session_name_to_timestamp", style=b_vis, n_clicks=0),
                ' ',
                html.Button("Load session data to explore", id="load_session_data", style=b_vis, n_clicks=0),  
                html.Br(),
                dcc.Checklist(options=['Load session data from EDF'], value=['Load session data from EDF'], id='load_from_edf'),
                html.Br(),
                html.Br(),
                dcc.Checklist(options=['Save actions on encoder hold'], value=['Save actions on encoder hold'], id='log_actions_on_hold'),
                html.Br(),
                dbc.Button("Show session data panel", id="open_plot_panel", n_clicks=0),  
                # dbc.Offcanvas(children=[html.Br(),
                #  dbc.Row(justify="start", id='message_row', children=[]),
                #   html.Div(id='training_figure_container', children=[]),
                #   html.Br(),
                #   html.Div(id='signal_figure_container', children=[]),
                #   html.Br(),
                #   html.Div(signal_log_exploration_controls, style={'width':'100%'}),],
                #   id='plot_panel',
                #   title='Session data',
                #   is_open=False,
                #   placement="end",
                #   scrollable=True,
                #   style={'width':'95%'},
                # ),
                html.Br(),
                'Clear trainer results every N seconds: ',
                dcc.Input(type='number', placeholder='Interval, s  (-1 for never)', value=-1, id='clear_trainer_data_n_seconds'), 
                html.Br(),
                html.Br(),
                html.Div(
                   children=[dcc.Markdown("#### Mic logging"),
                             html.Hr(),
                             dcc.Dropdown(multi=False, options=['Continuous logging', 'Log on click', 'Separate files logging'], value='Separate files logging', id='mic_log_opts', style={'width': '50%'}),
                             ' ',
                             html.Button("Clear unclicked .wav logs ", id="clear_unclicked_wavs", style=b_vis, n_clicks=0),                       
                             ]
                ),
                html.Br(),
                html.Div(children=[
                dcc.Markdown("#### Deterministic modes"),
                html.Hr(),
                dcc.Markdown("##### Timer"),
     
                'Timer interval or timer query interval: ',
                dcc.Input(type='number', placeholder='N minutes', value=90, id='timer_interval_mins'),
                html.Br(),   
                'Timer signal duration: ',    
                dcc.Input(type='number', placeholder='N seconds', value=60, id='timer_signal_duration_s'),  
                html.Br(),    
                'Timer reward threshold: ',    
                dcc.Input(type='number', placeholder='Reward value', value=-1, id='timer_reward_threshold'), 
                html.Br(), 

                html.Button("Run timer", id="run_timer", style=b_vis, n_clicks=0),  
                html.Button("Stop timer", id="stop_timer", style=b_invis, n_clicks=0),  
                dcc.Markdown("##### Direct feedback"),
    
                'Set reward to modulate: ',
                dcc.Dropdown(options=['Flash frequency',
                                      'LED intensity',
                                      'R',
                                      'G',
                                      'B',
                                      'Wave 1 frequency',
                                      'Wave 2 frequency',
                                      'Panner freq',
                                      'Sound volume'], value=['R'], id='deterministic_opts', multi=True), 
                'Reward interval to map: ',
                dcc.Input(type='text', placeholder='min-max', value='0-50', id='reward_mapping_interval'),  
                dcc.Checklist(options=['Overlay random signal'], value=[], id='overlay_random'),               
                html.Br(),   
                html.Button("Run direct feedback", id="run_direct_feedback", style=b_vis, n_clicks=0),
                html.Button("Stop direct feedback", id="stop_direct_feedback", style=b_invis, n_clicks=0), ],  style={"width": "50%"},),


                html.Div(children=[
                dcc.Markdown("#### OLED suggestion setup"),
                html.Hr(),
                dcc.Textarea(
                    id='oled_text',
                    value='Information to display',
                    style={'width': '100%', 'height': 300, 'font-size': '14px', 'line-height': '1'},
                    maxLength=1024, # Adjust the maxLength as needed
                    rows=64, # Adjust the number of rows as needed
                    cols=128, # Adjust the number of columns as needed
                ),
                html.Br(),
                'Text size: ',
                dcc.Input(type='number', placeholder='1,2,3 etc.', value=2, id='text_size'),  
                html.Br(),
                dcc.Checklist(options=['Send reward to display'], value=['Send reward to display'], id='send_reward_to_display'),
                dcc.Checklist(options=["Send Neuroplay signal quality to display"], value=["Send Neuroplay signal quality to display"], id="send_np_signal_to_display"),
                html.Button("Send to display", id="send_display_text", style=b_vis, n_clicks=0),
                html.Button("Clear display (if anything present)", id="clear_display", style=b_vis, n_clicks=0),                 
                dcc.Checklist(options=['Pause or restart signal on encoder click'], value=['Pause or restart signal on encoder click'], id='pause_on_click'),
                dcc.Markdown("##### Launch action from json string"),
                html.Hr(),
                dcc.Textarea(
                    id='action_text',
                    value='{"leddelay": [100.0], "lv1r": [190.03234773874283], "lv1g": [168.18682730197906], "lv1b": [255.0], "lv2r": [196.86115473508835], "lv2g": [159.4429063796997], "lv2b": [185.24352610111237], "lv3r": [149.18937146663666], "lv3g": [151.4224249124527], "lv3b": [10.0], "lv4r": [77.70063683390617], "lv4g": [255.0], "lv4b": [255.0], "lv5r": [66.86547353863716], "lv5g": [255.0], "lv5b": [17.525383979082108], "lv6r": [10.0], "lv6g": [237.40646064281464], "lv6b": [110.9641245007515], "lv7r": [213.2022413611412], "lv7g": [255.0], "lv7b": [145.3699791431427], "lv8r": [10.0], "lv8g": [255.0], "lv8b": [182.14710593223572], "wave_1_freq": [14318.983242064714], "wave_2_freq": [6777.882110029459], "panner_freq": [32.50046396255493], "phasor_1_freq": [8.703051596879959], "phasor_2_freq": [9.96783521771431], "phasor_1_min": [1.0], "phasor_2_min": [1.0], "phasor_1_dif": [5.6508409678936005], "phasor_2_dif": [27.42316609621048], "panner_div": [3.862565517425537], "wave_1_type": [3.0], "wave_2_type": [0.5501725673675537], "maxivolume": [9.150283694267273]}',
                    style={'width': '100%', 'height': 150, 'font-size': '14px', 'line-height': '1'},
                    maxLength=2024, # Adjust the maxLength as needed
                    rows=32, # Adjust the number of rows as needed
                    cols=128, # Adjust the number of columns as needed
                ),
                html.Br(),
                html.Button("Get current action", id="get_current_action", style=b_vis, n_clicks=0),
                html.Button("Launch action", id="action_from_string", style=b_vis, n_clicks=0),
                html.Br(),
                html.Button("Stop audio-visual output", id="stop_audiovis", style=b_vis, n_clicks=0),
                ],
                style={"width": "70%"},) ,
                  ])]      
          
)
          ]) 




def process_edf_mainlog(logf):
    import mne
    import pandas as pd
    print('Processing maain EDF log file')
    edfp='/home/biorp/Documents/PlatformIO/Projects/SFSystem_withAV/session_lib/default_session/default_session_edf/default_session_edf.edf'
    edf = mne.io.read_raw_edf(edfp, preload=True)

    # Extract the data from the EDF file
    data = edf.get_data()
    annotations=edf.annotations.to_data_frame()

    # Get the channel names
    ch_names = edf.ch_names

    # Create a pandas DataFrame
    df = pd.DataFrame(data.T, columns=ch_names)
    df['t']=edf.times
    annotations['t']=edf.annotations.onset
    anndescriptors=annotations['description'].unique().tolist()
    sann=False
    eann=False
    micann=False
    actann=False
    rewann=False
    acts={}
    micfs={}
    steprewards={}
    episode_total_rewards={}
    alltimestamps=df['t'].tolist()

    anndata={}

    cstep=0
    cepisode=0
    cts=0
    crev=0
    etrev=0
    cact=None
    cmiclogf=None
    onset=None
    for i in annotations.index.tolist():
        anndt=annotations.iloc[i]
        descr=anndt['description']
        #print(anndt)
        towrite=False
        #onset=anndata['t']
        if 'episode_' in descr:
            episode=descr.split('episode_')[1].split('_')[0]
            cepisode=episode
            towrite=True
        if '_step_' in descr:
            step=descr.split('_step_')[1]
            cstep=step
            onset=anndt['t'] #onsets aalign with step onset
            towrite=True
        if 'sr_' in descr:
            reward=float(descr.split('sr_')[1].split('_')[0])
            crev=reward
            steprewards[f't_{onset}_ep_{cepisode}_step{cstep}']=crev
            towrite=True
        if '_tcer_' in descr:
            episode_total_reward=float(descr.split('_tcer_')[1])
            etrev=episode_total_reward
            episode_total_rewards[f't_{onset}_ep_{cepisode}']=etrev
            towrite=True
        if 'MICLOGF' in descr:
            miclogf=descr.split('MICLOGF_')[1]
            cmiclogf=miclogf
            micfs[onset]=cmiclogf
            towrite=True
        if 'action' in descr:
            action=descr.split('action_')[1]
            cact=action
            acts[onset]=action
            towrite=True
        if onset!=None:
            if towrite==True:
                anndata[onset]={'episode':cepisode,
                                'step':cstep,
                                'miclogf':cmiclogf,
                                'action':cact,
                                'reward':crev}


    anndata_edfonsets={}
    anndata_keys=np.array(list(anndata.keys()))
    for i in alltimestamps:
        tarkeys=anndata_keys[anndata_keys<i]
        if len(tarkeys)>0:
            tarind=np.argmin(np.abs(tarkeys - i))
            closest=tarkeys[tarind]
            anndata_edfonsets[i]=anndata[closest]
        else:
            anndata_edfonsets[i]={'episode':-1,
                                'step':-1,
                                'miclogf':'None',
                                'action':'None',
                                'reward':0}

    for i in anndescriptors:
        if 'episode_' in i:
            eann==True
        if 'step_' in i:
            sann==True
        if 'MICLOGF_' in i:
            micann==True
        if 'sr_' in i:
            rewann=True



    if rewann==False:
        df['rev']=0
    else:
        rewards=[]
        for i in anndata_edfonsets:
            rewards.append(anndata_edfonsets[i]['reward'])
        df['rev']=rewards
    if sann==False:
        df['step']=0
    else:
        steps=[]
        for i in anndata_edfonsets:
            steps.append(anndata_edfonsets[i]['step'])
        df['step']=steps
    if sann==False:
        df['episode']=0
    else:
        episodes=[]
        for i in anndata_edfonsets:
            episodes.append(anndata_edfonsets[i]['episode'])
        df['episode']=episodes

    acts={}
    miclogs={}


    if actann==False:
        for i in alltimestamps:
            acts[i]='NotLogged'
    else:
        for i in alltimestamps:
            acts[i]=anndata_edfonsets[i]['action']
    if micann==False:
        for i in alltimestamps:
            miclogs[i]='NotLogged'
    else:
        for i in alltimestamps:
            miclogs[i]=anndata_edfonsets[i]['miclogf']
    df['datapoint']='ep_'+df['episode'].astype(str)+'|st_'+df['step'].astype(str)+'|'+df['t'].astype(str)+'|'+df['rev'].astype(str)

    fdf=df[['episode','step','t']]
    fdf['dtp']=0
    fdf['wsi']=0
    fdf['rev']=df['rev']
    chofi=[]
    for i in ch_names:
        fdf['np_'+i]=df[i]
        chofi.append('np_'+i)
    sf_chnames=['sf_ch1','sf_ch2','sf_ch3','sf_ch4','sf_ch5','sf_ch6','sf_ch7','sf_ch8','sf_enc']
    for i in sf_chnames:
        fdf[i]=0
    fdf['datapoint']=df['datapoint']
    print(fdf.columns.tolist())
    print(fdf.shape)
    print('EDF log processing complete')
    return {'session_settings':{},
            'delay':8, #because 125 Hz by default
            'n_timepoints_per_sample':10,
            'fbins':[[2,4]], #just stump
            'channels':chofi,
            'reward_formula_string_orig':'raw_ch0',
            'step_rewards':steprewards, 
            'episode_total_rewards':episode_total_rewards, 
            'acts':acts, 
            'miclogs':miclogs,
            'rdf':fdf}


def process_sfs_mainlog(logf):
    global channel_spec
    print('Reading the log file')
    datas=[]
    with open(logf, 'r') as f: #reading the log file
        while True:
            try:
                data = f.readline()
                data=json.loads(data)
                datas.append(data)
            except:
                break
    #print(datas)
    out_dict=datas[0]['out_dict']
    session_settings=datas[0]['session_settings']
    chofi=datas[0]['channels_of_interest']
    datas=datas[1:]
    delay=int(session_settings['delay'])
    n_timepoints_per_sample=int(session_settings['n_timepoints_per_sample'])
    channels=chofi
    reward_formula_string_orig=session_settings['reward_formula_string']
    step_rewards={}
    episode_total_rewards={}
    acts={}
    curepisode=0
    curstep=0
    rd=[]
    datapiece=0
    curtimestep=None
    crev=None
    for ind in range(len(datas)):
        i=datas[ind]
        if 'Episode' in i.keys() and 'Step' in i.keys():
            curepisode=i['Episode']
            curstep=i['Step']
            curtimestep=i['Timestamp']
        if 'Step reward' in i.keys():
            print(i)
            reward=i['Step reward']
            step_rewards[f't_{curtimestep}_ep_{curepisode}_step_{curstep}_dtp_{datapiece}']=reward
        if 'Action reward' in i.keys():
            #print(i)
            action=datas[ind+1]
            action_reward=i['Action reward']
            acts[curtimestep]=action
        if 'Episode total reward' in i.keys():
            print(i)
            #print(i)
            episode_total_rewards[f't_{curtimestep}_ep_{curepisode}_dtp_{datapiece}']=i['Episode total reward']
        # if 'fft' in i.keys():
        #     dt=i['fft']
        #     rs={'episode':curepisode, 'step':curstep}
        #     ffts.append(rs)
        if 'raw_data' in i.keys():
            dr=np.array(i['raw_data']).T
           # print(dr)
           
            rs={'episode':[curepisode for i in range(n_timepoints_per_sample)], 'step':[curstep for i in range(n_timepoints_per_sample)], 't':[curtimestep for i in range(n_timepoints_per_sample)], 'dtp':datapiece, 'wsi':list(range(n_timepoints_per_sample)), 'rev':[np.round(reward,2) for i in range(n_timepoints_per_sample)]}
            for ci in range(len(dr)):
              #  print(ci)
                rs[channel_spec[ci]]=dr[ci]
            rd.append(pd.DataFrame(rs))
        datapiece+=1
        
    rdf=pd.concat(rd)

   # print(rd)
    rdf['datapoint']='ep_'+rdf['episode'].astype(str)+'|st_'+rdf['step'].astype(str)+'|'+rdf['t']+'|'+rdf['dtp'].astype(str)+'|'+rdf['wsi'].astype(str)+'|'+rdf['rev'].astype(str)
    print(rdf.columns.tolist())
    print(rdf.shape)
    print('Logfile read done.')
    return {'session_settings':session_settings,
            'datas':datas,
            'delay':delay,
            'n_timepoints_per_sample':n_timepoints_per_sample,
            'fbins':session_settings['fbins'],
            'channels':channels,
            'reward_formula_string_orig':reward_formula_string_orig,
            'step_rewards':step_rewards, 
            'episode_total_rewards':episode_total_rewards, 
            'acts':acts, 
            'rdf':rdf}


@callback(
    #Output("open_plot_panel", "n_clicks", allow_duplicate=True),
    Output('signal_log_exploration_controls', "style", allow_duplicate=True),
    Output('training_figure_container', "children", allow_duplicate=True),
    Output('signal_figure_container',  "children",  allow_duplicate=True),
    Output('message_row', 'children', allow_duplicate=True),

    Output('log_range_slider', 'max'),
    Output('log_range_slider', 'marks'),
    Output('log_range_slider', 'value'),
    Output('log_range_slider', 'step'),
    Output('fft_range_slider', 'max'),
    Output('fft_range_slider', 'value'),
    Output('logpoints', 'children'),
    Output('log_session_data', 'data'),
    #Output('fbins_toplot','value'),
    Output('msg_logd', 'children'),


    Input("load_session_data", "n_clicks"),
    State("open_plot_panel", "n_clicks"),
    State('session_name','value'),
    State('new_formula_string', "value"),
    State('fbins_toplot', 'value'),
    State('load_from_edf', 'value'),
    prevent_initial_call=True
)
def explore_session_data_panel_formation(n1, n2, sn, nformstr, nfbins, fromedf):
    if len(fromedf)>0:
        fromedf=True
        logf='./session_lib/{sn}/{sn}_edf/{sn}_edf.edf'
        logdata=process_edf_mainlog(logf)
    else:
        fromedf=False
        logf=f'./session_lib/{sn}/current_training.log'
        logdata=process_sfs_mainlog(logf)
    timesteps=logdata['rdf']['datapoint'].tolist()
    print(len(timesteps))
    lrsmax=len(timesteps)
    dpts=logdata['rdf']['datapoint'].tolist()
    marks={}
    nts=len(timesteps)/(logdata['n_timepoints_per_sample'])
    n_marks=10
    if n_marks<nts:
        mark_every=int(np.floor(nts/n_marks))*logdata['n_timepoints_per_sample']
    else:
        mark_every=logdata['n_timepoints_per_sample']
    for i in range(0, len(timesteps), mark_every):
        marks[i]=dpts[i]
    print('marks setup')

    binlist=nfbins.split(';')
    fbins=[]
    for b in binlist:
        bin=b.split(',')
        bin=list(map(int, bin))
        bin=tuple(bin)
        fbins.append(bin)

    global log_env
    log_env=SFSystemCommunicator(offline_mode=True, 
                                input_channels=logdata['channels'], 
                                n_timepoints_per_sample=logdata['n_timepoints_per_sample'], 
                                delay=logdata['delay'],
                                reward_formula_string=nformstr,
                                fbins=fbins)
    opts=[]
    for  i in timesteps:
        opts.append(html.Option(label=i, value=i))


    logdata['rdf']=logdata['rdf'].to_json(orient='records')
    fbinstr=''
    for i in logdata['fbins']:
        fbinstr+=f'{i[0]},{i[1]};'
    fbinstr=fbinstr[:-1]
    return vis,[],[],[], lrsmax, marks,  [0, lrsmax],1, int(log_env.f_plot[-1]), [0, log_env.f_plot[-1]], opts, logdata, ['Original reward formula: '+logdata['reward_formula_string_orig'],
                                                                                                                                                            html.Br(),
                                                                                                                                                            'Originally explored fbins: '+fbinstr,
                                                                                                                                                            html.Br(),
                                                                                                                                                            'Original n timepoints per sample: '+str(logdata['n_timepoints_per_sample'])] #, timesteps #logdata['n_timepoints_per_sample']

#n2+1
# logdata['n_timepoints_per_sample']

# channel_spec={0:'np_O1',1:'np_P3',2:'np_C3',3:'np_F3',4:'np_F4',5:'np_C4',6:'np_P4',7:'np_O2',
#               8:'sf_ch1',9:'sf_ch2',10:'sf_ch3',11:'sf_ch4',12:'sf_ch5',13:'sf_ch6',14:'sf_ch7',15:'sf_ch8',16:'sf_enc'}
# signal_log_exploration_controls=[html.Div(
#     dcc.Store(id='log_session_data',data=None),
#     dcc.RangeSlider(id='log_range_slider',min=0, max=1, step=1, marks=None, value=[0, 1], tooltip={"placement": "bottom", "always_visible": True, "template": "Timestep {value}"}),
#     html.Br(),
#     'Interval for history plot calculations: ',
#     dcc.Input(type='number', placeholder='1,2,3 etc.', value=500, id='log_intermedcalc_interval'),
#     html.Br(),
#     'Plot channels: ',
#     dcc.Dropdown(options=list(channel_spec.values()), value=list(channel_spec.values()), id='logchs', multi=True), 
#     html.Br(),
#     'Fft frequency bindings ',
#     html.Br(),
#     dcc.RangeSlider(id='fft_range_slider',min=0, max=50, step=1, marks=None, value=[0, 1], tooltip={"placement": "bottom", "always_visible": True, "template": "{value} Hz"}),
#     html.Br(),
#     'Datapoint id for display: ',
#     dcc.Input(type='text', placeholder='datapoint_id', value='', id='query_datapoint'),
#     ' ',
#     html.Button("Display data point info", id="datapoint_display_btn", style=b_vis, n_clicks=0),
#     html.Br(),
#     dcc.Textarea(
#     id='point_info',
#     value=' ',
#     style={'width': '100%', 'height': 150, 'font-size': '14px', 'line-height': '1'},
#     maxLength=2024, # Adjust the maxLength as needed
#     rows=32, # Adjust the number of rows as needed
#     cols=128, # Adjust the number of columns as needed
#     ),
#     html.Br(),
#     'New reward formula: ',
#     dcc.Input(type='text', placeholder='Formula string', value='raw_ch16', id='new_formula_string', size='50'),
#     html.Br(),
#     'Frequency bins to plot: ',
#     dcc.Input(type='text', placeholder='Bin values, Hz', value='1,4;4,8;8,14;14,35;35,50', id='fbins_toplot'),  
#     html.Br(),
#     html.Button("Update plots", id="replot_data_btn", style=b_vis, n_clicks=0), style=invis)
# ]

@callback(
    Output('r_model_data', "value"),
    Input("show_r_model_stats", "n_clicks"),
    State('model prefix','value'),
    prevent_initial_call=True
)
def explore_session_data_panel_formation(n1, mpath):
    try:
        if 'CLASSIFIER' in mpath:
            suf=mpath.split('CLASSIFIER_')[1].split('.job')[0]
            tp='CLASSIFIER_'
        else:
            suf=mpath.split('REGRESSOR_')[1].split('.job')[0]
            tp='REGRESSOR_'

        mpath=f'./reward_models/{tp}{suf}.joblib'
        statpath=f'./reward_models/STATS_{suf}.pkl'
        with open(statpath, 'rb') as f:
            stats=pickle.load(f)
        print(stats)

        txt=f"ROC: {stats['roc_auc']} clsrep: {stats['clsrep']}, cm {stats['cm']}"
        if 'optimal_threshold' in stats.keys():
            txt+=f', optimal threshold: {stats["optimal_threshold"]}'
    except Exception as e:
        print(e)
    return txt








@callback(
    Output('point_info', "value"),
    Input("datapoint_display_btn", "n_clicks"),
    State('query_datapoint','value'),
    State('log_session_data', 'data'),
    prevent_initial_call=True
)
def explore_session_data_panel_formation(n1, did, sdata):
    try:
        acts=sdata['acts']
        print(acts.keys())
        rdf=pd.read_json(sdata['rdf'], orient='records')
        rdfp=rdf[rdf.datapoint==did]
        print(rdfp)
        cdtp=str(rdfp['t'].tolist()[0])
        print(cdtp)
        action=acts[cdtp]
        print(action)
    except Exception as e:
        print('H')
        print(e)
    return json.dumps(action)











@callback(
    Output('logplotcontainer','children'),


    Input("replot_data_btn", "n_clicks"),
    State('log_range_slider', "value"),
    State('logchs', "value"),
    State('fft_range_slider', "value"),
    State('log_session_data', 'data'),
    State('log_intermedcalc_interval', 'value'),
    State('msg_logd', 'children'),
    prevent_initial_call=True
)
def explore_session_data_panel_formation(n1, drange, logchs, fftrange, sdata, analysis_step, msg_logd):
    global log_env
    step_rewards=sdata['step_rewards']
    episode_total_rewards=sdata['episode_total_rewards']
    acts=sdata['acts']
    rdf=pd.read_json(sdata['rdf'], orient='records')

    orig_reward_fig=go.Figure(data=go.Scatter(x=list(step_rewards.keys()), y=list(step_rewards.values()), mode='lines+markers'))
    orig_reward_fig.update_layout(title=f'Step reward data')
    orig_episode_reward_fig=go.Figure(data=go.Scatter(x=list(episode_total_rewards.keys()), y=list(episode_total_rewards.values()), mode='lines+markers'))
    orig_episode_reward_fig.update_layout(title=f'Total episode reward data')
    startind=drange[0]
    endind=drange[1]
    startfft=fftrange[0]
    endfft=fftrange[1]
    f_plot_cur=[]
    #print(log_env.f_plot)
    for i in log_env.f_plot:
        if i>=startfft and i<=endfft:
        #    print('h')
            f_plot_cur.append(i)
    print(f_plot_cur)
    startfftind=list(log_env.f_plot).index(f_plot_cur[0])
    endfftind=list(log_env.f_plot).index(f_plot_cur[-1])

   
    crd=rdf.iloc[startind:endind]
    crd=crd.to_numpy()[:,6:-1]
    logchs_inds=[]
    for i in logchs:
        logchs_inds.append(log_env.all_input_channels.index(i))


   # print('H2')
    new_rewards=[]
    print(crd.shape)
    nsteps=int(crd.shape[0]/log_env.n_timepoints_per_sample)
    origdatasplit=np.array_split(crd,nsteps)
    try:
        for i in origdatasplit:
            print('h')
            print(i.shape)
            nobs={}
            nobs['raw_data']=i #[:,log_env.channels_of_interest_inds]
            nobs['fft']=log_env.get_fft_allchannels(nobs['raw_data'])
            nobs['fbins']=log_env.get_bin_values_allchannels(nobs['fft'])
            rev=log_env.get_reward(nobs, toreturn=True)
            new_rewards.append(rev)
        print(len(step_rewards.keys()))
        print(len(new_rewards))
        new_reward_fig=go.Figure(data=go.Scatter(x=list(step_rewards.keys()), y=new_rewards, mode='lines+markers'))
        new_reward_fig.update_layout(title=f'New step reward data')
    except Exception as e:
        print(e)

    crd=crd[:,logchs_inds]
    log_env.all_input_channels=logchs
    try:


        cffts=log_env.get_fft_allchannels(crd)

        #print(cffts.shape)
        fbins_data=log_env.get_bin_values_allchannels(cffts)
        cffts_filtered=cffts[:,startfftind:endfftind]

        raw_signal=crd
        figs_raw_signal = {}
        timepoints=rdf.datapoint.tolist()
        #print(raw_signal)
    # print(raw_signal.shape)
        print(log_env.all_input_channels)
        for i in range(raw_signal.shape[1]):
            #print(i)
            fig = go.Figure(data=go.Scatter(x=timepoints, y=raw_signal[:, i], mode='lines'))
            fig.update_layout(title=f'Raw Signal for Channel {log_env.all_input_channels[i]}')
            figs_raw_signal[log_env.all_input_channels[i]]=dcc.Graph(figure=fig)

        figs_fft = {}
        for i in range(cffts_filtered.shape[0]):
            fig = go.Figure(data=go.Scatter(x=f_plot_cur, y=cffts_filtered[i], mode='lines'))
            fig.update_layout(title=f'FFT for Channel {log_env.all_input_channels[i]}')
            figs_fft[log_env.all_input_channels[i]]=dcc.Graph(figure=fig)

        print(cffts_filtered.shape)
        print(len(f_plot_cur[:-1]))
        print(log_env.all_input_channels)
        fig_heatmap_fft = go.Figure(data=go.Heatmap(z=cffts_filtered, x=f_plot_cur[:-1], y=log_env.all_input_channels))
        fig_heatmap_fft.update_layout(title='Heatmap of FFT Results')
        #fig_heatmap_fft.show()

        fig_heatmap_raw = go.Figure(data=go.Heatmap(z=raw_signal.T, y=log_env.all_input_channels, x=np.arange(raw_signal.shape[1])))
        fig_heatmap_raw.update_layout(title='Heatmap of Raw Signal for All Channels')

        bin_figs={}
        for chidx in range(len(log_env.all_input_channels)):
            chbins=fbins_data[chidx]
            fig = go.Figure(data=go.Bar(x=log_env.fbin_axis_labels, y=chbins, name=f'Channel {log_env.all_input_channels[chidx]} frequency bins'))
            fig.update_layout(title=f'Signal Bins for {log_env.all_input_channels[chidx]}')
            bin_figs[log_env.all_input_channels[chidx]]=dcc.Graph(figure=fig)


        #print('H0')
        nbins=int(np.floor(crd.shape[0]/analysis_step))
        databins=np.array_split(crd,nbins)
        bin_vals_ns=[]
        for nsraw in databins:
            nsfft=log_env.get_fft_allchannels(nsraw)
            nsfbins=log_env.get_bin_values_allchannels(nsfft)
            bin_vals_ns.append(nsfbins)
        #print('H1')
        bin_vals_ns=np.array(bin_vals_ns)
        figs_bin_values = {}
        print(bin_vals_ns.shape)
        try:
            for i in range(bin_vals_ns.shape[1]):
                #print(i)
                chbins=bin_vals_ns[:,i,:]
                fig = go.Figure()
                for z in range(chbins.shape[1]):
                    #print(chbins[:,z].shape)
                    #print(log_env.fbin_axis_labels)
                    #print(z)
                    fig.add_trace(go.Scatter(x=list(range(chbins.shape[0])), y=chbins[:, z], mode='lines', name=f'Bin {log_env.fbin_axis_labels[z]}'))
                fig.update_layout(title=f'Bin Values for Channel {log_env.all_input_channels[i]} Over Time with step  {analysis_step}')
                figs_bin_values[log_env.all_input_channels[i]]=dcc.Graph(figure=fig)
        except Exception as e:
            print(e)
        figs_bin_heatmaps = {}
        try:
            for i in range(bin_vals_ns.shape[2]):
                chbins=bin_vals_ns[:,:,i]
                fig=go.Figure(data=go.Heatmap(z=chbins.T, x=list(range(chbins.shape[0])), y=log_env.all_input_channels))
                fig.update_layout(title=f'Heatmap of Bin {log_env.fbin_axis_labels[i]} Results')
                figs_bin_heatmaps[log_env.fbin_axis_labels[i]]=dcc.Graph(figure=fig)
        except Exception as e:
            print(e)
        # print('H2')
        # new_rewards=[]
        # print(crd.shape)
        # nsteps=int(crd.shape[0]/log_env.n_timepoints_per_sample)
        # origdatasplit=np.array_split(crd,nsteps)
        # try:
        #     for i in origdatasplit:
        #         print('h')
        #         print(i.shape)
        #         nobs={}
        #         nobs['raw_data']=i #[:,log_env.channels_of_interest_inds]
        #         nobs['fft']=log_env.get_fft_allchannels(nobs['raw_data'])
        #         nobs['fbins']=log_env.get_bin_values_allchannels(nobs['fft'])
        #         rev=log_env.get_reward(nobs, toreturn=True)
        #         new_rewards.append(rev)
        #     print(len(step_rewards.keys()))
        #     print(len(new_rewards))
        #     new_reward_fig=go.Figure(data=go.Scatter(x=list(step_rewards.keys()), y=new_rewards, mode='lines+markers'))
        #     new_reward_fig.update_layout(title=f'New step reward data')
        # except Exception as e:
        #     print(e)
        print("H&&&")
        col1=[]
        col2=[]
        mid=[]

        if 'sf_enc' in figs_raw_signal:
            mid.append(figs_raw_signal['sf_enc'])
        if 'sf_enc' in figs_fft:
            mid.append(figs_fft['sf_enc'])
        if 'sf_enc' in bin_figs:
            mid.append(bin_figs['sf_enc'])
        if 'sf_enc' in figs_bin_values:
            mid.append(figs_bin_values['sf_enc'])
        if 'sf_enc' in figs_bin_heatmaps:
            mid.append(figs_bin_heatmaps['sf_enc'])

        for i in ['np_F3','np_C3','np_P3','np_O1', 'sf_ch1', 'sf_ch2', 'sf_ch3','sf_ch4']:
            if i in figs_raw_signal:
                col1.append(figs_raw_signal[i])
        for i in ['np_F4','np_C4','np_P4','np_O2', 'sf_ch5', 'sf_ch6', 'sf_ch7', 'sf_ch8']:
            if i in figs_raw_signal:
                col2.append(figs_raw_signal[i])

        for i in ['np_F3','np_C3','np_P3','np_O1', 'sf_ch1', 'sf_ch2', 'sf_ch3','sf_ch4']:
            if i in figs_fft:
                col1.append(figs_fft[i])
        for i in ['np_F4','np_C4','np_P4','np_O2', 'sf_ch5', 'sf_ch6', 'sf_ch7', 'sf_ch8']:
            if i in figs_fft:
                col2.append(figs_fft[i])

        for i in ['np_F3','np_C3','np_P3','np_O1', 'sf_ch1', 'sf_ch2', 'sf_ch3','sf_ch4']:
            if i in bin_figs:
                col1.append(bin_figs[i])
        for i in ['np_F4','np_C4','np_P4','np_O2', 'sf_ch5', 'sf_ch6', 'sf_ch7', 'sf_ch8']:
            if i in bin_figs:
                col2.append(bin_figs[i])

        for i in ['np_F3','np_C3','np_P3','np_O1', 'sf_ch1', 'sf_ch2', 'sf_ch3','sf_ch4']:
            if i in figs_bin_values:
                col1.append(figs_bin_values[i])
        for i in ['np_F4','np_C4','np_P4','np_O2', 'sf_ch5', 'sf_ch6', 'sf_ch7', 'sf_ch8']:
            if i in figs_bin_values:
                col2.append(figs_bin_values[i])

        for i in ['np_F3','np_C3','np_P3','np_O1', 'sf_ch1', 'sf_ch2', 'sf_ch3','sf_ch4']:
            if i in figs_bin_heatmaps:
                col1.append(figs_bin_heatmaps[i])
        for i in ['np_F4','np_C4','np_P4','np_O2', 'sf_ch5', 'sf_ch6', 'sf_ch7', 'sf_ch8']:
            if i in figs_bin_heatmaps:
                col2.append(figs_bin_heatmaps[i])

        res=[dbc.Row(children=[dcc.Graph(figure=orig_reward_fig),
                           dcc.Graph(figure=orig_episode_reward_fig),
                           dcc.Graph(figure=new_reward_fig),
                           dcc.Graph(figure=fig_heatmap_fft),
                           dcc.Graph(figure=fig_heatmap_raw)]+mid),
        dbc.Row(children=[dbc.Col(children=col1),dbc.Col(children=col2)])]
    except Exception as e:
        print(e)
    #figs_raw_signal
    #figs_fft
    #bin_figs
    #figs_bin_values
    #figs_bin_heatmaps
    return res #[dcc.Graph(figure=orig_reward_fig), dcc.Graph(figure=orig_episode_reward_fig)]+[dcc.Graph(figure=new_reward_fig)]+figs_raw_signal+figs_fft+[dcc.Graph(figure=fig_heatmap_fft)]+[dcc.Graph(figure=fig_heatmap_raw)]+bin_figs+figs_bin_values+figs_bin_heatmaps



















@callback(
    Output('action_text', "value"),
    Input("get_current_action", "n_clicks"),
    prevent_initial_call=True
)
def toggle_offcanvas_scrollable(n1):
    global  env
    act=env.get_json_string_from_ordered_dict(env.current_actions)
    return act
    
@callback(
    Output("stop_audiovis", "n_clicks"),
    Input("stop_audiovis", "n_clicks"),
    prevent_initial_call=True
)
def stop_audiovis_fb(n1):
    global  env
    env.stop_audiovis_feedback()
    return n1

@callback(
    Output("send_display_text", "n_clicks"),
    Input("send_display_text", "n_clicks"),
    Input("clear_display", "n_clicks"),
    State('oled_text', 'value'),
    State('text_size','value'),
    prevent_initial_call=True
)
def toggle_offcanvas_scrollable(n_clicks_st, n_clicks_cd, text, text_size):
    trigger = ctx.triggered[0]
    trigger_id = trigger['prop_id'].split('.')[0]
    msg='display_text:'+str(text_size)+':'+text
    if 'env' in globals():
        global env
        if trigger_id=="send_display_text":
            env.ws_sf.send(msg)
        else:
            env.ws_sf.send("turn_off_display")
    else:
        env = SFSystemCommunicator()
        if trigger_id=="send_display_text":
            env.ws_sf.send(msg)
        else:
            env.ws_sf.send("turn_off_display")
        

@callback(
    Output('session_name', "value"),
    Input("set_session_name_to_timestamp", "n_clicks"),
    prevent_initial_call=True
)
def set_session_name_to_ts(n1):
    
    # Get the current date
    current_date = datetime.now()

    # Format the date as 'd.month.,year'
    formatted_date = current_date.strftime("%d.%m.%Y_%H:%M")
    return formatted_date




@callback(
    Output('data_cleaning_interval', "disabled"),
    Output('data_cleaning_interval', "interval"),
    Input('clear_trainer_data_n_seconds', "value"),
    prevent_initial_call=True
)
def toggle_offcanvas_scrollable(n_seconds):
    interval=n_seconds*1000
    if n_seconds==-1:
        return True, -1
    else:
        return False, interval



@callback(
    Output("clear_trainer_data", "n_clicks"),
    Input("clear_trainer_data", "n_clicks"),
    Input('data_cleaning_interval', "n_intervals"),
    prevent_initial_call=True
)
def toggle_offcanvas_scrollable(n1, n_intervals):
    global env
    global trainer
    trainer.env.previous_episodes_max_rewards=trainer.env.previous_episodes_max_rewards[-10:]
    trainer.env.previous_episodes_total_rewards=trainer.env.previous_episodes_max_rewards[-10:]
    return n1

@callback(
    Output('r_model_info', "is_open"),
    Input("show_r_model_list", "n_clicks"),
    State('r_model_info', "is_open"),
)
def toggle_offcanvas_scrollable(n1, is_open):
    if n1:
        return not is_open
    return is_open



@callback(
    Output('formula_instructions', "is_open"),
    Input("open_formula_instructions", "n_clicks"),
    State('formula_instructions', "is_open"),
)
def toggle_offcanvas_scrollable(n1, is_open):
    if n1:
        return not is_open
    return is_open



@callback(
    Output('plot_panel', "is_open"),
    Input("open_plot_panel", "n_clicks"),
    State('plot_panel', "is_open"),
)
def toggle_offcanvas_scrollable(n1, is_open):
    if n1:
        return not is_open
    return is_open

@callback(
    Output("offcanvas_channel_map", "is_open"),
    Input("open_channel_map", "n_clicks"),
    State("offcanvas_channel_map", "is_open"),
)
def toggle_offcanvas_scrollable(n1, is_open):
    if n1:
        return not is_open
    return is_open

@callback(
    Output("offcanvas_session_lib", "is_open"),
    Input("open_session_lib", "n_clicks"),
    State("offcanvas_session_lib", "is_open"),
)
def toggle_offcanvas_scrollable(n1, is_open):
    if n1:
        return not is_open
    return is_open

@callback(
    Output("offcanvas_session_lib", "children"),
    Input('session_library', "data"),
)
def populate_session_lib_offcanvas(data):
    return [html.P(item) for item in data]


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

def code_wave_shapes(w):
    if w=='Noise':
        return 0
    if w=='Sine':
        return 1
    if w=='Square':
        return 2
    if w=='Triangle':
        return 3
    
@callback([Output('l1c','value'),
          Output('l2c','value'),
          Output('l3c','value'),
          Output('l4c','value'),
          Output('l5c','value'),
          Output('l6c','value'),
          Output('l7c','value'),
          Output('l8c','value'),
          Output('maxivolume','value')],
          Input('defaults_mode', 'value'),
          State('maxivolume','value'),
          prevent_initial_call=True)
def change_defaults(dm, vol):
    if dm=='Set specific defaults':
        raise PreventUpdate
    if dm=='Set no lighting defaults':
        return ['0,0,0' for i in range(8)]+[vol]
    if dm=='Set fully blank defaults':
        return ['0,0,0' for i in range(8)]+[0]
    

@callback(Output('audio','src'),
          Output('audio','autoPlay'),
          Output('audio','loop'),
          Input('audio_recording_path', 'value'),
          Input('audio_options','value'),
          prevent_initial_call=True)
def audio_conttrols(src, opts):
    src='/suggestions/' +src
    if 'Auto Play' in opts:
        autoplay=True
    else:
        autoplay=False
    if 'Loop' in opts:
        loop=True
    else:
        loop=False
    return src, autoplay, loop

def copy_file(source_file_path, destination_folder_path):
    shutil.copy(source_file_path, destination_folder_path)
@callback(Output('session_library', 'data'),
          Input("move_data", 'n_clicks'),
          State('session_name', 'value'),
          prevent_initial_call=True)
def copy_session_logs_to_lib(n_clicks, sname):
    session_dir=f'./session_lib/{sname}'
    wavdir=session_dir+'/wavs'
    if os.path.isdir(session_dir):
        shutil.rmtree(session_dir)
    os.mkdir(session_dir)
    os.mkdir(wavdir)
    cfiles=os.listdir('./')
    for f in cfiles:
        if f in ['current_training.log', 'model_stats.log','best_total_episode_reward_model.zip','best_overall_reward_model.zip' ,'last_model.zip']:
            copy_file(f'./{f}', session_dir)
        if f.endswith('.wav'):
            copy_file(f'./{f}', wavdir)
        if 'act' in f:  #copy actions
            copy_file(f'./{f}', session_dir)
        if f==f'{sname}_edf':
            src_dir=os.getcwd()+'/'+f'{sname}_edf'
            copy_directory(src_dir, session_dir+'/'+f'{sname}_edf')
    return get_dir_tree('./session_lib')

@callback(Output('session_library', 'data', allow_duplicate=True),
          Input("clear_session_lib", 'n_clicks'),
          prevent_initial_call=True)
def clear_session_lib(n_clicks):
    shutil.rmtree('./session_lib')
    os.mkdir('./session_dir')
    return get_dir_tree('./session_lib')

@callback(Output('clear_unclicked_wavs', 'children'),
          Input('clear_unclicked_wavs', 'n_clicks'),
          State('clear_unclicked_wavs', 'children'),
          prevent_initial_call=True)
def clear_logfiles(n_clicks, ch):
    clear_unclicked_wavs()
    return ch

@callback(Output('clear_logfiles', 'children'),
          Input('clear_logfiles', 'n_clicks'),
          State('clear_logfiles', 'children'),
          State('logfn','value'),
          prevent_initial_call=True)
def clear_logfiles(n_clicks, ch, logfn):
    if os.path.isfile(logfn):
        os.remove(logfn)
    if not os.path.isfile(logfn):
        open(logfn, 'a').close()
    if os.path.isfile('model_stats.log'): #ADDITIONALLY REMOVING ALSO MODEL STATS
        os.remove('model_stats.log')
    if not os.path.isfile('model_stats.log'):
        open('model_stats.log', 'a').close()
    clear_wavs()
    clear_edfs()
    return ch

@callback(Output('message_row', 'children', allow_duplicate=True),
          Input('timer_interval', 'n_intervals'),
          State('timer_reward_threshold', 'value'),
          State('oled_text', 'value'),  #oled text will also be displayed
          State('text_size', 'value'), 
          prevent_initial_call=True)
def run_timer(n_intervals, reward_thresh, text, text_size):
    #print(n_intervals)
    msg='display_text:'+str(text_size)+':'+text
    global env
    env.ws.send(msg)
    if env.ws_np is not None:
        if env.write_edf_ann==True:
            anntxt=f'running timer'#f'current_overall_max_reward_{reward_val}_action_{actionstring}_inprev_{self.step_stim_length}_s'
            env.write_edf_annotation_fn(ann_text=anntxt, ann_duration_ms=env.delay)
    if reward_thresh==-1:
        env.step(env.default_actions)
        env.stop_audiovis_feedback()
        env.ws.send("turn_off_display")
        return [f'Timer signal has run {n_intervals} times']
    else:
        if env.use_np:
            tocont=False
            while tocont==False:
                obs=env.sample_and_process_observations_from_device() #only launch if signaal quality is ok
                if env.np_reward_sig_qual_filter==True:
                    tocont=True
                    break
        else:
            obs=env.sample_and_process_observations_from_device()
        reward=env.get_reward(observations=obs, toreturn=True)
        if reward>reward_thresh:
            env.step(env.default_actions)
            env.stop_audiovis_feedback()
            env.ws.send("turn_off_display")
            return [f'Timer signal has run {n_intervals} times']

@callback(Output('training_figure_container', "children"),
          Output('signal_figure_container', "children"),
          Output('message_row', 'children', allow_duplicate=True),
          Output('signal_log_exploration_controls', "style", allow_duplicate=True),
          Input('training_status_update', 'n_intervals'),
          State('run_type', 'data'),
          prevent_initial_call=True)
def collect_settings(n_intervals, runtype):
    global env
    global trainer
    #print(env.figures)
    training_fig=dcc.Graph(id=f'training_figure',
                    figure=env.figures['training_fig'],
                    config={'staticPlot': False},)
    signal_fig=dcc.Graph(id=f'training_figure',
                    figure=env.figures['signal_fig'],
                    config={'staticPlot': False},)
    messages=[]

    if runtype=='train':
        if str(trainer.cur_episode_no)!='None':
            messages=[f'Trainer episode N {trainer.cur_episode_no+1} (progress {int(trainer.cur_episode_no*100/trainer.num_episodes)}%)',
                html.Br(),
                f'Reward: {trainer.env.reward} ',
                f'Episode max reward: {trainer.env.episode_max_reward} ',
                f'Overall max reward: {trainer.env.overall_max_reward} ',
                html.Br(),
                f'Total current episode reward: {trainer.env.total_cur_episode_reward} ',
                f'Total total episode max reward: {trainer.env.total_episode_max_reward} ',
                html.Br(),
                f'Environment step N {trainer.env.cur_step} (progress {int(trainer.env.cur_step*100/trainer.env.n_steps_per_episode)}%)',
                    html.Br(),
                    f'{trainer.stat1} {trainer.stat2}',
                    html.Br(),
                    f'Best action overall: {trainer.env.best_action_overall} (reward {trainer.env.overall_max_reward})',
                    html.Br(),
                    f'Current action: {trainer.env.current_actions}',
                    f'Training completion: {trainer.training_completed}']   
        else:
            messages=['Training not launched yet.']         
    return training_fig, signal_fig, messages, invis



@callback(Output('signal_plot_color','value'),
          Input("color_resample", 'n_clicks'),
          State('signal_plot_color','value'),
          State('settings_dictionary', 'data'),
          prevent_initial_call=True)
def collect_settings(n_clicks, sigplot_color, setd):
    global env
    #print(env.figures)
    if sigplot_color.split(',')[0]=='random':
        sigplot_colors=get_random_css_color_names(setd['session_settings']['n_input_channels'], seed=int(sigplot_color.split(',')[1]))
    else:
        sigplot_colors=[sigplot_color for i in range(setd['session_settings']['n_input_channels'])]
    env.colors=sigplot_colors
                    
    return sigplot_color
def get_state_from_model_logfile(logfile=None):
    if str(logfile)=='None':
        logfile='model_stats.log'
    with open(logfile, 'r') as f:
        data = f.read()
    envdata=json.loads(data.split('\n')[0])
    return envdata

@callback(Output("start_session_train", "style"),
          Output("start_session_notrain", "style"),
          Output("start_session_static", "style"),
          Output("stop_session_train", "style"),
          Output("additional_session", "style"),
          Output('training_status_update', 'disabled'),
          Output('training_status_update', 'interval'),
          Output("run_trained", "style"),
          Output("run_type", "data"),

          Output("run_timer", "style"),
          Output("stop_timer", "style"),
          Output("run_direct_feedback", "style"),
          Output("stop_direct_feedback", "style"),
          Output('timer_interval', 'disabled'),
          Output('timer_interval', 'interval'),

          
          Input("start_session_train", "n_clicks"),
          Input("start_session_notrain", "n_clicks"),
          Input("start_session_static", "n_clicks"),
          Input("stop_session_train", "n_clicks"),
          Input("additional_session", "n_clicks"),
          Input("run_trained", "n_clicks"),
          Input("run_timer","n_clicks"),
          Input("run_direct_feedback","n_clicks"),
          Input("stop_timer","n_clicks"),
          Input("stop_direct_feedback","n_clicks"),
          Input("run_action","n_clicks"),
          Input("action_from_string","n_clicks"),



          State('settings_dictionary', 'data'),
          State('info_upd_interval', 'value'), 
          State('signal_plot_color','value'),
          State('n_steps_notrain','value'),
          State('model_name', 'value'),
          State('train_logged_orig', 'value'),
          State('train_logged_new', 'value'),
          State('timer_interval_mins','value'),
          State('timer_signal_duration_s','value'),
          State('deterministic_opts','value'),
          State('reward_mapping_interval', 'value'),
          State('overlay_random', 'value'),
          State('pause_on_click',   'value'),
          State('log_actions_on_hold', 'value'),
          State('action_log_name','value'),
          State('action_text', 'value'),


          State('write_edf_ann', 'value'),
          State('edf_step_annotation','value'),
          State('edf_rf_annotation','value'),
          State('edf_rf_annotation_threshold','value'),
          State('session_name','value'),
          State('send_reward_to_display', 'value'),
          State('text_size','value'),

          State('send_np_signal_to_display','value'),
          State('start_on_click','value'),
          State('pause_learning_if_reward_sig_qual_false','value'),
          State('start_on_reward_sig_qual', 'value'),
          State('reward_np_sigqual_thresh', 'value'),
          State('mic_log_opts', 'value'),

          State('use_reward_model','value'),
          State('model prefix', 'value'),
          State('r_model_inputdescr', 'value'),
          State('r_model_predtype','value'),
          State('r_model_voting','value'),
          State('r_model_customthresh', 'value'),
          prevent_initial_call=True)
def collect_settings(n_clicks_t, n_clicks_nt, n_clicks_static, n_clicks_stop, n_clicks_additional, n_clicks_run_trained, n_clicks_run_timer, n_clicks_run_direct_feedback, n_clicks_stop_timer, n_clicks_stop_direct_feedback, 
                     n_clicks_run_action,  n_clicks_action_from_string,
                     setd, info_upd_interval, sigplot_color, n_steps_notrain,
                     model_name, train_logged_orig, train_logged_new, 
                     timer_interval_mins,
                     timer_signal_duration_s,
                     deterministic_opts,
                     reward_mapping_interval,
                     overlay_random,
                     pause_on_click,
                     log_actions_on_hold,
                     action_log_name,
                    action_text,
                    write_edf_ann,
                    edf_step_annotation,
                    edf_rf_annotation,
                    edf_rf_annotation_threshold,
                    session_name,
                    send_reward_to_display,
                    text_size,
                    send_np_signal_to_display,
                    start_on_click,
                    pause_learning_if_reward_sig_qual_false,
                    start_on_reward_sig_qual,
                    reward_np_sigqual_thresh,
                    mic_log_opts,
                    use_reward_model,
                    model_prefix,
                    r_model_inputdescr,
                    r_model_predtype,
                    r_model_voting,
                    r_model_customthresh
                    ):
    
    if len(use_reward_model)>0:
        use_reward_model=True
    else:
        use_reward_model=False
    if 'CLASSIFIER' in model_prefix:
        suf=model_prefix.split('CLASSIFIER_')[1].split('.job')[0]
        tp='CLASSIFIER_'
    else:
        suf=model_prefix.split('REGRESSOR_')[1].split('.job')[0]
        tp='REGRESSOR_'

    r_model_path=f'./reward_models/{tp}{suf}.joblib'
    r_model_stats_path=f'./reward_models/STATS_{suf}.pkl'




    edf_ann_fn=session_name+'_edf'
    if 'Continuous logging' in mic_log_opts:
        mic_log_continuous=True
    else:
        mic_log_continuous=False

    if 'Log on click' in mic_log_opts:
        mic_log_onclick=True
    else:
        mic_log_onclick=False

    if 'Separate files logging' in mic_log_opts:
        mic_log_sepfiles=True
    else:
        mic_log_sepfiles=False

    print(f'Mic Log Opt at init: {mic_log_sepfiles}')

    if len(start_on_reward_sig_qual)>0:
        start_on_reward_sig_qual=True
    else:
        start_on_reward_sig_qual=False


    if len(pause_learning_if_reward_sig_qual_false)>0:
        pause_learning_if_reward_sig_qual_false=True
    else:
        pause_learning_if_reward_sig_qual_false=False

    if len(start_on_click)>0:
        start_on_click=True
    else:
        start_on_click=False

    if len(send_np_signal_to_display)>0:
        send_np_signal_to_display=True
    else:
        send_np_signal_to_display=False

    if len(send_reward_to_display)>0:
        send_reward_to_display=True
    else:
        send_reward_to_display=False



    if len(edf_rf_annotation)>0:
        edf_rf_annotation=True
    else:
        edf_rf_annotation=False
    if len(edf_step_annotation)>0:
        edf_step_annotation=True
    else:
        edf_step_annotation=False
    if len(write_edf_ann)>0:
        write_edf_ann=True
    else:
        write_edf_ann=False
    global env
    global trainer
    timer_interval_ms=timer_interval_mins*60*1000
    if len('pause_on_click')>0:
        pause_on_click=True
    else:
        pause_on_click=False

    if len('log_actions_on_hold')>0:
        log_actions_on_hold=True
    else:
        log_actions_on_hold=False

    trigger = ctx.triggered[0]
    trigger_id = trigger['prop_id'].split('.')[0]
    trigger_value = trigger['value']
    out_dict=setd['out_dict']
    sd=setd['session_settings']

    if sigplot_color.split(',')[0]=='random':
        sigplot_colors=get_random_css_color_names(sd['n_input_channels'], seed=int(sigplot_color.split(',')[1]))
    else:
        sigplot_colors=[sigplot_color for i in range(sd['n_input_channels'])]


    if trigger_id in ['start_session_train', "start_session_static", "start_session_notrain", "run_timer", "run_direct_feedback", "run_action", "action_from_string"]:
        if trigger_id=="run_timer":
            sd['step_stim_length_millis']=timer_signal_duration_s*1000
        env = SFSystemCommunicator(out_dict=out_dict,
                                                #n_input_channels=sd['n_input_channels'],
                                                input_channels=sd['channels_of_interest'],
                                                n_timepoints_per_sample=sd['n_timepoints_per_sample'],
                                                #max_sfsystem_output=sd['max_sfsystem_output'],
                                                reward_formula_string=sd['reward_formula_string'],
                                                fbins=sd['fbins'],
                                                delay=sd['delay'],
                                                use_raw_in_os_def=sd['use_raw_in_os_def'],
                                                use_freq_in_os_def=sd['use_freq_in_os_def'],
                                                use_fbins_in_os_def=sd['use_fbins_in_os_def'],
                                                device_address=sd['device_address'],
                                                step_stim_length_millis=sd['step_stim_length_millis'],
                                                episode_time_seconds=sd['episode_time_seconds'],
                                                logfn=sd['logfn'],
                                                log_steps=sd['log_steps'],
                                                log_episodes=sd['log_episodes'],
                                                log_best_actions_final=sd['log_best_actions_final'],
                                                signal_plot_width=sd['signal_plot_width'],
                                                signal_plot_height=sd['signal_plot_height'],
                                                training_plot_width=sd['training_plot_width'],
                                                training_plot_height=sd['training_plot_height'],
                                                write_raw=sd['write_raw'],
                                                write_fft=sd['write_fft'],
                                                write_bins=sd['write_bins'],
                                                log_best_actions_every_episode=sd['log_best_actions_every_episode'],
                                                render_data=sd['render_data'],
                                                render_each_step=sd['render_each_step'],
                                                log_actions_every_step=sd['log_actions_every_step'],
                                                stim_length_on_reset=sd['stim_length_on_reset'],
                                                only_pos_encoder_mode=sd['only_pos_encoder_mode'],
                                                use_abs_values_for_raw_data_in_reward=sd['use_abs_values_for_raw_data_in_reward'],
                                                colors=sigplot_colors,
                                                log_actions_on_hold=log_actions_on_hold,
                                                channel_spec=channel_spec,
                                                use_unfiltered_np_data=sd['use_unfiltered_np_data'],
                                                edf_rf_annotation=edf_rf_annotation,
                                                edf_rf_annotation_threshold=edf_rf_annotation_threshold,
                                                edf_step_annotation=edf_step_annotation,
                                                write_edf_ann=write_edf_ann,
                                                edf_ann_fn=edf_ann_fn,
                                                send_reward_to_display=send_reward_to_display,
                                                text_size=text_size,
                                                reward_np_sigqual_thresh=reward_np_sigqual_thresh,
                                                send_np_signal_to_display=send_np_signal_to_display,
                                                mic_log_continuous=mic_log_continuous,
                                                mic_log_onclick=mic_log_onclick,
                                                mic_log_sepfiles=mic_log_sepfiles,
                                                use_reward_model=use_reward_model,
                                                r_model_path=r_model_path,
                                                r_model_stats_path=r_model_stats_path,
                                                r_model_inputdescr=r_model_inputdescr,
                                                r_model_voting=r_model_voting,
                                                r_model_predtype=r_model_predtype,
                                                r_model_customthresh=r_model_customthresh)
        time.sleep(5)    


                #print(sd)
            #print(out_dict)
            #env.step(env.action_space.sample()) #sample step
        if trigger_id != "run_timer":
            trainer=stable_baselines_model_trainer(initialized_environment=env,
                                                                algorithm=sd['algorithm'],
                                                                policy='MlpPolicy',
                                                                logfn='model_stats.log',
                                                                n_steps_per_timestep=sd['n_steps_per_timestep'],
                                                                start_on_click=start_on_click,
                                                                pause_learning_if_reward_sig_qual_false=pause_learning_if_reward_sig_qual_false,
                                                                start_on_reward_sig_qual=start_on_reward_sig_qual)
                

            training_args={
                    'num_episodes':sd['num_episodes'],
                    'log_model':sd['log_model'],
                    'n_total_timesteps':sd['n_total_timesteps'],
                    'log_or_plot_every_n_timesteps':sd['log_or_plot_every_n_timesteps'],
                    'jnb':False
                    
                }
    if  trigger_id=="action_from_string":
        env.launch_action_from_json_string(action_text)
        raise PreventUpdate
    if trigger_id=="run_action":
        action_log_fn='session_lib/'+action_log_name
        env.read_and_launch_logged_actions(actionlogfn=action_log_fn)
        raise PreventUpdate
    if trigger_id =='start_session_train':
          #print(sd)
          #print(out_dict)
          #env.step(env.action_space.sample()) #sample step
          #print(training_args)
          training_thread = threading.Thread(target=start_training, args=(training_args,))
          training_thread.daemon = True
          training_thread.start()
          return b_invis, b_invis, b_invis, b_vis, b_vis, False, info_upd_interval, b_invis, 'train', b_invis, b_invis, b_invis, b_invis, True, timer_interval_ms
    if trigger_id=="start_session_static":
          training_thread = threading.Thread(target=start_session_static)
          training_thread.daemon = True
          training_thread.start()
          return b_invis, b_invis, b_invis, b_vis, b_vis, False, info_upd_interval, b_invis, 'static', b_invis, b_invis, b_invis, b_invis, True, timer_interval_ms
    if trigger_id=="start_session_notrain":
          training_thread = threading.Thread(target=start_session_notrain, args=({'n_steps_notrain':n_steps_notrain,
                                                                                  'pause_on_click':pause_on_click},))
          training_thread.daemon = True
          training_thread.start()
          return b_invis, b_invis, b_invis, b_vis, b_vis, False, info_upd_interval, b_invis, 'notrain', b_invis, b_invis, b_invis, b_invis, True, timer_interval_ms
    if trigger_id=="additional_session":
        #we collect training arguments in case they changed and run the same trainer
        #we do not initialize either new environment or new trainer
        training_args={
                'num_episodes':sd['num_episodes'],
                'log_model':sd['log_model'],
                'n_total_timesteps':sd['n_total_timesteps'],
                'log_or_plot_every_n_timesteps':sd['log_or_plot_every_n_timesteps'],
                'jnb':False,
                'pause_on_click':pause_on_click
                
        }
        training_thread = threading.Thread(target=start_training, args=(training_args,))
        training_thread.daemon = True
        training_thread.start()
        return b_invis, b_invis, b_invis, b_vis, b_vis, False, info_upd_interval, b_invis, 'train', b_invis, b_invis, b_invis, b_invis, True, timer_interval_ms

    if trigger_id=="run_trained":
        tkns=model_name.split('/')
        session_name=tkns[0]
        model_name=tkns[1]

        envstats=get_state_from_model_logfile(f'session_lib/{session_name}/model_stats.log')
        
  
        out_dict=envstats['out_dict']
        sd_s=envstats['session_settings']
        for key in sd.keys():
            if key in sd_s.keys():
                if len(train_logged_new)==0:
                    sd[key]=sd_s[key]
                else:
                    if key in ['delay', 'n_steps_per_timestep', 'num_episodes', 'n_total_timesteps', 'log_or_plot_every_n_timesteps']: #options which can be changed
                        sd[key]=sd_s[key]
        
        env = SFSystemCommunicator(out_dict=out_dict,
                                                input_channels=sd['channels_of_interest'],
                                                n_timepoints_per_sample=sd['n_timepoints_per_sample'],
                                                #max_sfsystem_output=sd['max_sfsystem_output'],
                                                reward_formula_string=sd['reward_formula_string'],
                                                fbins=sd['fbins'],
                                                delay=sd['delay'],
                                                use_raw_in_os_def=sd['use_raw_in_os_def'],
                                                use_freq_in_os_def=sd['use_freq_in_os_def'],
                                                use_fbins_in_os_def=sd['use_fbins_in_os_def'],
                                                device_address=sd['device_address'],
                                                step_stim_length_millis=sd['step_stim_length_millis'],
                                                episode_time_seconds=sd['episode_time_seconds'],
                                                logfn=sd['logfn'],
                                                log_steps=sd['log_steps'],
                                                log_episodes=sd['log_episodes'],
                                                log_best_actions_final=sd['log_best_actions_final'],
                                                signal_plot_width=sd['signal_plot_width'],
                                                signal_plot_height=sd['signal_plot_height'],
                                                training_plot_width=sd['training_plot_width'],
                                                training_plot_height=sd['training_plot_height'],
                                              write_raw=sd['write_raw'],
                                              write_fft=sd['write_fft'],
                                              write_bins=sd['write_bins'],
                                              log_best_actions_every_episode=sd['log_best_actions_every_episode'],
                                              render_data=sd['render_data'],
                                              render_each_step=sd['render_each_step'],
                                              log_actions_every_step=sd['log_actions_every_step'],
                                              stim_length_on_reset=sd['stim_length_on_reset'],
                                              only_pos_encoder_mode=sd['only_pos_encoder_mode'],
                                              use_abs_values_for_raw_data_in_reward=sd['use_abs_values_for_raw_data_in_reward'],
                                              colors=sigplot_colors,
                                              log_actions_on_hold=log_actions_on_hold,
                                              channel_spec=channel_spec,
                                              use_unfiltered_np_data=sd['use_unfiltered_np_data'],
                                              edf_rf_annotation=edf_rf_annotation,
                                            edf_rf_annotation_threshold=edf_rf_annotation_threshold,
                                            edf_step_annotation=edf_step_annotation,
                                            write_edf_ann=write_edf_ann,
                                            edf_ann_fn=edf_ann_fn,
                                            send_reward_to_display=send_reward_to_display,
                                            text_size=text_size,
                                            reward_np_sigqual_thresh=reward_np_sigqual_thresh,
                                            send_np_signal_to_display=send_np_signal_to_display,
                                            mic_log_continuous=mic_log_continuous,
                                            mic_log_onclick=mic_log_onclick,
                                            mic_log_sepfiles=mic_log_sepfiles,

                                            use_reward_model=sd['use_reward_model'],
                                            r_model_path=sd['r_model_path'],
                                            r_model_stats_path=sd['r_model_stats_path'],
                                            r_model_inputdescr=sd['r_model_inputdescr'],
                                            r_model_voting=sd['r_model_voting'],
                                            r_model_predtype=sd['r_model_predtype'],
                                            r_model_customthresh=sd['r_model_customthresh'])    
        time.sleep(20) 
        trainer=stable_baselines_model_trainer(initialized_environment=env,
                                                            algorithm=sd['algorithm'],
                                                            policy='MlpPolicy',
                                                            logfn='model_stats.log',
                                                            n_steps_per_timestep=sd['n_steps_per_timestep'],
                                                            start_on_click=start_on_click,
                                                            pause_learning_if_reward_sig_qual_false=pause_learning_if_reward_sig_qual_false,
                                                            start_on_reward_sig_qual=start_on_reward_sig_qual)
        trainer.load_model(f'session_lib/{session_name}/{model_name}')
        if len(train_logged_orig)==0 and len(train_logged_new)==0:
            training_thread = threading.Thread(target=start_session_trained_model, args=({'n_steps_notrain':n_steps_notrain,
                                                                                          'pause_on_click':pause_on_click},))
            training_thread.daemon = True
            training_thread.start()
            return b_invis, b_invis, b_invis, b_vis, b_vis, False, info_upd_interval, b_invis, 'log', b_invis, b_invis, b_invis, b_invis, True, timer_interval_ms
        else:
            trainer.set_model_environment() #if we want to train, we need to connect the model to the environment
            training_args={
                'num_episodes':sd['num_episodes'],
                'log_model':sd['log_model'],
                'n_total_timesteps':sd['n_total_timesteps'],
                'log_or_plot_every_n_timesteps':sd['log_or_plot_every_n_timesteps'],
                'jnb':False,
                'pause_on_click':pause_on_click
                
            }
            training_thread = threading.Thread(target=start_training, args=(training_args,))
            training_thread.daemon = True
            training_thread.start()

            return b_invis, b_invis, b_invis, b_vis, b_vis, False, info_upd_interval, b_invis, 'train', b_invis, b_invis, b_invis, b_invis, True, timer_interval_ms
    
    if trigger_id=="stop_session_train":
        try:
            trainer.close_env()
        except Exception as e:
            print(f"On session stop received: {e}")
            try:
                env.close()
            except Exception as e:
                print(f"On environment stop received: {e}")
        return b_vis, b_vis, b_vis, b_invis, b_invis, True, info_upd_interval, b_vis, 'stop', b_vis, b_invis, b_vis, b_invis, True, timer_interval_ms
    
    if trigger_id=="run_timer":
        print('Timer is started')
        print(f'Timer query interval: {timer_interval_ms} ms')
        if env.write_edf_ann:
            env.start_edf_log()
            print('EDF log started:')
            print(env.edfpath)
        env.stop_audiovis_feedback()
        return b_invis, b_invis, b_invis, b_invis, b_invis, True, info_upd_interval, b_vis, 'run_timer', b_invis, b_vis, b_invis, b_invis, False, timer_interval_ms
    if trigger_id=="stop_timer":
        try:
            env.close()
        except Exception as e:
            print(f"On timer stopping received: {e}")
        print('Timer is stopped')
        return b_vis, b_vis, b_vis, b_invis, b_invis, True, info_upd_interval, b_vis, 'stop', b_vis, b_invis, b_vis, b_invis, True, timer_interval_ms

    if trigger_id=="run_direct_feedback":

        if len(overlay_random)>0:
            overlay_random=True
        else:
            overlay_random=False
        training_args={
            'n_steps_notrain':n_steps_notrain,
            'mapped_outputs':deterministic_opts,
            'overlay_random':overlay_random,
            'reward_mapping_min':int(reward_mapping_interval.split('-')[0]),
            'reward_mapping_max':int(reward_mapping_interval.split('-')[1]),
            'pause_on_click':pause_on_click
        }

        training_thread = threading.Thread(target=start_session_direct_feedback, args=(training_args,))
        training_thread.daemon = True
        training_thread.start()

        return b_invis, b_invis, b_invis, b_invis, b_invis, False, info_upd_interval, b_invis, 'run_direct_feedback', b_invis, b_invis, b_invis, b_vis, True, timer_interval_ms
    if trigger_id=="stop_direct_feedback":
        try:
            trainer.close_env()
        except Exception as e:
            print(f"On session stop received: {e}")
            try:
                env.close()
            except Exception as e:
                print(f"On environment stop received: {e}")
        return b_vis, b_vis, b_vis, b_invis, b_invis, True, info_upd_interval, b_vis, 'stop', b_vis, b_invis, b_vis, b_invis, True, timer_interval_ms
    
        
def start_session_direct_feedback(arg):
    global env
    global trainer
    env.step(env.default_actions)
    nsteps_notrain=arg['n_steps_notrain']
    reward_mapping_min=arg['reward_mapping_min']
    reward_mapping_max=arg['reward_mapping_max']
    overlay_random=arg['overlay_random']
    mapped_outputs=arg['mapped_outputs']
    pause_on_click=arg['pause_on_click']
    env_paused=False
    while nsteps_notrain>0:
        try:
            if env_paused==False:
                trainer.direct_feedback_run(reward_mapping_min, reward_mapping_max, overlay_random, mapped_outputs, min_space_value=-1, max_space_value=1)
                nsteps_notrain-=1
                if pause_on_click==True:
                    if trainer.env.enc_is_clicked==1:
                        env_paused=True
            else:
                trainer.env.sample_observations()
                if trainer.env.enc_is_clicked==1:
                    env_paused=False
        except Exception as e:
            n_steps_notrain=0
            print(f"Training thread terminated: {e}")
            try:
                trainer.env.close()
                return
            except Exception as e:
                print(f"On environment closure: {e}")
                return



def start_session_trained_model(arg):
    global env
    global trainer
    nsteps_notrain=arg['n_steps_notrain']
    pause_on_click=arg['pause_on_click']
    env_paused=False
    while nsteps_notrain>0:
        n_notrain=nsteps_notrain
        try:
            if env_paused==False:
                obs=trainer.env.reset()[0]
                action, _states = trainer.model.predict(obs, deterministic=True)
                obs, reward, done, info = trainer.env.step(action)
                nsteps_notrain-=1
                if pause_on_click==True:
                    if trainer.env.enc_is_clicked==1:
                        env_paused=True
            else:
                trainer.env.sample_observations()
                if trainer.env.enc_is_clicked==1:
                    env_paused=False

        except Exception as e:
            n_steps_notrain=0
            print(f"Training thread terminated: {e}")
            try:
                trainer.env.close()
                return
            except Exception as e:
                print(f"On environment closure: {e}")
                return
            


         
def start_training(sd):
    global env
    global trainer
    try:
        trainer.train(num_episodes=sd['num_episodes'], log_model=sd['log_model'],n_total_timesteps=sd['n_total_timesteps'],
                            log_or_plot_every_n_timesteps=sd['log_or_plot_every_n_timesteps'], jnb=False)

    except Exception as e:
        print(f"Training thread terminated: {e}")
        try:
            env.close()
            return
        except Exception as e2:
            print(e2)
            return
        

def start_session_static():
    global env
    global trainer
    try:
        trainer.static_launch()
    except Exception as e:
        print(f"Training thread terminated: {e}")
        try:
            trainer.env.close()
        except Exception as e2:
            print(f"On environment closure: {e2}")
def start_session_notrain(arg):
    global env
    global trainer
    nsteps_notrain=arg['n_steps_notrain']
    pause_on_click=arg['pause_on_click']
    env_paused=False
    while nsteps_notrain>0:
        try:
            print(env_paused)
            if env_paused==False:
                try:
                    res=trainer.dynamic_launch()
                    if res==True:
                        nsteps_notrain=0
                        break
                    else:
                        nsteps_notrain-=1
                        #print(trainer.env.current_sample)
                        #print(trainer.env.enc_is_clicked)
                        if pause_on_click==True:
                            if trainer.env.enc_is_clicked==1:
                                env_paused=True
                except:
                    print('HERE')
                    break

            else:
                trainer.env.sample_observations()
                if trainer.env.enc_is_clicked==1:
                    env_paused=False

        except Exception as e:
            break
            n_steps_notrain=0
            print(f"Training thread terminated: {e}")
            try:
                trainer.env.close()
                return
            except Exception as e2:
                print(f"On environment closure: {e2}")
                return
            
            



@callback(Output('settings_dictionary', 'data'),
          
              Input('flash_frequency_lb','value'),
              Input('flash_frequency_ub','value'),
              Input('flash_frequency_iv','value'),
              Input('rgb_value_range', 'value'),
              Input('l1c','value'),
              Input('l2c','value'),
              Input('l3c','value'),
              Input('l4c','value'),
              Input('l5c','value'),
              Input('l6c','value'),
              Input('l7c','value'),
              Input('l8c','value'),
              Input('sound_wave_frange','value'),
              Input('wave_1_freq','value'),
              Input('wave_2_freq','value'),
              Input('w1sh','value'),
              Input('w2sh','value'),
              Input('volume_range', 'value'),
              Input('panner_phasor_frange', 'value'),
              Input('panner_freq', 'value'),
              Input('panner_div_range', 'value'),
              Input('panner_div', 'value'),
              Input('phasor_1_freq', 'value'),
              Input('phasor_2_freq', 'value'),
              Input('phasor_1_span', 'value'),
              Input('phasor_2_span', 'value'),
              Input('maxivolume','value'),
              Input('channels_of_interest','value'),
              Input('n_timepoints_per_sample','value'),
              Input('delay','value'),
             # Input('max_sfsystem_output','value'),
              Input('formula_string','value'),
              Input('fbins','value'),
              Input('device_address','value'),
              Input('step_stim_length_millis', 'value'),
              Input('episode_time_seconds','value'),
              Input('n_total_timesteps','value'),
              Input('num_episodes', 'value'),
              Input('logfn', 'value'),
              Input('logging_plotting_opts', 'value'),
              Input('log_or_plot_every_n_timesteps', 'value'),
              Input('algorithm', 'value'),
              Input('n_steps_per_timestep','value'),
              Input('obs_space_opts', 'value'),
              Input('signal_plot_width', 'value'),
              Input('signal_plot_height', 'value'),
              Input('training_plot_width', 'value'),
              Input('training_plot_height', 'value'),
              Input('render_data','value'),
              Input('stim_length_on_reset','value'),
              Input('data_proc_options', 'value'),

              Input('use_unfiltered_np_data', 'value'),
              prevent_initial_call=False)
def collect_settings(ffminf, ffmaxf, ffinitf, rgbrange,
                     l1c,l2c,l3c,l4c,l5c,l6c,l7c,l8c,sound_wave_frange,
                     w1f, w2f, w1sh, w2sh, volrange, panner_phasor_frange,panner_freq,
                     panner_div_range,panner_div, phasor_1_freq,phasor_2_freq,phasor_1_span,
                     phasor_2_span,maxivolume, 
                      channels_of_interest,n_timepoints_per_sample,
                     delay,
                    # max_sfsystem_output, 
                     formula_string, fbins, device_address,
                     step_stim_length_millis,episode_time_seconds,n_total_timesteps,
                     num_episodes,logfn,
                     logging_plotting_opts,
                     log_or_plot_every_n_timesteps, algorithm, n_steps_per_timestep, obs_space_opts,
                     signal_plot_width, signal_plot_height,training_plot_width,training_plot_height, render_data,
                     stim_length_on_reset, data_proc_options,
                     use_unfiltered_np_data):
    ffminf=1000/ffminf #delay = 1000 ms/ n flashes per second
    ffmaxf=1000/ffmaxf
    ffinitf=1000/ffinitf
    lvals=[]
    for v in [l1c,l2c,l3c,l4c,l5c,l6c,l7c,l8c]:
        vals=list(map(int, v.split(',')))
        lvals+=vals
    
    w1sh=code_wave_shapes(w1sh)
    w2sh=code_wave_shapes(w2sh)
    
    session_settings_dict={}
    if 'Use absolute raw values in reward calculation' in data_proc_options:
        use_abs_values_for_raw_data_in_reward=True
    else:
        use_abs_values_for_raw_data_in_reward=False
    session_settings_dict['use_abs_values_for_raw_data_in_reward']=use_abs_values_for_raw_data_in_reward

    if 'Use directionality-agnostic encoder mode' in data_proc_options:
        only_pos_encoder_mode=True
    else:
        only_pos_encoder_mode=False
    session_settings_dict['only_pos_encoder_mode']=only_pos_encoder_mode
    if len(use_unfiltered_np_data)>0:
        use_unfiltered_np_data=True
    else:
        use_unfiltered_np_data=False


    session_settings_dict['n_input_channels']=17
    session_settings_dict['channels_of_interest']=channels_of_interest.split(',')
    session_settings_dict['n_timepoints_per_sample']=n_timepoints_per_sample
    session_settings_dict['delay']=delay
    #session_settings_dict['max_sfsystem_output']=max_sfsystem_output
    session_settings_dict['reward_formula_string']=formula_string
    session_settings_dict['use_unfiltered_np_data']=use_unfiltered_np_data
    binlist=fbins.split(';')
    fbins=[]
    for b in binlist:
        bin=b.split(',')
        bin=list(map(int, bin))
        bin=tuple(bin)
        fbins.append(bin)
    session_settings_dict['fbins']=fbins
    session_settings_dict['device_address']=device_address
    session_settings_dict['step_stim_length_millis']=step_stim_length_millis
    session_settings_dict['episode_time_seconds']=episode_time_seconds


    session_settings_dict['n_total_timesteps']=n_total_timesteps
    session_settings_dict['num_episodes']=num_episodes
    session_settings_dict['logfn']=logfn
    session_settings_dict['signal_plot_width']=signal_plot_width
    session_settings_dict['signal_plot_height']=signal_plot_height
    session_settings_dict['training_plot_width']=training_plot_width
    session_settings_dict['training_plot_height']=training_plot_height
    session_settings_dict['render_each_step']=True #no control, by default
    session_settings_dict['stim_length_on_reset']=stim_length_on_reset #no control, by default
    if len(render_data)>0:
        session_settings_dict['render_data']=True
    else:
        session_settings_dict['render_data']=False

    for i in logging_plotting_opts:
        if i=='Step data':
            session_settings_dict['log_steps']=True
        if i=='Episode data':
            session_settings_dict['log_episodes']=True
        if i=='Best episode actions':
            session_settings_dict['log_best_actions_every_episode']=True
        if i=='Each step actions':
            session_settings_dict['log_actions_every_step']=True
        if i=='Final best actions':
            session_settings_dict['log_best_actions_final']=True
        if i=='Raw data':
            session_settings_dict['write_raw']=True
        if i=='FFT results':
            session_settings_dict['write_fft']=True
        if i=='Bin values':
            session_settings_dict['write_bins']=True
        if i=='Models':
            session_settings_dict['log_model']=True
    session_settings_dict['log_or_plot_every_n_timesteps']=log_or_plot_every_n_timesteps
    session_settings_dict['algorithm']=algorithm

    if n_steps_per_timestep==0:
        n_steps_per_timestep=int((episode_time_seconds*1000)/step_stim_length_millis)+1
    session_settings_dict['n_steps_per_timestep']=n_steps_per_timestep
    if 'Raw signal values' in obs_space_opts:
        session_settings_dict['use_raw_in_os_def']=True
    else:
        session_settings_dict['use_raw_in_os_def']=False
    if 'Frequency spectra' in obs_space_opts:
        session_settings_dict['use_freq_in_os_def']=True
    else:
        session_settings_dict['use_freq_in_os_def']=False
    if 'Frequency bin values' in obs_space_opts:
        session_settings_dict['use_fbins_in_os_def']=True
    else:
        session_settings_dict['use_fbins_in_os_def']=False

 
    out_dict={'leddelay':{'names':['leddelay'], 'value_range':{'min':ffmaxf, 'max':ffminf}, 'init_val':{'leddelay':ffinitf}},
          'ledcontrols':{'names':['lv1r','lv1g','lv1b','lv2r','lv2g','lv2b','lv3r','lv3g','lv3b','lv4r','lv4g','lv4b', 'lv5r','lv5g','lv5b','lv6r','lv6g',
          'lv6b',
          'lv7r',
          'lv7g',
          'lv7b',
          'lv8r',
          'lv8g',
          'lv8b'], 'value_range':{'min':rgbrange[0], 'max':rgbrange[1]}, 'init_val':{'lv1r':lvals[0], 'lv1g':lvals[1], 'lv1b':lvals[2], 
                                                                               'lv2r':lvals[3], 'lv2b':lvals[4], 'lv2b':lvals[5], 
                                                                               'lv3r':lvals[6], 'lv3g':lvals[7], 'lv3b':lvals[8], 
                                                                               'lv4r':lvals[9], 'lv4g':lvals[10], 'lv4b':lvals[11], 
                                                                               'lv5r':lvals[12],   'lv5g':lvals[13], 'lv5b':lvals[14], 
                                                                               'lv6r':lvals[15], 'lv6g':lvals[16],   'lv6b':lvals[17], 
                                                                               'lv7r':lvals[18], 'lv7b':lvals[19], 'lv7b':lvals[20], 
                                                                               'lv8r':lvals[21], 'lv8g':lvals[22], 'lv8b':lvals[23]}},
          'sound_wave_frequencies':{'names':['wave_1_freq','wave_2_freq'], 'value_range':{'min':sound_wave_frange[0], 'max':sound_wave_frange[1]}, 'init_val':{'wave_1_freq':w1f, 
                                                                                                                                         'wave_2_freq':w2f}},
          'panner_phasor_frequencies':{'names':['panner_freq', 'phasor_1_freq', 'phasor_2_freq','phasor_1_min',  'phasor_2_min', 'phasor_1_dif', 'phasor_2_dif'],  'value_range':{'min':panner_phasor_frange[0], 'max':panner_phasor_frange[1]},
                                       'init_val':{'panner_freq':panner_freq,
                                                    'phasor_1_freq':phasor_1_freq,
                                                    'phasor_2_freq':phasor_2_freq,
                                                    'phasor_1_min':phasor_1_span[0],
                                                    'phasor_2_min':phasor_2_span[0],
                                                    'phasor_1_dif':phasor_1_span[1]-phasor_1_span[0],
                                                    'phasor_2_dif':phasor_2_span[1]-phasor_2_span[0]}},
          'panner_div':{'names':['panner_div'], 'value_range':{'min':panner_div_range[0], 'max':panner_div_range[1]}, 'init_val':{'panner_div':panner_div}},
          'sound_wave_shapes':{'names':['wave_1_type', 'wave_2_type'], 'value_range':{'min':0, 'max':3}, 
                               'init_val':{'wave_1_type':w1sh,
                                           'wave_2_type':w2sh}},
          'maxivolume':{'names':['maxivolume'], 'value_range':{'min':volrange[0], 'max':volrange[1]}, 
                        'init_val':{'maxivolume':maxivolume}}
}



    return {'out_dict':out_dict, 'session_settings':session_settings_dict}








