import dash
import dash_bootstrap_components as dbc
import RLSystem
from dash_extensions.enrich import html, dcc, Input, Output, State, ctx
from dash import callback

dash.register_page(__name__,'/')

#invis={'display':'none'}
#vis={'display':'inline-block'}
#d_vis={'color': 'Black', 'font-size': 20}
b_vis={"padding": "1rem 1rem", "margin-top": "2rem", "margin-bottom": "1rem", 'display':'inline-block'}
b_invis={"padding": "1rem 1rem", "margin-top": "2rem", "margin-bottom": "1rem", 'display':'none'}

layout=html.Div([
    dcc.Store(id='settings_dictionary',data=None),
    dbc.Row(justify="start", children=[dcc.Markdown("##### Audiovisual space setup"),
                      html.Hr(),
                      dbc.Col(width='auto',children=[            
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
                        dcc.Input(type='number', placeholder='Volume level', value=10, id='maxivolume')                                            
                                            
                                            
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
                                            dcc.Input(type='number', placeholder='Frequency, Hz', value=440, id='phasor_1_freq')]),
                        'Phasor 2 initial frequency span',
                        dcc.RangeSlider(min=1, max=30, step=1, marks=None, value=[1, 50], tooltip={"placement": "bottom", "always_visible": True},id='phasor_2_span'),
                          ]
                        ),
                      ]),
          html.Br(),
          dbc.Row(justify='start', children=[
            dcc.Markdown('##### Session settings'),
            html.Hr(),
            dbc.Col(width='auto', children=[
            html.Div(children=[
                'Total channel count: ',
                dcc.Input(type='number', placeholder='N channels', value=8, id='n_input_channels'),
                html.Br(),
                'Channels to observe (indexes, 0-based): ',
                dcc.Input(type='text', placeholder='Channels of interest idx0,...idxn', value='0,1,2,3,4,5,6,7', id='channels_of_interest_inds'),
                html.Br(),
                'N timepoints per sample: ',
                dcc.Input(type='number', placeholder='N points', value=100, id='n_timepoints_per_sample'),
                html.Br(),               
                'Delay between datapoints: ',
                dcc.Input(type='number', placeholder='Delay, ms', value=10, id='delay'), 
                html.Br(), 
                'Max ADS output: ',
                dcc.Input(type='number', placeholder='Value', value=1023, id='max_sfsystem_output'),
                html.Br(),               
                ]
            ),

            html.Div(children=[
                html.Br(),
                dcc.Markdown('Data processing and usage settings'),  
                html.Hr(),              
                'Reward formula: ',
                dcc.Input(type='text', placeholder='Bin values, Hz', value='(fbin_1_4_ch0+freq_30_ch0)/fbin_12_30_ch0', id='fbins', size='50'),  
                html.Br(),
                'Observational space data types: ',
                dcc.Checklist(options=['Raw signal values','Frequency spectra', 'Frequency bin values'], value=['Raw signal values','Frequency spectra', 'Frequency bin values'], id='observational_space_datatypes'),
                html.Br(),
                'Frequency bins to record: ',
                dcc.Input(type='text', placeholder='Bin values, Hz', value='0,1;1,4;4,8;8,12;12,30', id='fbins'),   
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
                dcc.Input(type='number', placeholder='Length, ms', value=60, id='episode_time_seconds'), 
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
                dcc.Input(type='number', placeholder='N steps', value=1, id='episode_time_seconds'), 
                html.Br(),
                'N timesteps per algorithm training episode: ',
                dcc.Input(type='number', placeholder='N timesteps', value=0, id='n_total_timesteps'), 
                html.Br(),
                '(to use the step count corresponding to the episode time set 0)',
                html.Br(),
                'N episodes: ',
                dcc.Input(type='number', placeholder='N episodes', value=5, id='num_episodes'), 
                html.Br(),


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
                'Training plot width: ',
                dcc.Slider(id='training_plot_width',min=500, max=5000, step=100, value=2000, marks=None, tooltip={"placement": "bottom", "always_visible": True, "template": "{value} px"}),
                'Training plot height: ',
                dcc.Slider(id='training_plot_height',min=50, max=5000, step=50, value=500, marks=None, tooltip={"placement": "bottom", "always_visible": True, "template": "{value} px"}),                
                'Signal plot width: ',
                dcc.Slider(id='signal_plot_width',min=500, max=5000, step=100, value=2000, marks=None, tooltip={"placement": "bottom", "always_visible": True, "template": "{value} px"}),  
                'Signal plot height: ',
                dcc.Slider(id='signal_plot_height',min=50, max=5000, step=50, value=1500, marks=None, tooltip={"placement": "bottom", "always_visible": True, "template": "{value} px"}),                  
                ])             
            ]),
            html.Div(children=[
            html.Button("Launch training session", id="start_session_train", style=b_vis, n_clicks=0),
            ' ',
            html.Button("Launch dynamic session without training", id="start_session_notrain", style=b_vis, n_clicks=0),
            ' ',
            html.Button("Launch static session with current settings", id="start_session_static", style=b_vis, n_clicks=0),])
          ]),                                        
]),

def code_wave_shapes(w):
    if w=='Noise':
        return 0
    if w=='Sine':
        return 1
    if w=='Square':
        return 2
    if w=='Triangle':
        return 3

@callback(Output('settings_dictionary', 'data'),
          
              Input('flash_frequency_lb','value'),
              Input('flash_frequency_ub','value'),
              Input('flash_frequency_ib','value'),
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

              

              prevent_initial_call=False)
def collect_settings(ffminf, ffmaxf, ffinitf, rgbrange,
                     l1c,l2c,l3c,l4c,l5c,l6c,l7c,l8c,sound_wave_frange,
                     w1f, w2f, w1sh, w2sh, volrange, panner_phasor_frange,panner_freq,
                     panner_div_range,panner_div, phasor_1_freq,phasor_2_freq,phasor_1_span,
                     phasor_2_span,maxivolume):
    ffminf=1000/ffminf #delay = 1000 ms/ n flashes per second
    ffmaxf=1000/ffmaxf
    ffinitf=1000/ffinitf
    lvals=[]
    for v in [l1c,l2c,l3c,l4c,l5c,l6c,l7c,l8c]:
        vals=list(map(int, v.split(',')))
        lvals+=vals
    
    w1sh=code_wave_shapes(w1sh)
    w2sh=code_wave_shapes(w2sh)


 
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
          'maxibolume':{'names':['maxivolume'], 'value_range':{'min':volrange[0], 'max':volrange[1]}, 
                        'init_val':{'maxivolume':maxivolume}}
}



    return 




















""" {'leddelay':{'names':['leddelay'], 'value_range':{'min':1, 'max':10001, 'step':100}, 'init_val':{'leddelay':10}},
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
          'maxibolume':{'names':['maxivolume'], 'value_range':{'min':0, 'max':50, 'step':10}, 
                        'init_val':{'maxivolume':10}}
}

self, out_dict=out_dict, out_order=out_order,n_input_channels=8, channels_of_interest_inds=list(range(8)), n_timepoints_per_sample=100, max_sfsystem_output=1023,reward_formula_string='(fbin_1_4_ch0+freq_30_ch0)/fbin_12_30_ch0', 
                 fbins=[(0,1), (1,4), (4,8), (8,12), (12,30)], delay=10,

                 use_raw_in_os_def=False, use_freq_in_os_def=False, use_fbins_in_os_def=False, device_address="ws://10.42.0.231:80/",
                 step_stim_length_millis=10000, episode_time_seconds=60, render_data=True, return_plotly_figs=False,
                 logfn='current_training.log', log_steps=True, log_episodes=True, log_best_actions_final=True, signal_plot_width=2000, signal_plot_height=1500, training_plot_width=2000, training_plot_height=500, 
                 write_raw=True,
                 write_fft=True,
                 write_bins=True,
                 log_best_actions_every_episode=True,
                 log_actions_every_step=True,
                 render_each_step=True """


