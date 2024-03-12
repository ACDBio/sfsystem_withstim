import dash
import dash_bootstrap_components as dbc
import RLSystem
from dash_extensions.enrich import html, dcc, Input, Output, State, ctx
dash.register_page(__name__,'/')
layout=html.Div([
    dbc.Row(justify="start", children=[dcc.Markdown("Audiovisual space setup"),
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
        dbc.Row(justify="start", children=[dcc.Markdown("Sound space setup"),
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
                                            dcc.RadioItems(['Noise','Sine', 'Square', 'Triangle'], 'Noise')]),
                        html.Div(['Wave 2 initial shape: ', 
                                            dcc.RadioItems(['Noise','Sine', 'Square', 'Triangle'], 'Noise')]),
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
                      ])
                      
                      
                      
                      
                      
                      
]),

            




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


