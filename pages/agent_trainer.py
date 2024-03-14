import dash
import dash_bootstrap_components as dbc
from RLSystem import SFSystemCommunicator, stable_baselines_model_trainer
from dash_extensions.enrich import html, dcc, Input, Output, State, ctx
from dash import callback
import threading

dash.register_page(__name__,'/')

#invis={'display':'none'}
#vis={'display':'inline-block'}
#d_vis={'color': 'Black', 'font-size': 20}
b_vis={"padding": "1rem 1rem", "margin-top": "2rem", "margin-bottom": "1rem", 'display':'inline-block'}
b_invis={"padding": "1rem 1rem", "margin-top": "2rem", "margin-bottom": "1rem", 'display':'none'}
layout=html.Div([dbc.Row(justify="start", children=[dbc.Col(width=4, children=[
    dcc.Store(id='settings_dictionary',data=None),
    dcc.Interval(id='training_status_update', disabled=True, n_intervals=0, max_intervals=-1),
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
                                            dcc.Input(type='number', placeholder='Frequency, Hz', value=440, id='phasor_2_freq')]),
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
                'Data types to use in observational space: ',
                dcc.Dropdown(options=['Raw signal values','Frequency spectra', 'Frequency bin values'], value=['Raw signal values','Frequency spectra', 'Frequency bin values'], id='obs_space_opts', multi=True), 
                html.Br(),                
                ]
            ),

            html.Div(children=[
                html.Br(),
                dcc.Markdown('Data processing and usage settings'),  
                html.Hr(),              
                'Reward formula: ',
                dcc.Input(type='text', placeholder='Formula string', value='(fbin_1_4_ch0+freq_30_ch0)/fbin_12_30_ch0', id='formula_string', size='50'),  
                html.Br(),
                'Observational space data types: ',
                dcc.Checklist(options=['Raw signal values','Frequency spectra', 'Frequency bin values'], value=['Raw signal values','Frequency spectra', 'Frequency bin values'], id='observational_space_datatypes'),
                html.Br(),
                'Frequency bins to record: ',
                dcc.Input(type='text', placeholder='Bin values, Hz', value='0,1;1,4;4,8;8,12;12,30;30,50', id='fbins'),   
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
                dcc.Input(type='number', placeholder='N steps', value=1, id='n_steps_per_timestep'), 
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
                ])             
            ]),
            html.Br(),
            html.Div(children=['Session data update minimal interval: ',
            dcc.Input(type='number', placeholder='interval, ms', value=1000, id='info_upd_interval', size=30),]), 
            html.Div(children=[
            html.Button("Launch training session", id="start_session_train", style=b_vis, n_clicks=0),
            ' ',
            html.Button("Launch dynamic session without training", id="start_session_notrain", style=b_vis, n_clicks=0),
            ' ',
            html.Button("Launch static session with current settings", id="start_session_static", style=b_vis, n_clicks=0),
            ' ',
            html.Button("Stop training", id="stop_session_train", style=b_invis, n_clicks=0),
            ' ',
            html.Button("Run additional episodes", id="additional_session", style=b_invis, n_clicks=0) ])])
]),
dbc.Col(children=[dcc.Markdown("### Session Data"),
                  html.Div(id='training_figure_container', children=[]),
                  html.Br(),
                  html.Div(id='signal_figure_container', children=[]),
                  html.Br(),
                  
                  
                  ])]      
          
)
          ])                                        

def code_wave_shapes(w):
    if w=='Noise':
        return 0
    if w=='Sine':
        return 1
    if w=='Square':
        return 2
    if w=='Triangle':
        return 3
    
@callback(Output('training_figure_container', "children"),
          Output('signal_figure_container', "children"),
          Input('training_status_update', 'n_intervals'),
          prevent_initial_call=True)
def collect_settings(n_intervals):
    global env
    #print(env.figures)

    training_fig=dcc.Graph(id=f'training_figure',
                    figure=env.figures['training_fig'],
                    config={'staticPlot': False},)
    signal_fig=dcc.Graph(id=f'training_figure',
                    figure=env.figures['signal_fig'],
                    config={'staticPlot': False},)
                    
    return training_fig, signal_fig




@callback(Output("start_session_train", "style"),
          Output("start_session_notrain", "style"),
          Output("start_session_static", "style"),
          Output("stop_session_train", "style"),
          Output("additional_session", "style"),
          Output('training_status_update', 'disabled'),
          Output('training_status_update', 'interval'),

          
          Input("start_session_train", "n_clicks"),
          Input("start_session_notrain", "n_clicks"),
          Input("start_session_static", "n_clicks"),
          Input("stop_session_train", "n_clicks"),
          Input("additional_session", "n_clicks"),



          State('settings_dictionary', 'data'),
          State('info_upd_interval', 'value'), 
          prevent_initial_call=True)
def collect_settings(n_clicks_t, n_clicks_nt, n_clicks_static, n_clicks_stop, n_clicks_additional, setd, info_upd_interval):
    global env
    global trainer
    trigger = ctx.triggered[0]
    trigger_id = trigger['prop_id'].split('.')[0]
    trigger_value = trigger['value']
    out_dict=setd['out_dict']
    sd=setd['session_settings']
    if trigger_id =='start_session_train':
          #print(sd)
          #print(out_dict)
          env = SFSystemCommunicator(out_dict=out_dict,
                                              n_input_channels=sd['n_input_channels'],
                                              channels_of_interest_inds=sd['channels_of_interest_inds'],
                                              n_timepoints_per_sample=sd['n_timepoints_per_sample'],
                                              max_sfsystem_output=sd['max_sfsystem_output'],
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
                                              log_actions_every_step=sd['log_actions_every_step'])
          #env.step(env.action_space.sample()) #sample step
          trainer=stable_baselines_model_trainer(initialized_environment=env,
                                                          algorithm=sd['algorithm'],
                                                          policy='MlpPolicy',
                                                          logfn='model_stats.log',
                                                          n_steps_per_timestep=sd['n_steps_per_timestep'])
          

          training_args={
              'num_episodes':sd['num_episodes'],
              'log_model':sd['log_model'],
              'n_total_timesteps':sd['n_total_timesteps'],
              'log_or_plot_every_n_timesteps':sd['log_or_plot_every_n_timesteps'],
              'jnb':False
              
          }
          #print(training_args)
          training_thread = threading.Thread(target=start_training, args=(training_args,))
          training_thread.daemon = True
          training_thread.start()
          return b_invis, b_invis, b_invis, b_vis, b_vis, False, info_upd_interval
    if trigger_id=="stop_session_train":
        trainer.close_env()
        return b_vis, b_vis, b_vis, b_invis, b_invis, True, info_upd_interval

          

          



         
def start_training(sd):
    global env
    global trainer
    try:
        trainer.train(num_episodes=sd['num_episodes'], log_model=sd['log_model'],n_total_timesteps=sd['n_total_timesteps'],
                            log_or_plot_every_n_timesteps=sd['log_or_plot_every_n_timesteps'], jnb=False)

    except Exception as e:
        print(f"Training thread terminated: {e}")

                                              
      






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
              Input('n_input_channels','value'),
              Input('channels_of_interest_inds','value'),
              Input('n_timepoints_per_sample','value'),
              Input('delay','value'),
              Input('max_sfsystem_output','value'),
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
              prevent_initial_call=False)
def collect_settings(ffminf, ffmaxf, ffinitf, rgbrange,
                     l1c,l2c,l3c,l4c,l5c,l6c,l7c,l8c,sound_wave_frange,
                     w1f, w2f, w1sh, w2sh, volrange, panner_phasor_frange,panner_freq,
                     panner_div_range,panner_div, phasor_1_freq,phasor_2_freq,phasor_1_span,
                     phasor_2_span,maxivolume, 
                     n_input_channels, channels_of_interest_inds,n_timepoints_per_sample,
                     delay,max_sfsystem_output, formula_string, fbins, device_address,
                     step_stim_length_millis,episode_time_seconds,n_total_timesteps,
                     num_episodes,logfn,
                     logging_plotting_opts,
                     log_or_plot_every_n_timesteps, algorithm, n_steps_per_timestep, obs_space_opts,
                     signal_plot_width, signal_plot_height,training_plot_width,training_plot_height, render_data,
                     ):
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
    session_settings_dict['n_input_channels']=n_input_channels
    session_settings_dict['channels_of_interest_inds']=list(map(int, channels_of_interest_inds.split(',')))
    session_settings_dict['n_timepoints_per_sample']=n_timepoints_per_sample
    session_settings_dict['delay']=delay
    session_settings_dict['max_sfsystem_output']=max_sfsystem_output
    session_settings_dict['reward_formula_string']=formula_string
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

    if n_total_timesteps==0:
        n_total_timesteps='episode'
    session_settings_dict['n_total_timesteps']=n_total_timesteps
    session_settings_dict['num_episodes']=num_episodes
    session_settings_dict['logfn']=logfn
    session_settings_dict['signal_plot_width']=signal_plot_width
    session_settings_dict['signal_plot_height']=signal_plot_height
    session_settings_dict['training_plot_width']=training_plot_width
    session_settings_dict['training_plot_height']=training_plot_height
    session_settings_dict['render_each_step']=True #no control, by default
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








