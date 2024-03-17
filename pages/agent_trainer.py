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


dash.register_page(__name__,'/')

#invis={'display':'none'}
#vis={'display':'inline-block'}
#d_vis={'color': 'Black', 'font-size': 20}
b_vis={"padding": "1rem 1rem", "margin-top": "2rem", "margin-bottom": "1rem", 'display':'inline-block'}
b_invis={"padding": "1rem 1rem", "margin-top": "2rem", "margin-bottom": "1rem", 'display':'none'}

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


layout=html.Div(
    [dcc.Store(id='run_type', data=None),
     dcc.Store(id='session_library', data=os.listdir('./session_lib')),
     #dbc.Row(justify="start", id='message_row', children=[]),
                 dbc.Row(justify="start", children=[dbc.Col(width=4, children=[
    dcc.Store(id='settings_dictionary',data=None),
    dcc.Interval(id='training_status_update', disabled=True, n_intervals=0, max_intervals=-1),
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
                      ]),
          html.Br(),
          dbc.Row(justify='start', children=[
            dcc.Markdown('##### Session settings'),
            html.Hr(),
            dbc.Col(width='auto', children=[
            html.Div(children=[
                'Total channel count: ',
                dcc.Input(type='number', placeholder='N channels', value=9, id='n_input_channels'),
                html.Br(),
                'Channels to observe (indexes, 0-based): ',
                dcc.Input(type='text', placeholder='Channels of interest idx0,...idxn', value='0,1,2,3,4,5,6,7,8', id='channels_of_interest_inds'),
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
                dcc.Input(type='text', placeholder='Formula string', value='raw_ch8', id='formula_string', size='50'),  
                ' ',
                dbc.Button("Help", id="open_formula_instructions", n_clicks=0),    
                html.Br(),
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
            dcc.Input(type='number', placeholder='interval, ms', value=1000, id='info_upd_interval', size=30),
            ' ',
            'Step count for a dynamic notrain session or a trained model run: ',
            dcc.Input(type='number', placeholder='N steps', value=360, id='n_steps_notrain', size=30),
            html.Br(),
            'Model location for upload (for the corresponding regimen): ',
            dcc.Input(type='text', placeholder='session name/model name', value='default_session/best_total_episode_reward_model.zip', id='model_name', size=30),]), 
            offcanvas_session_lib,
            html.Div(children=[
            html.Button("Run training session", id="start_session_train", style=b_vis, n_clicks=0),
            ' ',
            html.Button("Run dynamic session without training", id="start_session_notrain", style=b_vis, n_clicks=0),
            ' ',
            html.Button("Run static session", id="start_session_static", style=b_vis, n_clicks=0),
            ' ',
            html.Button("Stop training", id="stop_session_train", style=b_invis, n_clicks=0),
            ' ',
            html.Button("Run additional episodes", id="additional_session", style=b_invis, n_clicks=0),
            ' ',
            html.Button("Run trained model", id="run_trained", style=b_vis, n_clicks=0),
            html.Br(),
            'Additional options for running the trained model: ',
            dcc.Checklist(options=['Train the logged model with the original settings'], value=[], id='train_logged_orig'),
             dcc.Checklist(options=['Train the logged model with the new settings'], value=[], id='train_logged_new'), ])])
]),
dbc.Col(children=[dcc.Markdown("### Session Data"),
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
                html.Button("Clear trainer results to last 10 points", id="clear_trainer_data", style=b_vis, n_clicks=0),
                html.Br(),
                'Session name: ',
                dcc.Input(type='text', placeholder='Session name (old data, if present, will be overwritten)', value='default_session', id='session_name', size=30),]),
                html.Br(),
                dbc.Button("Show session data", id="open_plot_panel", n_clicks=0),    
                dbc.Offcanvas(children=[html.Br(),
                 dbc.Row(justify="start", id='message_row', children=[]),
                  html.Div(id='training_figure_container', children=[]),
                  html.Br(),
                  html.Div(id='signal_figure_container', children=[]),
                  html.Br(),],
                  id='plot_panel',
                  title='Session data',
                  is_open=False,
                  placement="end",
                  scrollable=True,
                  style={'width':'95%'},
                ),
                  
                  ])]      
          
)
          ])                                        

@callback(
    Output("clear_trainer_data", "n_clicks"),
    Input("clear_trainer_data", "n_clicks"),
    prevent_initial_call=True
)
def toggle_offcanvas_scrollable(n1):
    global env
    global trainer
    trainer.env.previous_episodes_max_rewards=trainer.env.previous_episodes_max_rewards[-10:]
    trainer.env.previous_episodes_total_rewards=trainer.env.previous_episodes_max_rewards[-10:]
    return n1

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
    

def copy_file(source_file_path, destination_folder_path):
    shutil.copy(source_file_path, destination_folder_path)
@callback(Output('session_library', 'data'),
          Input("move_data", 'n_clicks'),
          State('session_name', 'value'),
          prevent_initial_call=True)
def copy_session_logs_to_lib(n_clicks, sname):
    session_dir=f'./session_lib/{sname}'
    if os.path.isdir(session_dir):
        os.rmtree(session_dir)
    os.mkdir(session_dir)
    cfiles=os.listdir('./')
    for f in cfiles:
        if f in ['current_training.log', 'model_stats.log','best_total_episode_reward_model.zip','best_overall_reward_model.zip']:
            copy_file(f'./{f}', session_dir)
    return os.listdir('./session_lib')

@callback(Output('session_library', 'data', allow_duplicate=True),
          Input("clear_session_lib", 'n_clicks'),
          prevent_initial_call=True)
def clear_session_lib(n_clicks):
    os.rmtree('./session_lib')
    os.mkdir('./session_dir')
    return os.listdir('./session_lib')


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
    return ch



@callback(Output('training_figure_container', "children"),
          Output('signal_figure_container', "children"),
          Output('message_row', 'children'),
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
                f'Training completion: {trainer.training_completed}']            
    return training_fig, signal_fig, messages



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

          
          Input("start_session_train", "n_clicks"),
          Input("start_session_notrain", "n_clicks"),
          Input("start_session_static", "n_clicks"),
          Input("stop_session_train", "n_clicks"),
          Input("additional_session", "n_clicks"),
          Input("run_trained", "n_clicks"),



          State('settings_dictionary', 'data'),
          State('info_upd_interval', 'value'), 
          State('signal_plot_color','value'),
          State('n_steps_notrain','value'),
          State('model_name', 'value'),
          State('train_logged_orig', 'value'),
          State('train_logged_new', 'value'),
          prevent_initial_call=True)
def collect_settings(n_clicks_t, n_clicks_nt, n_clicks_static, n_clicks_stop, n_clicks_additional, n_clicks_run_trained, setd, info_upd_interval, sigplot_color, n_steps_notrain,
                     model_name, train_logged_orig, train_logged_new):
    global env
    global trainer
    trigger = ctx.triggered[0]
    trigger_id = trigger['prop_id'].split('.')[0]
    trigger_value = trigger['value']
    out_dict=setd['out_dict']
    sd=setd['session_settings']
    if sigplot_color.split(',')[0]=='random':
        sigplot_colors=get_random_css_color_names(sd['n_input_channels'], seed=int(sigplot_color.split(',')[1]))
    else:
        sigplot_colors=[sigplot_color for i in range(sd['n_input_channels'])]


    if trigger_id in ['start_session_train', "start_session_static", "start_session_notrain"]:
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
                                                log_actions_every_step=sd['log_actions_every_step'],
                                                stim_length_on_reset=sd['stim_length_on_reset'],
                                                only_pos_encoder_mode=sd['only_pos_encoder_mode'],
                                                use_abs_values_for_raw_data_in_reward=sd['use_abs_values_for_raw_data_in_reward'],
                                                colors=sigplot_colors)

                #print(sd)
            #print(out_dict)
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
    
    if trigger_id =='start_session_train':
          #print(sd)
          #print(out_dict)
          #env.step(env.action_space.sample()) #sample step
          #print(training_args)
          training_thread = threading.Thread(target=start_training, args=(training_args,))
          training_thread.daemon = True
          training_thread.start()
          return b_invis, b_invis, b_invis, b_vis, b_vis, False, info_upd_interval, b_invis, 'train'
    if trigger_id=="start_session_static":
          training_thread = threading.Thread(target=start_session_static)
          training_thread.daemon = True
          training_thread.start()
          return b_invis, b_invis, b_invis, b_vis, b_vis, False, info_upd_interval, b_invis, 'static'    
    if trigger_id=="start_session_notrain":
          training_thread = threading.Thread(target=start_session_notrain, args=({'n_steps_notrain':n_steps_notrain},))
          training_thread.daemon = True
          training_thread.start()
          return b_invis, b_invis, b_invis, b_vis, b_vis, False, info_upd_interval, b_invis, 'notrain'
    if trigger_id=="additional_session":
        #we collect training arguments in case they changed and run the same trainer
        #we do not initialize either new environment or new trainer
        training_args={
                'num_episodes':sd['num_episodes'],
                'log_model':sd['log_model'],
                'n_total_timesteps':sd['n_total_timesteps'],
                'log_or_plot_every_n_timesteps':sd['log_or_plot_every_n_timesteps'],
                'jnb':False
                
        }
        training_thread = threading.Thread(target=start_training, args=(training_args,))
        training_thread.daemon = True
        training_thread.start()
        return b_invis, b_invis, b_invis, b_vis, b_vis, False, info_upd_interval, b_invis, 'train'

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
                                              log_actions_every_step=sd['log_actions_every_step'],
                                              stim_length_on_reset=sd['stim_length_on_reset'],
                                              only_pos_encoder_mode=sd['only_pos_encoder_mode'],
                                              use_abs_values_for_raw_data_in_reward=sd['use_abs_values_for_raw_data_in_reward'],
                                              colors=sigplot_colors)    
        trainer=stable_baselines_model_trainer(initialized_environment=env,
                                                            algorithm=sd['algorithm'],
                                                            policy='MlpPolicy',
                                                            logfn='model_stats.log',
                                                            n_steps_per_timestep=sd['n_steps_per_timestep'])
        trainer.load_model(f'session_lib/{session_name}/{model_name}')
        if len(train_logged_orig)==0 and len(train_logged_new)==0:
            training_thread = threading.Thread(target=start_session_trained_model, args=({'n_steps_notrain':n_steps_notrain},))
            training_thread.daemon = True
            training_thread.start()
            return b_invis, b_invis, b_invis, b_vis, b_vis, False, info_upd_interval, b_invis, 'log'
        else:
            trainer.set_model_environment() #if we want to train, we need to connect the model to the environment
            training_args={
                'num_episodes':sd['num_episodes'],
                'log_model':sd['log_model'],
                'n_total_timesteps':sd['n_total_timesteps'],
                'log_or_plot_every_n_timesteps':sd['log_or_plot_every_n_timesteps'],
                'jnb':False
                
            }
            training_thread = threading.Thread(target=start_training, args=(training_args,))
            training_thread.daemon = True
            training_thread.start()

            return b_invis, b_invis, b_invis, b_vis, b_vis, False, info_upd_interval, b_invis, 'train'
    
    if trigger_id=="stop_session_train":
        try:
            trainer.close_env()
        except Exception as e:
            print(f"On session stop received: {e}")
            try:
                env.close()
            except Exception as e:
                print(f"On environment stop received: {e}")
        return b_vis, b_vis, b_vis, b_invis, b_invis, True, info_upd_interval, b_vis, 'stop'

          

def start_session_trained_model(arg):
    global env
    global trainer
    nsteps_notrain=arg['n_steps_notrain']
    while nsteps_notrain>0:
        n_notrain=nsteps_notrain
        try:
            obs=trainer.env.reset()[0]
            action, _states = trainer.model.predict(obs, deterministic=True)
            obs, reward, done, info = trainer.env.step(action)
            nsteps_notrain-=1
        except Exception as e:
            n_steps_notrain=0
            print(f"Training thread terminated: {e}")
            try:
                trainer.env.close()
            except Exception as e:
                print(f"On environment closure: {e}")
            break


         
def start_training(sd):
    global env
    global trainer
    try:
        trainer.train(num_episodes=sd['num_episodes'], log_model=sd['log_model'],n_total_timesteps=sd['n_total_timesteps'],
                            log_or_plot_every_n_timesteps=sd['log_or_plot_every_n_timesteps'], jnb=False)

    except Exception as e:
        print(f"Training thread terminated: {e}")

def start_session_static():
    global env
    global trainer
    try:
        trainer.static_launch()
    except Exception as e:
        print(f"Training thread terminated: {e}")
        try:
            trainer.env.close()
        except Exception as e:
            print(f"On environment closure: {e}")
def start_session_notrain(arg):
    global env
    global trainer
    nsteps_notrain=arg['n_steps_notrain']
    while nsteps_notrain>0:
        try:
            trainer.dynamic_launch()
            nsteps_notrain-=1
        except Exception as e:
            n_steps_notrain=0
            print(f"Training thread terminated: {e}")
            try:
                trainer.env.close()
            except Exception as e:
                print(f"On environment closure: {e}")
            break
            



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
              Input('stim_length_on_reset','value'),
              Input('data_proc_options', 'value'),
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
                     stim_length_on_reset, data_proc_options):
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








