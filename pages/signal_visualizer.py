import dash
import numpy as np
import pandas as pd
import os
import base64
from dash_extensions.enrich import html, dcc, Input, Output, State, ctx
import plotly
import json
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import matplotlib
matplotlib.use('SVG')
#from dash.dependencies import Input, Output, State
import threading
import queue
from dash_extensions import WebSocket
import plotly.graph_objs as go
from datetime import datetime
from tqdm.autonotebook import tqdm
from dash import callback
import dash_bootstrap_components as dbc

import threading
import queue

dash.register_page(__name__)
# app = dash.Dash(__name__,prevent_initial_callbacks='initial_duplicate', 
#                 external_stylesheets=[
#     {
#         'href': 'https://cdn.plot.ly/plotly-basic-1.62.2.min.css',
#         'rel': 'stylesheet',
#         'type': 'text/css'
#     },
# ]
#)

plotly.io.json.config.default_engine = 'orjson'
thresh_fft_n_samples=30
logfn='signals.log'
if not os.path.isfile(logfn):
    open(logfn, 'a').close()



device_server_address="ws://10.42.0.231:80/"

n_channels=8
signal_chnames=[f'ch{i}' for i in range(1, n_channels+1)]
signal_chnames_zindexed=[f'ch{i}' for i in range(n_channels)]
log_queue = queue.Queue()
def log_worker():
    while True:
        message = log_queue.get()
        if message is None:
            break
        # Serialize message to JSON and write to the log file
        with open('signals.log', 'a') as log_file:
            json_message = json.dumps(message)
            log_file.write(json_message + '\n')

# Start the logging thread
logging_thread = threading.Thread(target=log_worker)
logging_thread.daemon = True # Daemonize thread
logging_thread.start()
def div_graph_raw(name):
    return html.Div(
        className="row",
        children=[
            html.Div(id=f"div-{name}-raw_graph", className="ten columns",
            children=[dcc.Graph(
            id=f'{name}_raw_graph',
            figure={'layout': {#'title': f'Raw Signal History Graph for {name}',
                               #'margin': dict(l=50, r=10, t=50, b=200),
                               'barmode': 'overlay',
                                       'xaxis': {
                                        'title': "Timestep"
                                        },
                                        'yaxis': {
                                            'title': f"",
                                            'rangemode': 'auto'
                                                }},
                    'data': [{'x': [0], 'y': [0]},]
                    }),
                    ],
                style=invis)
        ]
    )


def div_graph_fft(name):
    return html.Div(
        className="row",
        children=[
            html.Div(id=f"div-{name}-fft_graph", className="ten columns",
            children=[dcc.Graph(
            id=f'{name}_fft_graph',
            figure={'layout': {#'title': f'Raw Signal History Graph for {name}',
                               #'margin': dict(l=50, r=10, t=50, b=200),
                               'barmode': 'overlay',
                                       'xaxis': {
                                        'title': "Frequency (Hz)"
                                        },
                                        'yaxis': {
                                            'title': f"",
                                            'rangemode': 'auto'
                                                }},
                    'data': [{'x': [0], 'y': [0]},]
                    }),
                    ],
                style=invis)
        ]
    )


def div_graph_fft_bar(name):
    return html.Div(
        className="row",
        children=[
            html.Div(id=f"div-{name}-fft_bar_graph", className="ten columns",
            children=[dcc.Graph(
            id=f'{name}_fft_bar_graph',
            figure={'layout': {#'title': f'Raw Signal History Graph for {name}',
                               #'margin': dict(l=50, r=10, t=50, b=200),
                               'barmode': 'overlay',
                                       'xaxis': {
                                        'title': "Frequency Bins"
                                        },
                                        'yaxis': {
                                            'title': f"",
                                            'rangemode': 'nonnegative'
                                                }},
                    'data': [{'x': [], 'y': [], 'type':'bar'}],
                    }),
                    ],
                style=invis)
        ]
    )


def div_graph_fastupdate(name):
    return html.Div(
        className="row",
        children=[
            html.Div(id=f"div-{name}-fagraph", className="ten columns",
            children=[dcc.Graph(
            id=f'{name}_fagraph',
            figure={'layout': {#'title': f'Raw Signal History Graph for {name}',
                               'margin': dict(l=50, r=10, t=50, b=200),
                               'barmode': 'overlay',
                                       'xaxis': {
                                        'title': "Timepoint"
                                        },
                                        'yaxis': {
                                            'title': f"",
                                            'rangemode': 'auto'
                                                }},
                    'data': [{'x': [datetime.now().strftime("%H:%M:%S:%f")], 'y': [0]},]
                    }),
                    ],
                style=invis)
        ]
    )

def div_graph_fastupdate_bins(name):
    return html.Div(
        className="row",
        children=[
            html.Div(id=f"div-{name}-fagraph_bins", className="ten columns",
            children=[dcc.Graph(
            id=f'{name}_fagraph_bins',
            figure={'layout': {#'title': f'Raw Signal History Graph for {name}',
                               'margin': dict(l=50, r=10, t=50, b=200),
                               'barmode': 'overlay',
                                       'xaxis': {
                                        'title': "Timepoint"
                                        },
                                        'yaxis': {
                                            'title': f"",
                                            'rangemode': 'auto'
                                                }},
                    'data': [{'x': [datetime.now().strftime("%H:%M:%S:%f")], 'y': [], 'name':'bin 1'},
                             {'x': [datetime.now().strftime("%H:%M:%S:%f")], 'y': [], 'name':'bin 2'},
                             {'x': [datetime.now().strftime("%H:%M:%S:%f")], 'y': [], 'name':'bin 3'},
                             {'x': [datetime.now().strftime("%H:%M:%S:%f")], 'y': [], 'name':'bin 4'},
                             {'x': [datetime.now().strftime("%H:%M:%S:%f")], 'y': [], 'name':'bin 5'},
                             {'x': [datetime.now().strftime("%H:%M:%S:%f")], 'y': [], 'name':'bin 6'},]
                    }),
                    ],
                style=invis)
        ]
    )

def div_graph_fastupdate_spec():
    return html.Div(
        className="row",
        children=[
            html.Div(id=f"div-fagraph_spec", className="ten columns",
            children=[dcc.Graph(
            id=f'fagraph_spec',
            figure={'layout': {#'title': f'Raw Signal History Graph for {name}',
                               'margin': dict(l=50, r=10, t=50, b=200),
                               'barmode': 'overlay',
                                       'xaxis': {
                                        'title': "Timepoint"
                                        },
                                        'yaxis': {
                                            'title': f"",
                                            'rangemode': 'auto'
                                                }},
                    'data': [{'x': [datetime.now().strftime("%H:%M:%S:%f")], 'y': [], 'name':'ch 1'},
                             {'x': [datetime.now().strftime("%H:%M:%S:%f")], 'y': [], 'name':'ch 2'},
                             {'x': [datetime.now().strftime("%H:%M:%S:%f")], 'y': [], 'name':'ch 3'},
                             {'x': [datetime.now().strftime("%H:%M:%S:%f")], 'y': [], 'name':'ch 4'},
                             {'x': [datetime.now().strftime("%H:%M:%S:%f")], 'y': [], 'name':'ch 5'},
                             {'x': [datetime.now().strftime("%H:%M:%S:%f")], 'y': [], 'name':'ch 6'},
                             {'x': [datetime.now().strftime("%H:%M:%S:%f")], 'y': [], 'name':'ch 7'},
                             {'x': [datetime.now().strftime("%H:%M:%S:%f")], 'y': [], 'name':'ch 8'},]
                    }),
                    ],
                style=invis)
        ]
    )


def div_graph_fheatmap():
    figure={
                 'data': [go.Heatmap(
                      z=np.zeros(shape=(1,1)),
                      colorscale='Viridis'
                 )],
                 'layout': go.Layout(
                      height=700,
                      xaxis=dict(
                    title='Frequency'
                    ),
                    yaxis=dict(
                        title='Channel'
                    )
                      
                 )
              }
    return html.Div(
        className="row",
        children=[
            html.Div(id=f"div-hmgraph", className="ten columns",
            children=[dcc.Graph(
            id=f'hmgraph',
            figure=figure,
            config={'staticPlot': False},),
                    ],
                style=invis)
        ]
    )

def div_graph_fheatmap_max():
    figure={
                 'data': [go.Heatmap(
                      z=np.zeros(shape=(1,1)),
                      colorscale='Viridis'
                 )],
                 'layout': go.Layout(
                      height=700,
                      xaxis=dict(
                    title='Timestep'
                    ),
                    yaxis=dict(
                        title='Channel'
                    )
                      
                 )
              }
    return html.Div(
        className="row",
        children=[
            html.Div(id=f"div-hmgraph_max", className="ten columns",
            children=[dcc.Graph(
            id=f'hmgraph_max',
            figure=figure,
            config={'staticPlot': False},),
                    ],
                style=invis)
        ]
    )

def div_graph_fheatmap_spec():
    figure={
                 'data': [go.Heatmap(
                      z=np.zeros(shape=(1,1)),
                      colorscale='Viridis'
                 )],
                 'layout': go.Layout(
                      height=700,
                      xaxis=dict(
                    title='Timestep'
                    ),
                    yaxis=dict(
                        title='Channel'
                    )
                      
                 )
              }
    return html.Div(
        className="row",
        children=[
            html.Div(id=f"div-hmgraph_spec", className="ten columns",
            children=[dcc.Graph(
            id=f'hmgraph_spec',
            figure=figure,
            config={'staticPlot': False},),
                    ],
                style=invis)
        ]
    )


def div_graph_rawheatmap():
    figure={
                 'data': [go.Heatmap(
                      z=np.zeros(shape=(1,1)),
                      colorscale='Viridis'
                 )],
                 'layout': go.Layout(
                      height=700,
                      xaxis=dict(
                    title='Timestep'
                    ),
                    yaxis=dict(
                        title='Channel'
                    )
                      
                 )
              }
    return html.Div(
        className="row",
        children=[
            html.Div(id=f"div-hmgraph_raw", className="ten columns",
            children=[dcc.Graph(
            id=f'hmgraph_raw',
            figure=figure,
            config={'staticPlot': False},),
                    ],
                style=invis)
        ]
    )

def div_graph(name):
    return dbc.Row(html.Div(
       # className="row",
        id=f"div-{name}-graphs_container",
        children=[dcc.Markdown(f'### {" ".join(name.upper().split("_"))}'),
                  div_graph_fastupdate(name),
                  div_graph_fastupdate_bins(name),
                  div_graph_raw(name),
                  div_graph_fft(name),
                  div_graph_fft_bar(name),
                  html.Br()],
        style=invis))

def get_time():
    now=datetime.now()
    formatted_time = now.strftime("%Y:%m:%d:%H:%M:%S:%f")[8:-3]
    return formatted_time
def get_bin_values(f_plot, frequency_magnitudes, tar_freqs=[(0.5, 4), (30, 50)], nontar_freqs=[(10,20)], return_sep=False, return_ratio=False):
    fpl=np.array(f_plot)
    xmp=np.array(frequency_magnitudes)
    magnitudes=[]
    for low, high in tar_freqs:
            mask = (fpl >= low) & (fpl < high)
            magnitude = np.abs(xmp[mask]).mean()
            magnitudes.append(magnitude)
    if return_ratio==True:
        if nontar_freqs!=[]:
            nontar_magnitudes=[]
            for low, high in nontar_freqs:
                    mask = (fpl >= low) & (fpl < high)
                    nontar_magnitude = np.abs(xmp[mask]).mean()
                    nontar_magnitudes.append(nontar_magnitude)
            tar_magnitudes_m=np.array(magnitudes).mean()
            nontar_magnitudes_m=np.array(nontar_magnitudes).mean()
            #Ratio here will be difference
            ratio=tar_magnitudes_m-nontar_magnitudes_m
        else:
            ratio=np.array(magnitudes).mean()
        return ratio
    if return_sep==True:
        return magnitudes
    else:
        return np.array(magnitudes).mean()

#STYLE SETTINGS
invis={'display':'none'}
vis={'display':'inline-block'}
d_vis={'color': 'Black', 'font-size': 20}
b_vis={"padding": "1rem 1rem", "margin-top": "2rem", "margin-bottom": "1rem", 'display':'inline-block'}
b_invis={"padding": "1rem 1rem", "margin-top": "2rem", "margin-bottom": "1rem", 'display':'none'}
layout = html.Div(children=[
                    WebSocket(url=device_server_address, id="ws"),
                    dcc.Interval(id='logplots_upd_interval', disabled=True, n_intervals=0, max_intervals=1),
                    html.Div(id='message', children=[]),
                    dcc.Checklist(id='show_message', options=['Show info messages'], value=['Show info messages']),
                    dcc.Store(id='logged_df', data=False),
                    dcc.Store(id='signal_acquistion_options_data', data=None),

                    dcc.Store(id='start_websock_communication_switch', data=False),
                    dcc.Store(id='start_log_plotting_switch', data=False),


                    dcc.Store(id='do_fft', data=False),
                    dcc.Store(id='log_fft', data=False),
                    dcc.Store(id='do_bins', data=False),
                    dcc.Store(id='log_bins', data=False),
                    dcc.Store(id='log_signal', data=False),
                    dcc.Store(id='do_maxfreq_buffer', data=False),
                    dcc.Store(id='do_specfreq_buffer', data=False),
                    dcc.Store(id='targetnontarget_freqs', data=False),
                    dcc.Store(id='n_indiv_samples_to_store', data=False),

                    dcc.Store(id='current_split_logdata', data=False),

                    dcc.Store(id='logging_counter', data=False),

                    dcc.Store(id='get_all_channel_data', data=False),

                    dcc.Store(id='logged_data_timestamps', data=False),

                    html.Div([dcc.Store(id=f'{ch}_raw_buffer', data=[]) for ch in signal_chnames]), 
                    html.Div([dcc.Store(id=f'{ch}_fft_buffer', data=[]) for ch in signal_chnames]), 
                    html.Div([dcc.Store(id=f'{ch}_raw_current', data=[]) for ch in signal_chnames]),
                    html.Div([dcc.Store(id=f'{ch}_fft_current', data=[]) for ch in signal_chnames]), 
                    html.Div([dcc.Store(id=f'{ch}_bins_buffer', data=[]) for ch in signal_chnames]), 
                    html.Div([dcc.Store(id=f'{ch}_bins_current', data=[]) for ch in signal_chnames]), 
                    html.Div([dcc.Store(id=f'{ch}_maxfreq_buffer', data=[]) for ch in signal_chnames]),
                    html.Div([dcc.Store(id=f'{ch}_specfreq_current', data=[]) for ch in signal_chnames]),
                    html.Div([dcc.Store(id=f'{ch}_specfreq_buffer', data=[]) for ch in signal_chnames]),

                    html.Br(),
                    html.Div(id='main_gui', style=vis, children=[
                    dcc.Markdown(children='''#### SIGNAL VISUALIZER'''),
                    dcc.Markdown(children="Connetion not established...", id='connection_status', style=vis),
                    html.Br(),
                    dcc.RadioItems(['Plot signals', 'Explore logged data'], 'Plot signals', id='task_launcher', style=vis),
                    html.Br(),
                    html.Div(children=[
                    html.Div(id='model_launcher', style=invis, children=[
                        html.Button("Load feedback program model", id="feedback_model_upload", style=b_vis, n_clicks=0),
                        html.Button("Create new feedback program model", id="feedback_model_creation", style=b_vis, n_clicks=0),
                        html.Button("Save feedback program model", id="feedback_model_saving", style=b_invis, n_clicks=0),]),
                    
                    
                    html.Div(id='logdata_panel', style=invis, 
                             children=[
                                 html.Div(id='logdata_upload_panel', style=vis, children=[html.Button("Load current logging file", id="upload_current_logging_file", n_clicks=0, style=b_vis),
                                                                               dcc.Upload(
                                                                                            id='upload_logdata',
                                                                                            children=html.Div([
                                                                                                'Drag and Drop or ',
                                                                                                html.A('Select the Logging File')
                                                                                            ]),
                                                                                            style={
                                                                                                'width': '100%',
                                                                                                'height': '60px',
                                                                                                'lineHeight': '60px',
                                                                                                'borderWidth': '1px',
                                                                                                'borderStyle': 'dashed',
                                                                                                'borderRadius': '5px',
                                                                                                'textAlign': 'center',
                                                                                                'margin': '10px'
                                                                                            },
                                                                                            multiple=False
                                                                                        ),
                                                                                dcc.Checklist(options=['Save parsing results as separate dataframes'], id='save_dfs', value=[]),]),
                                                                                html.Br(),
                                                                                html.Div(children=['Note: This section uses raw data from the log file for spectral analyses. It uses timestep information, not timestamps for plotting. Thus, current time will be plotted.']),
                                                                                ]),
                                html.Br(),
                                html.Div(id='logdata_plotting_control_panel', style=invis, children=[
                                    html.Br(),
                                    dcc.Markdown('Logging data interval to plot:'),
                                    dcc.RangeSlider(id='logdata_range_slider',min=0, max=1, step=1, marks=None, value=[0, 1], tooltip={"placement": "bottom", "always_visible": True, "template": "Timestep {value}"}),
                                    html.Br(),
                                    html.Div(id='time_info', children=[]),
                                    html.Br(),
                                    dcc.Markdown('Data portion size per processing step:'),
                                    dcc.Input(id='signal_processing_step',
                                    type="number",
                                    min=0,
                                    step=1,
                                    value=100,
                                    placeholder='N points:'),
                                    html.Br(),
                                    dcc.Checklist(options=['Plot data for the whole range as a single step'], id='signal_processing_step_is_whole_range', value=[]),
                                    html.Br(),
                                    dcc.Markdown('Logging data portion processing interval:'),
                                    dcc.Slider(id='logdata_processing_interval', min=10, max=5000, step=10, value=1000, marks=None, tooltip={"placement": "bottom", "always_visible": True, "template": "{value} ms"}),
                                    html.Br(),
                                ]),


















                    html.Div(id='signal_plotting', style=invis, children=[
                        html.Br(),
                        dcc.Checklist(
                        options=['Raw signal lineplots', 
                                 'Signal spectra lineplots', 
                                 'Signal spectra barplots', 
                                 'Raw signal history lineplots',
                                 'Signal spectra history lineplots',

                                 'Signal spectra joint heatmaps (all channels)',
                                 'Peak frequency history heatmaps (all channels)',

                                 'Specific frequency history heatmaps (all channels)',
                                 'Specific frequency history lineplots (all channels)',
                                 'Raw signal history heatmaps (all channels)'],
                        value=[],
                        id='signal_visualization_options',
                        labelStyle={'display': 'block'},
                        inline=True,
                        style={"width":1000, "overflow":"auto"}),
                    ]),],
                    id='post_connection_menu',
                    style=invis,
                    ), 
                    html.Br(),
                    html.Div(id='data_transmission_rates_panel', style=invis,
                        children=[
                    html.Br(),
                    dcc.Markdown("Signal plotting update interval, point count:"),
                    dcc.Input(id="signal_plotting_update_interval",
                            type="number",
                            min=0,
                            step=1,
                            value=100,
                            placeholder='N points:'),
                    html.Br(), 
                    dcc.Markdown("Signal buffer update interval, point count:"),
                    dcc.Input(id="signal_buffer_update_interval",
                            type="number",
                            min=0,
                            step=1,
                            value=100,
                            placeholder='N points:'),
                    html.Br(),]
                    ),
                    html.Br(),
                    html.Div(id='tf_container', style=vis, children = [
                    dcc.Markdown("Target frequency difference:"),
                    dcc.Markdown("Can be given as target freq bins/non-target freq bins or just target freq bins"),
                    dcc.Input(id="target_frequency_eqation",
                            type="text",
                            value="0.5-4,30-40/10-20",
                            style={'width': '50%'},
                            placeholder='Frequency equation:'),]),
                    html.Br(),
                    html.Div(id='data_logging_options_panel',style=invis, children = [
                    html.Br(),                                  
                    dcc.Checklist(
                        options=['Log data'],
                        value=[],
                        id='log_data',
                        labelStyle={'display': 'block'},
                        inline=True,
                        style={"width":1000, "overflow":"auto"}),
                    # html.Br(),   
                    # dcc.Input(id="logfile_name",
                    #         type="text",
                    #         min=0,
                    #         step=1,
                    #         value='signals.log',
                    #         placeholder='Logfile name:', style=invis),
                    html.Br(),   
                    html.Br(),  
                    html.Div(id='logging_opts',style=invis, children=[  
                    html.Div(id='raw_logging', style=vis, children=[
                    dcc.Checklist(
                        options=['Log all channels'],
                        value=[],
                        id='log_all_channels',
                        labelStyle={'display': 'block'},
                        inline=True,
                        style={"width":1000, "overflow":"auto"}),
                    html.Br(),
                    dcc.Checklist(
                        options=['Log raw data'],
                        value=['Log raw data'],
                        id='log_raw_data',
                        labelStyle={'display': 'block'},
                        inline=True,
                        style={"width":1000, "overflow":"auto"}),                        
                      ]),
                    html.Br(), 
                    html.Div(id='raw_logging_options', style=invis, children=[
                    dcc.Markdown("Max raw signal points per log (leave -1 for unlimited):"),
                    dcc.Input(id="max_points_per_log",
                            type="number",
                            min=-1,
                            step=1,
                            value=-1,
                            placeholder='Max N points:'),
                            ]),
                    html.Br(),
                    html.Br(),    
                    dcc.Checklist(
                        options=['Log signal spectra'],
                        value=[],
                        id='log_fft_checkbox',
                        labelStyle={'display': 'block'},
                        inline=True,
                        style={"width":1000, "overflow":"auto"}), 
                    html.Br(),    
                    dcc.Checklist(
                        options=['Log frequency bins'],
                        value=[],
                        id='log_bins_checkbox',
                        labelStyle={'display': 'block'},
                        inline=True,
                        style={"width":1000, "overflow":"auto"}), 
                    html.Br(),    
                    # dcc.Checklist( #HERE - COMMENTED, BECAUSE PARSING NOT INTRODUCED
                    #     options=['Log specific frequency'],
                    #     value=[],
                    #     id='log_specfreq',
                    #     labelStyle={'display': 'block'},
                    #     inline=True,
                    #     style={"width":1000, "overflow":"auto"}),
                    # html.Br(),    
                    # dcc.Checklist(
                    #     options=['Log peak frequency'],
                    #     value=[],
                    #     id='log_peakfreq',
                    #     labelStyle={'display': 'block'},
                    #     inline=True,
                    #     style={"width":1000, "overflow":"auto"}),
                    html.Button("Clear the log file...", id="clear_log_button", n_clicks=0, style=b_vis),
                        ],
                    ), ]),
                        
                    html.Div(id="frequency_bar_selection_panel",
                            children=[dcc.RangeSlider(id='frequency_bin_selector', min=0, max=50, value=[0,0.5, 4, 8, 12, 30, 50], pushable=2, step=0.1, marks=None, tooltip={"placement": "bottom", "always_visible": True, "template": "{value} Hz"}),
                            html.Br(),
                            dcc.Checklist(
                            options=['Auto EEG ranges setup'],
                            value=['Auto EEG ranges setup'],
                            id='eegranges',
                            labelStyle={'display': 'block'},
                            style={"width":1000, "overflow":"auto"}),
                            html.Div(id='bin_description', children=[]
                                     ),
                                     html.Br()],
                            style=invis),
                    html.Br(),
                    html.Div(id='signal_acquistion_options_panel', style=invis, children=[
                            dcc.Markdown('Data channel selection'),
                            dcc.Dropdown(id='channel_selector',
                                        options=signal_chnames,
                                        value=[],
                                        multi=True), 
                            html.Div(id='ads_sampling_opts', style=invis,      children=[         
                            html.Br(),
                            dcc.Markdown('ADS sampling options'),
                            html.Br(),
                            html.Div('Sampling frequency'),
                            html.Br(),
                            dcc.RangeSlider(
                                        id='sampling_frequency_selector',
                                        min=1,
                                        max=1000,
                                        step=1,
                                        marks={
                                            1: '1 Hz',
                                            250: '250 Hz',
                                            500: '500 Hz',
                                            750: '750 Hz',
                                            1000: '1000 Hz',
                                        },
                                        tooltip={"placement": "bottom", "always_visible": True, "template": "{value} Hz"},
                                        value=[100]
                                    ), 
                            html.Br(),
                            html.Div('Sample count'),
                            html.Br(),
                            dcc.RangeSlider(
                                        id='sample_count_per_iter_selector',
                                        min=1,
                                        max=1000,
                                        step=1,
                                        marks={
                                            1: '1 sample',
                                            250: '250 samples',
                                            500: '500 samples',
                                            750: '750 samples',
                                            1000: '1000 samples',
                                        },
                                        tooltip={"placement": "bottom", "always_visible": True, "template": "{value} samples"},
                                        value=[100]
                            ),
                            html.Br(),]),
                            html.Br(),
                            html.Div(id='sampling_stats_info', children=[]),],
                    ),
                    html.Br(),
                    html.Button("Start live update...", id="live_update_button", n_clicks=0, style=b_invis),
                    html.Button("Plot selected data...", id="update_logplots_button", n_clicks=0, style=b_invis),
                    html.Button("Clear plotted data (if present)", id="clear_plots_button", n_clicks=0, style=b_invis),
                    html.Br(),
                    html.Div(id='server_msg', style=invis, children=[dcc.Markdown(children='Live updates server message:'),html.Div(id='message_text')]),
                    html.Br(),
                    html.Div(id='signal_joint_plot_container', children=[div_graph_fheatmap(), div_graph_fheatmap_max(), div_graph_fheatmap_spec(), div_graph_fastupdate_spec(), div_graph_rawheatmap()]),
                    html.Br(),
                    html.Div(id='signal_plot_container', style=invis, children=[div_graph(chname) for chname in signal_chnames]),
                    ])
])

@callback(Output('message', 'style'),
          Input('show_message', 'value'))
def show_messages(val):
    if len(val)>0:
     return vis
    else:
        return invis

@callback(Output('bin_description', 'children'),
              Input('frequency_bin_selector','value'),
              prevent_initial_call=False)
def set_raw_signal_logger(val): 
    strings=[dcc.Markdown('Bin legend:')]
    ind=1
    for low, high in zip(val[:-1], val[1:]):
        cs=f'{ind}. {low}-{high} Hz'
        strings.append(dcc.Markdown(cs))
        ind+=1

    return strings

@callback(Output('raw_logging_options', 'style'),
              Input('log_raw_data','value'),
              prevent_initial_call=False)
def set_raw_signal_logger(val): 
    if len(val)>0:
        return vis
    else:
        return invis

@callback(Output('tf_container', 'style'),
              #Input('log_specfreq','value'),
              Input('signal_visualization_options','value'),
              prevent_initial_call=False)
def set_raw_signal_logger(sv):  #(log, sv)
    # if len(log)>0:
    #     return vis
    if 'Specific frequency history heatmaps (all channels)' in sv:
        return vis
    if 'Specific frequency history lineplots (all channels)' in sv:
        return vis
    else:
        return invis



@callback(Output('message', 'children', allow_duplicate=True),
              Input("clear_log_button",'n_clicks'),
              prevent_initial_call=True)
def set_raw_signal_logger(n_clicks): 
    if os.path.isfile(logfn):
        os.remove(logfn)
        if not os.path.isfile(logfn):
            open(logfn, 'a').close()
    return "Log file cleared..."

@callback(Output('frequency_bar_selection_panel','style'),
              Input('log_bins_checkbox', 'value'),
              Input('signal_visualization_options','value'),
              allow_duplicate=True,
              prevent_intial_call=True)
def set_raw_signal_logger(log_bins, visopts):
    vs={"width":1000, "overflow":"auto", 'height':400}
    if len(log_bins)>0:
        return vs
    if 'Signal spectra barplots' in visopts:
        return vs
    if 'Signal spectra history lineplots' in visopts:
        return vs
    else:
        return invis
    
@callback(Output('logging_opts','style'),
              Input('log_data', 'value'),)
def set_raw_signal_logger(log_data):
    if len(log_data)>0:
        return vis
    else:
        return invis

@callback(Output('logging_counter','data', allow_duplicate=True),
              Input('signal_buffer_update_interval', 'value'),
              prevent_initial_call=True)
def set_raw_signal_logger(interval):
    return interval
    

@callback(Output('n_indiv_samples_to_store','data'),
              Input('signal_buffer_update_interval', 'value'),
              Input('signal_acquistion_options_data', 'data'),
              allow_duplicate=True,
              prevent_initial_call=True)
def set_raw_signal_logger(buf_upd_interval, sigaq_opts):
    n_samples_per_iter=sigaq_opts['n_samples']
    return n_samples_per_iter*buf_upd_interval



@callback(Output('targetnontarget_freqs', 'data'),
              Input("target_frequency_eqation",'value'))
def set_tarnontarfreqs(eq):
    tar=eq.split('/')[0]
    nontar=eq.split('/')[1]
    tar_bins=tar.split(',')
    tar_bins_res=[]
    for bin in tar_bins:
            b=bin.split('-')
            b[0]=float(b[0])
            b[1]=float(b[1])
            tar_bins_res.append(tuple(b))

    if len(nontar)>0:
        nontar_bins_res=[]
        nontar_bins=nontar.split(',')
        for bin in nontar_bins:
            b=bin.split('-')
            b[0]=float(b[0])
            b[1]=float(b[1])
            nontar_bins_res.append(tuple(b))
    else:
        nontar=[]  
    return {'tar':tar_bins_res, 'nontar':nontar_bins_res}

@callback(Output('get_all_channel_data', 'data'),
              Input('log_all_channels','value'),
              Input('signal_visualization_options','value'))
def set_raw_signal_logger(log_all_channels, sigvisopts): #do_fft, sigvis_opts
    res=False
    if len(log_all_channels):
        res=True
    if 'Signal spectra joint heatmaps (all channels)' in sigvisopts:
        res=True
    if 'Peak frequency history heatmaps (all channels)' in sigvisopts:
        res=True
    if 'Specific frequency history heatmaps (all channels)' in sigvisopts:
        res=True
    if 'Specific frequency history lineplots (all channels)' in sigvisopts:
        res=True
    if 'Raw signal history heatmaps (all channels)' in sigvisopts:
        res=True
    return res



@callback(Output('do_fft', 'data'),
              Output('log_fft', 'data'),
              Output('do_bins', 'data'),
              Output('log_bins', 'data'),
              Output('do_maxfreq_buffer','data'),
              Output('do_specfreq_buffer','data'),
              Input('log_fft_checkbox', 'value'),
              Input('log_bins_checkbox', 'value'),
              Input('signal_visualization_options','value'))
def set_fft_logger(log_fft_chb, log_bins_chb, sigvis_opts):
    res=[False, False, False, False, False, False]
    if len(log_bins_chb)>0:
        res=[True, False, True, True, False, False]
    if len(log_fft_chb)>0:
        res[0]=True
        res[1]=True
    if 'Signal spectra lineplots' in sigvis_opts:
        res[0]=True
    if 'Signal spectra barplots' in sigvis_opts:
        res[0]=True
        res[2]=True
    if 'Signal spectra history lineplots' in sigvis_opts:
        res[0]=True
        res[2]=True
    if 'Signal spectra joint heatmaps (all channels)' in sigvis_opts:
        res[0]=True
    if 'Peak frequency history heatmaps (all channels)' in sigvis_opts:
        res[0]=True
        res[1]=True
        res[4]=True
    if 'Specific frequency history heatmaps (all channels)' in sigvis_opts:
        res[0]=True
        res[5]=True
    if 'Specific frequency history lineplots (all channels)' in sigvis_opts:
        res[0]=True
        res[5]=True
    return res[0],res[1], res[2], res[3], res[4], res[5]

@callback(Output('log_signal', 'data'),
              Input('log_data','value'),
              Input('signal_visualization_options','value'))
def set_raw_signal_logger(logdata, sigvis_opts): 
    res=False
    if 'Raw signal history heatmaps (all channels)' in sigvis_opts:
        res=True
    if len(logdata)>0:
        res=True
    return res

# @callback(Output("logfile_name", 'style'),
#               Input('log_data', 'value'))
# def update_connection_status(logdata):
#     if len(logdata)>0:
#         return vis
#     else:
#         return invis

@callback(Output('connection_status', 'children'),
              [Input('ws', 'message')])
def update_connection_status(message):
    if message is not None:
        return 'Connection to the device server established'
    else:
        return 'Connection still not established...'

@callback(Output('connection_status', 'style'),
              Output('post_connection_menu', 'style', allow_duplicate=True),
              #Output('data_transmission_rates_panel','style', allow_duplicate=True),
              #Output('data_logging_options_panel','style', allow_duplicate=True),
              #Output('signal_acquistion_options_panel','style', allow_duplicate=True),
              #Output("live_update_button",'style', allow_duplicate=True),
              #Output('signal_joint_plot_container','style', allow_duplicate=True),
              #Output('signal_plot_container','style', allow_duplicate=True),
              Output('task_launcher','options', allow_duplicate=True),
              Output('task_launcher','value', allow_duplicate=True),

              Input('connection_status', 'children'),
               prevent_initial_call=True)
def visualize_post_connection_menu(connection_status):
    if connection_status ==  'Connection to the device server established':
        return b_invis, vis, ['Plot signals', 'Explore logged data'], 'Plot signals' #b_invis, vis, vis, vis, vis, vis, vis, vis,
    else:
        return b_vis, vis, ['Explore logged data'], 'Explore logged data' #b_vis, invis, invis, invis, invis, invis, invis, invis, 
    
@callback(Output('model_launcher', 'style'), #HERE - FIX ANOMALOUS LAYOUT MODS
              Output('signal_plotting', 'style'),
              Output('signal_acquistion_options_panel', 'style'),
              Output("live_update_button", 'style'),
              Output('logdata_panel', 'style'),
              Output('data_transmission_rates_panel', 'style'),
              Output('data_logging_options_panel', 'style'),
              Output('signal_joint_plot_container', 'style'),
              Output('signal_plot_container', 'style'),
              Output("update_logplots_button", 'style'),
              Output('ads_sampling_opts','style'),
              Output('logplots_upd_interval', 'disabled', allow_duplicate=True),
              Output('logdata_plotting_control_panel', 'style'),
              Output("clear_plots_button", "style"),
             # Output('frequency_bar_selection_panel','style', allow_duplicate=True),
             # Output("signal_plotting_update_interval", "value"),
             # Output("signal_buffer_update_interval", "value"),
             # Output("signal_visualization_options", "value"),



              Input('task_launcher', 'value'),
              Input('logged_df', 'data'),
              #allow_duplicate=True,
              prevent_initial_call=True)
def visualize_task_launcher_options_further(task_launcher_option, data):
    if task_launcher_option == 'Plot signals':
        return invis, vis, vis, b_vis, invis, vis, vis, vis, vis, b_invis, {"width":1000, "overflow":"auto"}, True, invis, b_vis #, invis, 100, 100, []
    elif task_launcher_option == 'Explore logged data':
        if data:
            return invis, vis, vis, b_invis, vis, invis, invis, vis, vis, b_vis, invis, True, {"width":1000, "overflow":"auto"}, b_vis #, invis, 100, 100, []
        else:
            return invis, invis, invis, b_invis, vis, invis, invis, invis, invis, b_invis, invis, True, invis, b_invis #, invis, 100, 100, []
    else:
        return vis, invis, invis, b_invis, invis, invis, invis, invis, invis, b_invis, invis, True, invis, b_invis #, invis, 100, 100, []
    
@callback(Output('sampling_stats_info', 'children'),
              Output('signal_acquistion_options_data', 'data'),
              Output('frequency_bin_selector', 'max'),
              Output('frequency_bin_selector', 'value'),


              Input('sampling_frequency_selector', 'value'),
              Input('sample_count_per_iter_selector', 'value'),
              Input('eegranges', 'value'),
              State('frequency_bin_selector', 'value'),
               prevent_initial_call=False)
def calculate_sampling_stats(sf, scount, seteegranges, fbins):
    scount=scount[0]
    sf=sf[0]
    delay=int((1/sf)*1000)
    sampling_period=int(delay*scount)
    maxfreq=int(sf/2)
    time_steps=np.linspace(0, (scount-1)*delay, scount)
    fstep=sf/scount
    f=np.linspace(0, (scount-1)*fstep, scount)
    f_plot=f[0:int(scount/2 + 1)]
    if seteegranges == ['Auto EEG ranges setup']:
        nbins=[0, 0.5, 4, 8, 12, 30, maxfreq]
    else:
        nbins=fbins
    bins=[(nbins[i-1], nbins[i]) for i in range(1, len(nbins))]
    return [f'Delay: {delay} ms',
            html.Br(),
            f'Sampling period: {sampling_period} ms', 
            html.Br(),
            f'Estimated maximal observable frequency for this sampling rate: {maxfreq} Hz'], {'delay':delay, 'f_plot':f_plot, 'time_steps': time_steps, 'n_samples':scount, 'sampling_freq':sf, 'bins':bins}, maxfreq, nbins
    



    
@callback(
    Output("live_update_button", 'children'),
    Output("live_update_button", 'style', allow_duplicate=True),
    Output('signal_plot_container', 'style', allow_duplicate=True),
    Output('start_websock_communication_switch', 'data'),
   # Output('server_msg', 'style'),
    Output("div-ch1-graphs_container", 'style'),
    Output("div-ch2-graphs_container", 'style'),
    Output("div-ch3-graphs_container", 'style'),
    Output("div-ch4-graphs_container", 'style'),
    Output("div-ch5-graphs_container", 'style'),
    Output("div-ch6-graphs_container", 'style'),
    Output("div-ch7-graphs_container", 'style'),
    Output("div-ch8-graphs_container", 'style'),
    
    Input("live_update_button", 'n_clicks'),
    State('start_websock_communication_switch', 'data'),
    Input('channel_selector', 'value'),
    State("live_update_button", 'children'),
    State('task_launcher', 'value'),
    prevent_initial_call=True
)
def start_websocket_data_retrieval(n_clicks, live_update_state, selchannels, btitle, task):
    trigger = ctx.triggered[0]
    trigger_id = trigger['prop_id'].split('.')[0]
    trigger_value = trigger['value']
    if 'ch1' in selchannels:
        ch1st=vis
    else:
        ch1st=invis
    if 'ch2' in selchannels:
        ch2st= vis
    else:
        ch2st = invis
    if 'ch3' in selchannels:
        ch3st=vis
    else:
        ch3st=invis
    if 'ch4' in selchannels:
        ch4st=vis
    else:
        ch4st=invis
    if 'ch5' in selchannels:
        ch5st=vis
    else:
        ch5st=invis
    if 'ch6' in selchannels:
        ch6st=vis   
    else:
        ch6st=invis
    if 'ch7' in selchannels:
        ch7st=vis
    else:
        ch7st=invis
    if 'ch8' in selchannels:
        ch8st=vis  
    else:
        ch8st=invis
    if trigger_id=="live_update_button":
        if live_update_state==False:
            return 'Stop live update...', b_vis, vis, True, ch1st, ch2st, ch3st, ch4st, ch5st, ch6st, ch7st, ch8st
        else:
            return 'Start live update...', b_vis, vis, False, ch1st, ch2st, ch3st, ch4st, ch5st, ch6st, ch7st, ch8st
    else:
        if task=='Plot signals':
            return btitle, b_vis, vis, live_update_state, ch1st, ch2st, ch3st, ch4st, ch5st, ch6st, ch7st, ch8st
        else:
            return 'Stop live update...', b_invis, vis, live_update_state, ch1st, ch2st, ch3st, ch4st, ch5st, ch6st, ch7st, ch8st



rawsig_list=['div-ch1-raw_graph',
    'div-ch2-raw_graph',
    'div-ch3-raw_graph',
    'div-ch4-raw_graph',
    'div-ch5-raw_graph',
    'div-ch6-raw_graph',
    'div-ch7-raw_graph',
    'div-ch8-raw_graph']

fft_list=[ 'div-ch1-fft_graph',
    'div-ch2-fft_graph',
    'div-ch3-fft_graph',
    'div-ch4-fft_graph',
    'div-ch5-fft_graph',
    'div-ch6-fft_graph',
    'div-ch7-fft_graph',
    'div-ch8-fft_graph']

fft_bar_list=['div-ch1-fft_bar_graph',
    'div-ch2-fft_bar_graph',
    'div-ch3-fft_bar_graph',
    'div-ch4-fft_bar_graph',
    'div-ch5-fft_bar_graph',
    'div-ch6-fft_bar_graph',
    'div-ch7-fft_bar_graph',
    'div-ch8-fft_bar_graph']

fa_list=['div-ch1-fagraph',
    'div-ch2-fagraph',
    'div-ch3-fagraph',
    'div-ch4-fagraph',
    'div-ch5-fagraph',
    'div-ch6-fagraph',
    'div-ch7-fagraph',
    'div-ch8-fagraph']

fa_bin_list=['div-ch1-fagraph_bins',
            'div-ch2-fagraph_bins',
            'div-ch3-fagraph_bins',
            'div-ch4-fagraph_bins',
            'div-ch5-fagraph_bins',
            'div-ch6-fagraph_bins',
            'div-ch7-fagraph_bins',
            'div-ch8-fagraph_bins']








@callback(
    Output('div-ch1-raw_graph', 'style'),
    Output('div-ch2-raw_graph', 'style'),
    Output('div-ch3-raw_graph', 'style'),
    Output('div-ch4-raw_graph', 'style'),
    Output('div-ch5-raw_graph', 'style'),
    Output('div-ch6-raw_graph', 'style'),
    Output('div-ch7-raw_graph', 'style'),
    Output('div-ch8-raw_graph', 'style'),

    Output('div-ch1-fft_graph', 'style'),
    Output('div-ch2-fft_graph', 'style'),
    Output('div-ch3-fft_graph', 'style'),
    Output('div-ch4-fft_graph', 'style'),
    Output('div-ch5-fft_graph', 'style'),
    Output('div-ch6-fft_graph', 'style'),
    Output('div-ch7-fft_graph', 'style'),
    Output('div-ch8-fft_graph', 'style'),

    Output('div-ch1-fft_bar_graph', 'style'),
    Output('div-ch2-fft_bar_graph', 'style'),
    Output('div-ch3-fft_bar_graph', 'style'),
    Output('div-ch4-fft_bar_graph', 'style'),
    Output('div-ch5-fft_bar_graph', 'style'),
    Output('div-ch6-fft_bar_graph', 'style'),
    Output('div-ch7-fft_bar_graph', 'style'),
    Output('div-ch8-fft_bar_graph', 'style'),  

    Output('div-ch1-fagraph', 'style'),
    Output('div-ch2-fagraph', 'style'),
    Output('div-ch3-fagraph', 'style'),
    Output('div-ch4-fagraph', 'style'),
    Output('div-ch5-fagraph', 'style'),
    Output('div-ch6-fagraph', 'style'),
    Output('div-ch7-fagraph', 'style'),
    Output('div-ch8-fagraph', 'style'),

    Output('div-ch1-fagraph_bins', 'style'),
    Output('div-ch2-fagraph_bins', 'style'),
    Output('div-ch3-fagraph_bins', 'style'),
    Output('div-ch4-fagraph_bins', 'style'),
    Output('div-ch5-fagraph_bins', 'style'),
    Output('div-ch6-fagraph_bins', 'style'),
    Output('div-ch7-fagraph_bins', 'style'),
    Output('div-ch8-fagraph_bins', 'style'),

    Output('div-hmgraph', 'style'),
    Output('div-hmgraph_max', 'style'),
    Output('div-hmgraph_spec', 'style'),
    Output('div-fagraph_spec', 'style'),
    Output('div-hmgraph_raw', 'style'),


    #Output("frequency_bar_selection_panel",'style', allow_duplicate=True),

    Input('signal_visualization_options', 'value'),
)
def update_graph_visibility(sigvis_opts):
    graph_vis_update_dict={
    'div-ch1-raw_graph':invis,
    'div-ch2-raw_graph':invis,
    'div-ch3-raw_graph':invis,
    'div-ch4-raw_graph':invis,
    'div-ch5-raw_graph':invis,
    'div-ch6-raw_graph':invis,
    'div-ch7-raw_graph':invis,
    'div-ch8-raw_graph':invis,

    'div-ch1-fft_graph':invis,
    'div-ch2-fft_graph':invis,
    'div-ch3-fft_graph':invis,
    'div-ch4-fft_graph':invis,
    'div-ch5-fft_graph':invis,
    'div-ch6-fft_graph':invis,
    'div-ch7-fft_graph':invis,
    'div-ch8-fft_graph':invis,

    'div-ch1-fft_bar_graph':invis,
    'div-ch2-fft_bar_graph':invis,
    'div-ch3-fft_bar_graph':invis,
    'div-ch4-fft_bar_graph':invis,
    'div-ch5-fft_bar_graph':invis,
    'div-ch6-fft_bar_graph':invis,
    'div-ch7-fft_bar_graph':invis,
    'div-ch8-fft_bar_graph':invis,

    'div-ch1-fagraph':invis,
    'div-ch2-fagraph':invis,
    'div-ch3-fagraph':invis,
    'div-ch4-fagraph':invis,
    'div-ch5-fagraph':invis,
    'div-ch6-fagraph':invis,
    'div-ch7-fagraph':invis,
    'div-ch8-fagraph':invis,
    
    'div-ch1-fagraph_bins':invis,
    'div-ch2-fagraph_bins':invis,
    'div-ch3-fagraph_bins':invis,
    'div-ch4-fagraph_bins':invis,
    'div-ch5-fagraph_bins':invis,
    'div-ch6-fagraph_bins':invis,
    'div-ch7-fagraph_bins':invis,
    'div-ch8-fagraph_bins':invis,
    
    'div-hmgraph':invis,
    'div-hmgraph_max':invis,
    'div-hmgraph_spec': invis,
    'div-fagraph_spec': invis,
    'div-hmgraph_raw': invis,}



    for sv in sigvis_opts:
        if 'Raw signal lineplots' in sv:
            for gn in rawsig_list:
                graph_vis_update_dict[gn]=vis
        if 'Signal spectra lineplots' in sv:
            for gn in fft_list:
                graph_vis_update_dict[gn]=vis
        if 'Signal spectra barplots' in sv:
            for gn in fft_bar_list:
                graph_vis_update_dict[gn]=vis
        if 'Raw signal history lineplots' in sv:
            for gn in fa_list:
                graph_vis_update_dict[gn]=vis
        if 'Signal spectra history lineplots' in sv:
            for gn in fa_bin_list:
                graph_vis_update_dict[gn]=vis
        if 'Signal spectra joint heatmaps (all channels)' in sv:
            graph_vis_update_dict["div-hmgraph"]=vis
        if 'Peak frequency history heatmaps (all channels)' in sv:
            graph_vis_update_dict["div-hmgraph_max"]=vis
        if 'Specific frequency history heatmaps (all channels)' in sv:
            graph_vis_update_dict['div-hmgraph_spec']=vis
        if 'Specific frequency history lineplots (all channels)' in sv:
            graph_vis_update_dict['div-fagraph_spec']=vis
        if 'Raw signal history heatmaps (all channels)' in sv:
            graph_vis_update_dict['div-hmgraph_raw']=vis
    return [graph_vis_update_dict[i] for i in rawsig_list + fft_list + fft_bar_list + fa_list+fa_bin_list + ["div-hmgraph", "div-hmgraph_max",  'div-hmgraph_spec','div-fagraph_spec','div-hmgraph_raw']]# + [freq_panel_visibility]



def get_fft(signal_values, n_samples):
    X=np.fft.fft(signal_values)
    X_mag=np.abs(X)/n_samples
    X_mag_plot=2*X_mag[0:int(n_samples/2 + 1)]
    return list(X_mag_plot)



raw_buffer_namelist={f'{ch}_raw_current':[] for ch in signal_chnames}
blank_updates={i:[] for i in raw_buffer_namelist}
@callback(
    Output("ws", "send"),
    Output('message', "children"),
    [Output(rbn, 'data') for rbn in raw_buffer_namelist],


    Input("ws", "message"),
    Input('start_websock_communication_switch', 'data'),
    Input('logplots_upd_interval', 'n_intervals'),
    State('signal_acquistion_options_data', 'data'),
    State('channel_selector', 'value'),
    State('get_all_channel_data', 'data'),
    State('current_split_logdata', 'data'),
    State('logplots_upd_interval', 'max_intervals'),
)
def websocket_communication(server_message, start_communication, n_intervals, signal_acquisition_options, selected_channels, log_all_channels, cursplitlogdata, max_intervals):
    updates=blank_updates.copy()
    trigger = ctx.triggered[0]
    trigger_id = trigger['prop_id'].split('.')[0]
    trigger_value = trigger['value']
    global signal_chnames
    global raw_buffer_namelist
    if trigger_id == 'ws': 
        data=server_message['data']
        if data == "Awaiting delay and data transfer buffer size in shape with space separator":
            delay = signal_acquisition_options['delay']
            n_samples = signal_acquisition_options['n_samples']
            message=f'{delay},{n_samples}'
            return [message] + [str(server_message)] + [updates[i] for i in raw_buffer_namelist]
        elif data == "Delay and data transfer buffer size set up":
            message = "start_data_transfer_from_ads"
            return [message] + [str(server_message)] + [updates[i] for i in raw_buffer_namelist]
        
        elif start_communication==True:
            data=json.loads(server_message['data'])
            for chname in signal_chnames:
                cdata=data[chname]


                #Use for data generation:
                #sampling_freq=signal_acquisition_options['sampling_freq']
                #tstep=1/sampling_freq
                # signal_freq=30
                #n_samples=signal_acquisition_options['n_samples']  #int(sampling_freq/signal_freq)
                #time_steps=np.linspace(0, (n_samples-1)*tstep, n_samples)
                #fstep=sampling_freq/n_samples
                #f=np.linspace(0, (n_samples-1)*fstep, n_samples)
                # y=1*np.sin(2*np.pi*signal_freq*time_steps) #uncomment for  synthesized signal usage
                # cdata=y
                if log_all_channels:
                    updates[f'{chname}_raw_current']=cdata
                else:
                    if chname in selected_channels:
                        updates[f'{chname}_raw_current']=cdata
            return ['']+[str(server_message)]+[updates[i] for i in raw_buffer_namelist]
        else:
            return ['']+[str(server_message)]+[updates[i] for i in raw_buffer_namelist]
    else:
        if trigger_id == 'logplots_upd_interval':
            data=cursplitlogdata[n_intervals-1] #here may need to fix
            for ind in range(len(signal_chnames)):

                chname_zind=signal_chnames_zindexed[ind] #FIX LATER THIS
                chname=signal_chnames[ind]

                cdata=data[chname_zind]
                if log_all_channels:
                    updates[f'{chname}_raw_current']=cdata
                else:
                    if chname in selected_channels:
                        updates[f'{chname}_raw_current']=cdata

            return ['']+[f'Interval N: {n_intervals}, current interval processing {(n_intervals/max_intervals)*100}% ready']+[updates[i] for i in raw_buffer_namelist]
        if start_communication == True:
            return ["set_delay_and_data_transfer_buffer_size"]+[str(server_message)]+[updates[i] for i in raw_buffer_namelist]
        if start_communication == False:
            return ["stop_data_transfer_from_ads"]+[str(server_message)]+[updates[i] for i in raw_buffer_namelist]



full_raw_buffer_namelist={f'{ch}_raw_buffer':[] for ch in signal_chnames}
fft_buffer_namelist={f'{ch}_fft_current':[] for ch in signal_chnames}



@callback( #here - optimize
    [Output(frbn, 'data') for frbn in full_raw_buffer_namelist],
    Output('logging_counter','data', allow_duplicate=True),

    [Input(rbn, 'data') for rbn in raw_buffer_namelist],
    [State(frbn, 'data') for frbn in full_raw_buffer_namelist],
    State('n_indiv_samples_to_store','data'),
    State('log_signal', 'data'),
    State('logging_counter','data'),
    prevent_initial_call=True
)
def buffer_updates(ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current,
                   ch1_buffer, ch2_buffer, ch3_buffer, ch4_buffer, ch5_buffer, ch6_buffer, ch7_buffer, ch8_buffer,
                   signal_buffer_update_interval, log_signal, logging_counter):
    if not log_signal:
        raise PreventUpdate
    else:
        csignals=[ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current]
        bsignals=[ch1_buffer, ch2_buffer, ch3_buffer, ch4_buffer, ch5_buffer, ch6_buffer, ch7_buffer, ch8_buffer]
        for i in range(len(csignals)):
            val=csignals[i]
            if len(val)!=0:
                bsignals[i]+=val
                bsignals[i]=bsignals[i][-signal_buffer_update_interval:] 
                logging_counter=logging_counter-1
        return bsignals + [logging_counter]
    



@callback( #here - optimize
    [Output(fbn, 'data') for fbn in fft_buffer_namelist],
    [Input(rbn, 'data') for rbn in raw_buffer_namelist],

    State('do_fft','data'),
    State('signal_acquistion_options_data', 'data'),
)
def fft_updates(ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current,
                   do_fft, signal_acquisition_options):

    if not do_fft:
        raise PreventUpdate
    else:
        csignals=[ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current]
        ffts=[[],[],[],[],[],[],[],[]]
        for i in range(len(csignals)):
            val=csignals[i]
            if len(val)!=0:
                if do_fft:
                    n_samples = signal_acquisition_options['n_samples']
                    if n_samples>thresh_fft_n_samples:
                        ffts[i]=get_fft(val, n_samples)
        return ffts


full_fft_buffer_namelist={f'{ch}_fft_buffer':[] for ch in signal_chnames}
maxfreq_buffer_namelist={f'{ch}_maxfreq_buffer':[] for ch in signal_chnames}
@callback(
    [Output(ffbn, 'data') for ffbn in full_fft_buffer_namelist],
    [Output(mfbn, 'data') for mfbn in maxfreq_buffer_namelist],

    [Input(fbn, 'data') for fbn in fft_buffer_namelist],
    [State(ffbn, 'data') for ffbn in full_fft_buffer_namelist],
    [State(mfbn, 'data') for mfbn in maxfreq_buffer_namelist],
    State("log_fft", "data"),
    State('do_maxfreq_buffer', "data"),
    State("signal_buffer_update_interval", "value"),
    State("signal_acquistion_options_data", "data"),
)
def update_fft_buffers(ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current,
                   ch1_buffer, ch2_buffer, ch3_buffer, ch4_buffer, ch5_buffer, ch6_buffer, ch7_buffer, ch8_buffer, 
                   ch1_mfbuffer, ch2_mfbuffer, ch3_mfbuffer, ch4_mfbuffer, ch5_mfbuffer, ch6_mfbuffer, ch7_mfbuffer, ch8_mfbuffer,
                   log_fft, do_maxfreq, signal_buffer_update_interval, sigaquisition_opts):
    
    if not log_fft:
        raise PreventUpdate
    else: 
        csignals=[ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current]
        bsignals=[ch1_buffer, ch2_buffer, ch3_buffer, ch4_buffer, ch5_buffer, ch6_buffer, ch7_buffer, ch8_buffer]
        mfs=[ch1_mfbuffer, ch2_mfbuffer, ch3_mfbuffer, ch4_mfbuffer, ch5_mfbuffer, ch6_mfbuffer, ch7_mfbuffer, ch8_mfbuffer]
        for i in range(len(csignals)):
            try:
                ffti=csignals[i]
                if len(ffti)!=0:
                    bsignals[i].append(ffti)
                    bsignals[i]=bsignals[i][-signal_buffer_update_interval:]
                if do_maxfreq:
                    f_plot=sigaquisition_opts['f_plot']
                    max_index=np.argmax(np.array(ffti))
                    max_frequency=f_plot[max_index]
                    mfs[i].append(max_frequency)
                    mfs[i]=mfs[i][-signal_buffer_update_interval:]
            except:
                print('Probably too high signal acquisition throughput, missed the fft buffer array update...')
                raise PreventUpdate
    return bsignals+mfs

specfreq_buffer_namelist={f'{ch}_specfreq_buffer':[] for ch in signal_chnames}
specfreq_current_namelist={f'{ch}_specfreq_current':[] for ch in signal_chnames}
@callback(
    [Output(sfbn, 'data') for sfbn in specfreq_buffer_namelist],
    [Output(sfcn, 'data') for sfcn in specfreq_current_namelist],

    [Input(fbn, 'data') for fbn in fft_buffer_namelist],
    [State(sfbn, 'data') for sfbn in specfreq_buffer_namelist],
    State('do_specfreq_buffer', "data"),
    State("signal_buffer_update_interval", "value"),
    State("signal_acquistion_options_data", "data"),
    State("targetnontarget_freqs","data"),
)
def update_specreq_buffers(ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current,
                   ch1_buffer, ch2_buffer, ch3_buffer, ch4_buffer, ch5_buffer, ch6_buffer, ch7_buffer, ch8_buffer, 
                   do_specfreq, signal_buffer_update_interval, sigaquisition_opts,targetnontarget_freqs):
    
    if not do_specfreq:
        raise PreventUpdate
    else: 
        f_plot=sigaquisition_opts['f_plot']
        tar_freqs=list(map(tuple, targetnontarget_freqs['tar']))
        nontar_freqs=list(map(tuple, targetnontarget_freqs['nontar']))
        csignals=[ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current]
        bsignals=[ch1_buffer, ch2_buffer, ch3_buffer, ch4_buffer, ch5_buffer, ch6_buffer, ch7_buffer, ch8_buffer]
        sfcur=[[],[],[],[],[],[],[],[]]
        for i in range(len(csignals)):
            try:
                ffti=csignals[i]
                if len(ffti)!=0:
                    specfreqres=get_bin_values(f_plot, ffti, tar_freqs=tar_freqs, 
                                               nontar_freqs=nontar_freqs, 
                                               return_sep=False, 
                                               return_ratio=True)
                    bsignals[i].append(specfreqres)
                    bsignals[i]=bsignals[i][-signal_buffer_update_interval:]
                    sfcur[i]=[specfreqres]
            except:
               print('Probably too high signal acquisition throughput, missed the specific frequency buffer array update...')
               raise PreventUpdate
    return bsignals+sfcur






@callback(
    [Output('ch1_fagraph', 'extendData')],
    [Output('ch2_fagraph', 'extendData')],
    [Output('ch3_fagraph', 'extendData')],
    [Output('ch4_fagraph', 'extendData')],
    [Output('ch5_fagraph', 'extendData')],
    [Output('ch6_fagraph', 'extendData')],
    [Output('ch7_fagraph', 'extendData')],
    [Output('ch8_fagraph', 'extendData')],

    Input('ch1_raw_current', 'data'),
    Input('ch2_raw_current', 'data'),
    Input('ch3_raw_current', 'data'),
    Input('ch4_raw_current', 'data'),
    Input('ch5_raw_current', 'data'),
    Input('ch6_raw_current', 'data'),
    Input('ch7_raw_current', 'data'),
    Input('ch8_raw_current', 'data'),


    State("signal_plotting_update_interval", 'value'),
    State('start_websock_communication_switch', 'data'),
    State('signal_visualization_options', 'value'),
    State('start_log_plotting_switch', 'data'),
)
def update_fast_update_signal_lineplots(ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current,                                        
                                        plotting_interval,commun_started, sigvis_opts, log_plotting_started):
    if ((not commun_started) and (not log_plotting_started)):
        raise PreventUpdate
    else:
        if 'Raw signal history lineplots' not in sigvis_opts:
            raise PreventUpdate
        else:
            cur=[ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current]
            updates=[[{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]], ]
            x_new=[datetime.now().strftime("%H:%M:%S:%f") for i in range(len(cur[0]))]
            for i in range(8):
                cdata=cur[i]
                if len(cdata)>0:
                    updates[i]=[{'x': [x_new], 'y': [cdata]}, [0], plotting_interval]
            return updates




@callback(
    [Output('ch1_raw_graph', 'extendData')],
    [Output('ch2_raw_graph', 'extendData')],
    [Output('ch3_raw_graph', 'extendData')],
    [Output('ch4_raw_graph', 'extendData')],
    [Output('ch5_raw_graph', 'extendData')],
    [Output('ch6_raw_graph', 'extendData')],
    [Output('ch7_raw_graph', 'extendData')],
    [Output('ch8_raw_graph', 'extendData')],

    Input('ch1_raw_current', 'data'),
    Input('ch2_raw_current', 'data'),
    Input('ch3_raw_current', 'data'),
    Input('ch4_raw_current', 'data'),
    Input('ch5_raw_current', 'data'),
    Input('ch6_raw_current', 'data'),
    Input('ch7_raw_current', 'data'),
    Input('ch8_raw_current', 'data'),


    State('start_websock_communication_switch', 'data'),
    State('signal_visualization_options', 'value'),
    State('start_log_plotting_switch', 'data'),
)
def update_current_signal_raw_lineplots(ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current,
                                        commun_started, sigvis_opts, log_plotting_started):
    if ((not commun_started) and (not log_plotting_started)):
        raise PreventUpdate
    else:
        if 'Raw signal lineplots' not in sigvis_opts:
            raise PreventUpdate
        else:
            cur=[ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current]
            x_new=[i for i in range(len(cur[0]))]
            intlen=len(x_new)
            updates=[[{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]], ]
            x_new=[i for i in range(len(cur[0]))]
            for i in range(8):
                cdata=cur[i]
                if len(cdata)>0:
                    updates[i]=[{'x': [x_new], 'y': [cdata]}, [0], intlen]
            return updates










@callback(
    Output('hmgraph', 'figure'),

    Input('ch1_fft_current', 'data'),
    Input('ch2_fft_current', 'data'),
    Input('ch3_fft_current', 'data'),
    Input('ch4_fft_current', 'data'),
    Input('ch5_fft_current', 'data'),
    Input('ch6_fft_current', 'data'),
    Input('ch7_fft_current', 'data'),
    Input('ch8_fft_current', 'data'),


    State('start_websock_communication_switch', 'data'),
    State('signal_acquistion_options_data', 'data'),
    State('signal_visualization_options', 'value'),
    State('start_log_plotting_switch', 'data'),
)
def update_fast_fourier_transform_lineplots(ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current,
                                        commun_started, sig_aquisition_opts, sigvis_opts, log_plotting_started):
    
    if ((not commun_started) and (not log_plotting_started)):
        raise PreventUpdate
    else:
        if 'Signal spectra joint heatmaps (all channels)' not in sigvis_opts:
            raise PreventUpdate
        else:
            cur=[ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current]
            f_plot=sig_aquisition_opts['f_plot']
            hmdata=np.vstack(cur)
            hmfigure={
                 'data': [go.Heatmap(
                      z=hmdata,
                      colorscale='Viridis'
                 )],
                 'layout': go.Layout(
                      height=700,
                      xaxis=dict(
                    title='Frequency',
                    tickvals=f_plot,
                    tickfont={'size': 10} # Set the font size for x-axis tick labels,
                    ),
                    yaxis=dict(
                        title='Channel'
                    ),
                      
                 )
              }
        return hmfigure
            
@callback(
    Output('hmgraph_max', 'figure'),

    Input('ch1_maxfreq_buffer', 'data'),
    Input('ch2_maxfreq_buffer', 'data'),
    Input('ch3_maxfreq_buffer', 'data'),
    Input('ch4_maxfreq_buffer', 'data'),
    Input('ch5_maxfreq_buffer', 'data'),
    Input('ch6_maxfreq_buffer', 'data'),
    Input('ch7_maxfreq_buffer', 'data'),
    Input('ch8_maxfreq_buffer', 'data'),


    State('start_websock_communication_switch', 'data'),
    State('signal_acquistion_options_data', 'data'),
    State('signal_visualization_options', 'value'),
    State("signal_plotting_update_interval", 'value'),
    State('start_log_plotting_switch', 'data'),
)
def update_fast_fourier_transform_lineplots(ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current,
                                        commun_started, sig_aquisition_opts, sigvis_opts, plotting_interval, log_plotting_started):
    
    if ((not commun_started) and (not log_plotting_started)):
        raise PreventUpdate
    else:
        if 'Peak frequency history heatmaps (all channels)' not in sigvis_opts:
            raise PreventUpdate
        else:
            cur=[ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current]
            hmdata=np.vstack(cur)[:,-plotting_interval:]
            hmfigure={
                 'data': [go.Heatmap(
                      z=hmdata,
                      colorscale='Viridis'
                 )],
                 'layout': go.Layout(
                      height=700,
                      xaxis=dict(
                    title='Timestep',
                    #tickvals=f_plot,
                    #tickfont={'size': 10} # Set the font size for x-axis tick labels,
                    ),
                    yaxis=dict(
                        title='Channel'
                    ),
                      
                 )
              }
        return hmfigure

@callback(
    Output('hmgraph_spec', 'figure'),

    Input('ch1_specfreq_buffer', 'data'),
    Input('ch2_specfreq_buffer', 'data'),
    Input('ch3_specfreq_buffer', 'data'),
    Input('ch4_specfreq_buffer', 'data'),
    Input('ch5_specfreq_buffer', 'data'),
    Input('ch6_specfreq_buffer', 'data'),
    Input('ch7_specfreq_buffer', 'data'),
    Input('ch8_specfreq_buffer', 'data'),


    State('start_websock_communication_switch', 'data'),
    State('signal_acquistion_options_data', 'data'),
    State('signal_visualization_options', 'value'),
    State("signal_plotting_update_interval", 'value'),
    State('start_log_plotting_switch', 'data'),
)
def update_fast_fourier_transform_lineplots(ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current,
                                        commun_started, sig_aquisition_opts, sigvis_opts, plotting_interval, log_plotting_started):
    
    if ((not commun_started) and (not log_plotting_started)):
        raise PreventUpdate
    else:
        if 'Specific frequency history heatmaps (all channels)' not in sigvis_opts:
            raise PreventUpdate
        else:
            cur=[ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current]
            hmdata=np.vstack(cur)[:,-plotting_interval:]
            hmfigure={
                 'data': [go.Heatmap(
                      z=hmdata,
                      colorscale='Viridis'
                 )],
                 'layout': go.Layout(
                      height=700,
                      xaxis=dict(
                    title='Timestep',
                    #tickvals=f_plot,
                    #tickfont={'size': 10} # Set the font size for x-axis tick labels,
                    ),
                    yaxis=dict(
                        title='Channel'
                    ),
                      
                 )
              }
        return hmfigure
    

@callback(
    Output('hmgraph_raw', 'figure'),

    Input('ch1_raw_buffer', 'data'),
    Input('ch2_raw_buffer', 'data'),
    Input('ch3_raw_buffer', 'data'),
    Input('ch4_raw_buffer', 'data'),
    Input('ch5_raw_buffer', 'data'),
    Input('ch6_raw_buffer', 'data'),
    Input('ch7_raw_buffer', 'data'),
    Input('ch8_raw_buffer', 'data'),


    State('start_websock_communication_switch', 'data'),
    State('signal_acquistion_options_data', 'data'),
    State('signal_visualization_options', 'value'),
    State("signal_plotting_update_interval", 'value'),
    State('start_log_plotting_switch', 'data'),
)
def update_fast_fourier_transform_lineplots(ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current,
                                        commun_started, sig_aquisition_opts, sigvis_opts, plotting_interval, log_plotting_started):
    
    if ((not commun_started) and (not log_plotting_started)):
        raise PreventUpdate
    else:
        if 'Raw signal history heatmaps (all channels)' not in sigvis_opts:
            raise PreventUpdate
        else:
            cur=[ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current]
            hmdata=np.vstack(cur)[:,-plotting_interval:]
            hmfigure={
                 'data': [go.Heatmap(
                      z=hmdata,
                      colorscale='Viridis'
                 )],
                 'layout': go.Layout(
                      height=700,
                      xaxis=dict(
                    title='Timestep',
                    #tickvals=f_plot,
                    #tickfont={'size': 10} # Set the font size for x-axis tick labels,
                    ),
                    yaxis=dict(
                        title='Channel'
                    ),
                      
                 )
              }
        return hmfigure

@callback(
    Output('fagraph_spec', 'extendData'),

    Input('ch1_specfreq_current', 'data'),
    Input('ch2_specfreq_current', 'data'),
    Input('ch3_specfreq_current', 'data'),
    Input('ch4_specfreq_current', 'data'),
    Input('ch5_specfreq_current', 'data'),
    Input('ch6_specfreq_current', 'data'),
    Input('ch7_specfreq_current', 'data'),
    Input('ch8_specfreq_current', 'data'),


    State("signal_plotting_update_interval", 'value'),
    State('start_websock_communication_switch', 'data'),
    State('signal_visualization_options', 'value'),
    State('signal_acquistion_options_data', 'data'),
    State('start_log_plotting_switch', 'data'),
   # State('signal_acquistion_options_data', 'data')
)
def update_fast_update_signal_bin_lineplots(ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current,                                        
                                        plotting_interval,commun_started, sigvis_opts, sig_aquisition_opts,log_plotting_started):
    if ((not commun_started) and (not log_plotting_started)):
        raise PreventUpdate
    else:
        if 'Specific frequency history lineplots (all channels)' not in sigvis_opts:
            raise PreventUpdate
        else:
            time_cur=datetime.now().strftime("%H:%M:%S:%f")
            cur=[ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current]
            nch=len(cur)
            updates=[{'x': [[time_cur] for i in range(nch)], 'y': cur,}, list(range(nch)), plotting_interval]
            return updates
        




@callback(
    [Output('ch1_fft_graph', 'extendData')],
    [Output('ch2_fft_graph', 'extendData')],
    [Output('ch3_fft_graph', 'extendData')],
    [Output('ch4_fft_graph', 'extendData')],
    [Output('ch5_fft_graph', 'extendData')],
    [Output('ch6_fft_graph', 'extendData')],
    [Output('ch7_fft_graph', 'extendData')],
    [Output('ch8_fft_graph', 'extendData')],

    Input('ch1_fft_current', 'data'),
    Input('ch2_fft_current', 'data'),
    Input('ch3_fft_current', 'data'),
    Input('ch4_fft_current', 'data'),
    Input('ch5_fft_current', 'data'),
    Input('ch6_fft_current', 'data'),
    Input('ch7_fft_current', 'data'),
    Input('ch8_fft_current', 'data'),


    State('start_websock_communication_switch', 'data'),
    State('signal_acquistion_options_data', 'data'),
    State('signal_visualization_options', 'value'),
    State('start_log_plotting_switch', 'data'),
)
def update_fast_fourier_transform_lineplots(ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current,
                                        commun_started, sig_aquisition_opts, sigvis_opts,log_plotting_started):
    
    if ((not commun_started) and (not log_plotting_started)):
        raise PreventUpdate
    else:
        if 'Signal spectra lineplots' not in sigvis_opts:
            raise PreventUpdate
        else:
            cur=[ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current]
            f_plot=sig_aquisition_opts['f_plot']
            intlen=len(f_plot)
            updates=[[{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]], ]
            for i in range(8):
                cdata=cur[i]
                if len(cdata)>0:
                    updates[i]=[{'x': [f_plot], 'y': [cdata]}, [0], intlen]
            return updates



bin_buffer_namelist={f'{ch}_bins_current':[] for ch in signal_chnames}
@callback(
    [Output(ffbn, 'data') for ffbn in bin_buffer_namelist],
    [Input(fbn, 'data') for fbn in fft_buffer_namelist],
    State("do_bins", "data"),
    State('signal_acquistion_options_data', 'data')
)
def get_binned_frequencies(ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current,
                           do_bins,
                           sig_aquisition_opts):
    if not do_bins:
        raise PreventUpdate
    else:
        results=[[],[],[],[],[],[],[],[]]
        csignals=[ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current]
        bins=sig_aquisition_opts['bins']
        fpl=np.array(sig_aquisition_opts['f_plot'])
        for i in range(len(csignals)):
            val=csignals[i]
            if len(val)>0:
                magnitudes=[]
                xmp=np.array(val)
                for low, high in bins:
                        mask = (fpl >= low) & (fpl < high) #here might be bug potential
                        magnitude = np.abs(xmp[mask]).mean()
                        magnitudes.append(magnitude)
                results[i]=magnitudes
    return results


@callback(
    [Output('ch1_fft_bar_graph', 'extendData')],
    [Output('ch2_fft_bar_graph', 'extendData')],
    [Output('ch3_fft_bar_graph', 'extendData')],
    [Output('ch4_fft_bar_graph', 'extendData')],
    [Output('ch5_fft_bar_graph', 'extendData')],
    [Output('ch6_fft_bar_graph', 'extendData')],
    [Output('ch7_fft_bar_graph', 'extendData')],
    [Output('ch8_fft_bar_graph', 'extendData')],

    Input('ch1_bins_current', 'data'),
    Input('ch2_bins_current', 'data'),
    Input('ch3_bins_current', 'data'),
    Input('ch4_bins_current', 'data'),
    Input('ch5_bins_current', 'data'),
    Input('ch6_bins_current', 'data'),
    Input('ch7_bins_current', 'data'),
    Input('ch8_bins_current', 'data'),


    State('start_websock_communication_switch', 'data'),
    State('signal_acquistion_options_data', 'data'),
    State('signal_visualization_options', 'value'),
    State('start_log_plotting_switch', 'data'),
)
def update_fast_fourier_transform_lineplots(ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current,
                                        commun_started, sig_aquisition_opts, sigvis_opts, log_plotting_started):
   
    if ((not commun_started) and (not log_plotting_started)):
        raise PreventUpdate
    else:
        if 'Signal spectra barplots' not in sigvis_opts:
            raise PreventUpdate
        else:
            cur=[ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current]
            updates=[[{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]],
                    [{'x': [[]], 'y': [[]]}, [0]], ]
            bins=sig_aquisition_opts['bins']
            x=[f"{low}-{high} Hz" for low, high in bins]
            for i in range(8):
                magnitudes=cur[i]
                if len(magnitudes)>0:
                    updates[i]=[{'x': [x], 'y': [magnitudes]}, [0], 6]
            return updates



@callback(
    [Output('ch1_fagraph_bins', 'extendData')],
    [Output('ch2_fagraph_bins', 'extendData')],
    [Output('ch3_fagraph_bins', 'extendData')],
    [Output('ch4_fagraph_bins', 'extendData')],
    [Output('ch5_fagraph_bins', 'extendData')],
    [Output('ch6_fagraph_bins', 'extendData')],
    [Output('ch7_fagraph_bins', 'extendData')],
    [Output('ch8_fagraph_bins', 'extendData')],

    Input('ch1_bins_current', 'data'),
    Input('ch2_bins_current', 'data'),
    Input('ch3_bins_current', 'data'),
    Input('ch4_bins_current', 'data'),
    Input('ch5_bins_current', 'data'),
    Input('ch6_bins_current', 'data'),
    Input('ch7_bins_current', 'data'),
    Input('ch8_bins_current', 'data'),


    State("signal_plotting_update_interval", 'value'),
    State('start_websock_communication_switch', 'data'),
    State('signal_visualization_options', 'value'),
    State('signal_acquistion_options_data', 'data'),
    State('start_log_plotting_switch', 'data'),
)
def update_fast_update_signal_bin_lineplots(ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current,                                        
                                        plotting_interval,commun_started, sigvis_opts, sig_aquisition_opts, log_plotting_started):
    if ((not commun_started) and (not log_plotting_started)):
        raise PreventUpdate
    else:
        if 'Signal spectra history lineplots' not in sigvis_opts:
            raise PreventUpdate
        else:
            cur=[ch1_current, ch2_current, ch3_current, ch4_current, ch5_current, ch6_current, ch7_current, ch8_current]
            bins=sig_aquisition_opts['bins']
            binnames=[f"{low}-{high} Hz" for low, high in bins]
            n_bins=len(cur[0])
            time_cur=datetime.now().strftime("%H:%M:%S:%f")
            singleblank=[{'x': [[] for i in range(n_bins)], 
                          'y': [[] for i in range(n_bins)]}, list(range(n_bins))]
            updates=[singleblank.copy() for i in range(len(cur))]
            for i in range(8):
                magnitudes=cur[i]
                if len(magnitudes)>0:
                    updates[i]=[{'x': [[time_cur] for i in range(n_bins)], 'y': [[i] for i in magnitudes],}, list(range(n_bins)), plotting_interval]
            return updates












@callback(
    Output('logging_counter','data', allow_duplicate=True),

    Input('logging_counter','data'),
    State('log_signal', 'data'),
    State('log_data', 'value'),
    State('signal_buffer_update_interval', 'value'),
    State('log_fft_checkbox','value'),
    State('log_bins_checkbox','value'),
    #State('log_specfreq','value'),
    #State('log_peakfreq','value'),
    State('log_raw_data','value'),

    [State(i, 'data') for i in full_raw_buffer_namelist],
    [State(i, 'data') for i in full_fft_buffer_namelist],
    [State(i, 'data') for i in specfreq_buffer_namelist],
    [State(i, 'data') for i in maxfreq_buffer_namelist],
    [State(i, 'data') for i in bin_buffer_namelist],
    State('signal_acquistion_options_data','data'),
    State('targetnontarget_freqs', 'data'),
    State("max_points_per_log", 'value'),
    prevent_initial_call=True
)
def signal_log(logging_counter,log_signal, log_data, buffer_update_interval,
               log_fft, log_bins, log_raw,
               ch1_buffer, ch2_buffer, ch3_buffer, ch4_buffer, ch5_buffer, ch6_buffer, ch7_buffer, ch8_buffer,
               ch1_fft, ch2_fft, ch3_fft, ch4_fft, ch5_fft, ch6_fft, ch7_fft, ch8_fft,
               ch1_spf, ch2_spf, ch3_spf, ch4_spf, ch5_spf, ch6_spf, ch7_spf, ch8_spf,
               ch1_mf, ch2_mf, ch3_mf, ch4_mf, ch5_mf, ch6_mf, ch7_mf, ch8_mf,
               ch1_b, ch2_b, ch3_b, ch4_b, ch5_b, ch6_b, ch7_b, ch8_b,
               sigaquis_opts, tarnontarf, maxpoints):
    #print(logging_counter)
    if not log_signal:
        if len(log_data)>0:
            data=dict()
            data['timestamp']=datetime.now().strftime("%Y:%m:%d:%H:%M:%S:%f")
            data['signal_stopped']='True'
            queue.put(data)
        return buffer_update_interval
    if len(log_data)==0:
        raise PreventUpdate
    if logging_counter>0:
        raise PreventUpdate
    else:
        csignals=[ch1_buffer, ch2_buffer, ch3_buffer, ch4_buffer, ch5_buffer, ch6_buffer, ch7_buffer, ch8_buffer]
        ffts=[ch1_fft, ch2_fft, ch3_fft, ch4_fft, ch5_fft, ch6_fft, ch7_fft, ch8_fft]
        spf=[ch1_spf, ch2_spf, ch3_spf, ch4_spf, ch5_spf, ch6_spf, ch7_spf, ch8_spf]
        mf=[ch1_mf, ch2_mf, ch3_mf, ch4_mf, ch5_mf, ch6_mf, ch7_mf, ch8_mf]
        b=[ch1_b, ch2_b, ch3_b, ch4_b, ch5_b, ch6_b, ch7_b, ch8_b]
        data=dict()
        data['timestamp']=datetime.now().strftime("%Y:%m:%d:%H:%M:%S:%f")
        data['signal_acquisition_options']=sigaquis_opts
        data['tarnontar_freq']=tarnontarf
        if len(log_raw)>0:
            if maxpoints==-1:
                data['raw_signal']={f'ch{i}':csignals[i] for i in range(len(csignals))}
            else:
                data['raw_signal']={f'ch{i}':csignals[i][-maxpoints:] for i in range(len(csignals))}
        if len(log_fft)>0:
            data['spectra']={f'ch{i}':ffts[i] for i in range(len(ffts))}
        if len(log_bins)>0:
            data['freqbins']={f'ch{i}':b[i] for i in range(len(b))}
        # if len(log_specfreq)>0:
        #     data['specfreq']={f'ch{i}':spf[i] for i in range(len(spf))}
        # if len(log_peakfreq)>0:
        #     data['peakfreq']={f'ch{i}':mf[i] for i in range(len(mf))}
        log_queue.put(data)
        return buffer_update_interval

#LOGGING
def parse_logfile(input, input_is_filename=True, save_dfs=False, n_channels=8, toreturn=True, return_signal_options=True):
    def min_length_of_vals(datadict):
        min_length = min(len(v) for v in datadict.values())
        return min_length
    def crop_data_dict_and_make_df(datadict):
        min_length=min_length_of_vals(datadict)
        cropped_dict={k: v[:min_length] for k, v in datadict.items()}
        df=pd.DataFrame(cropped_dict)
        return df
    import pandas as pd
    import numpy as np
    import json
    if input_is_filename==True:
        with open(input, 'r') as f:
        # Read the entire file as a string
            data = f.read()
    else:
        data=input
    objects=data.split('}\n{')
    objects[0]=objects[0][1:]

    dicts=[]
    for string in tqdm(objects):
        try:
            stringv='{'+string+'}'
            ds = json.loads(stringv)
            dicts.append(ds)
        except:
            print('failed to parse a line')
    opts=dicts[0]['signal_acquisition_options']
    f_plot=dicts[0]['signal_acquisition_options']['f_plot']
    bins=list(map(tuple, dicts[0]['signal_acquisition_options']['bins']))
    logkeys=dicts[0].keys()

    raw_signals={f'ch{i}':[] for i in range(n_channels)}
    raw_signals['timestamp']=[]
    binnames=[f'{low}-{high} Hz' for low,high in bins]
    binned_singals={f"ch{i} {binname}":[] for binname in binnames for i in range(n_channels)}
    binned_singals['timestamp']=[]
    spectra_signals={f"ch{i} {freq}":[] for freq in f_plot for i in range(n_channels)}
    spectra_signals['timestamp']=[]

    for dict in dicts:
        if 'signal_stopped' not in dict.keys():
            timestamp=dict['timestamp']
            if 'raw_signal' in logkeys:
                raw_data=dict['raw_signal']
            if 'spectra' in logkeys:
                spectra_data=dict['spectra']
            if 'freqbins' in logkeys:
                bin_data=dict['freqbins']

            if 'raw_signal' in logkeys:
                for ch in raw_data:
                    chraw=raw_data[ch]
                    raw_signals[ch]+=chraw
                raw_signals['timestamp']+=[timestamp for i in range(len(chraw))]

            if 'spectra' in logkeys:
                for ch in spectra_data:
                    spectraldch=spectra_data[ch]
                    for diter in spectraldch:
                        for freqi in range(len(f_plot)):
                            freq=f_plot[freqi]
                            spname=f'{ch} {freq}'
                            chfreq=diter[freqi]
                            spectra_signals[spname].append(chfreq)
                spectra_signals['timestamp']+=[timestamp for i in range(len(spectraldch))]

            if 'freqbins' in logkeys:
                for ch in bin_data:
                    ch_bins=bin_data[ch]
                    for bin in range(len(binnames)):
                        binname=binnames[bin]
                        curkey=f'{ch} {binname}'
                        cdata=ch_bins[bin]
                        binned_singals[curkey].append(cdata)
                binned_singals['timestamp'].append(timestamp)

    returndict={}
    if 'raw_signal' in logkeys:
        returndict['raw_signal']=crop_data_dict_and_make_df(raw_signals)
        if save_dfs:
            returndict['raw_signal'].to_csv('raw_signal.csv', index=False)
    if 'spectra' in logkeys:
        returndict['signal_spectra']=crop_data_dict_and_make_df(spectra_signals)
        if save_dfs:
            returndict['signal_spectra'].to_csv('spectra.csv', index=False)
    if 'freqbins' in logkeys:
        returndict['binned_signal']=crop_data_dict_and_make_df(binned_singals)
        if save_dfs:
            returndict['binned_signal'].to_csv('binned_signal.csv', index=False)
    if toreturn==True:
        if return_signal_options==True:
            return returndict, opts
        else:
            return returndict


@callback(Output('time_info','children'),
              Input('current_split_logdata', 'data'),
              Input('logplots_upd_interval', 'n_intervals'),
              Input('logdata_range_slider', 'value'),
              State('logged_data_timestamps','data'),
              prevent_initial_call=True)
def update_time_info(csplitlogd, n_intervals, range_slider_val, timestamps):
    init_ts=timestamps[0]
    end_ts=timestamps[1]
    if csplitlogd:
        int_start_time=csplitlogd[0]['timestamp'][0]
        int_end_time=csplitlogd[-1]['timestamp'][-1]
        ctime=csplitlogd[n_intervals-1]['timestamp'][-1]
    else:
        int_start_time='Unknown'
        int_end_time='Unknown'
        ctime='Unknown'
    int_start=timestamps[range_slider_val[0]]
    int_end=timestamps[range_slider_val[1]]
    return [dcc.Markdown(f'Currently selected interval: {int_start} - {int_end}'),
            dcc.Markdown(f'Whole logged data interval: {init_ts} - {end_ts}'),
            dcc.Markdown(f'The last processing task timeframe: {int_start_time} - {int_end_time}'),
            dcc.Markdown(f'The last plotted time: {ctime}')]



@callback(Output('logged_df', 'data'),
              Output('message', 'children', allow_duplicate=True),
              Output('logdata_range_slider', 'min'),
              Output('logdata_range_slider', 'max'),
              Output('signal_acquistion_options_data', 'data', allow_duplicate=True),
              Output('logdata_range_slider', 'value', allow_duplicate=True),
              Output('logged_data_timestamps','data'),
            #   Output('signal_processing_step', 'min'),
            #   Output('signal_processing_step', 'max'),
              Input('upload_logdata','contents'),
              Input('upload_current_logging_file','n_clicks'),
              State('save_dfs','value'),
              prevent_initial_call=True)
def upload_data(val, n_clicks, save_dfs):
    ctx=dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    else:
        if len(save_dfs)>0:
            save_dfs=True
        else:
            save_dfs=False
        input_id=ctx.triggered[0]['prop_id'].split('.')[0]
        if input_id == 'upload_current_logging_file':
            data, opts=parse_logfile(logfn, save_dfs=save_dfs)
        elif input_id == 'upload_logdata':
            if val is not None:
                content_type, content_string = val.split(',')
                decoded = base64.b64decode(content_string)
                data=decoded.decode('utf-8')
                data, opts=parse_logfile(data, input_is_filename=False)
            else:
                data = False # Empty DataFrame if no file is uploaded

        if 'raw_signal' in data:
            raw_json_data = data['raw_signal'].to_json(orient='split')
            minval=data['raw_signal'].index.tolist()[0]
            maxval=data['raw_signal'].index.tolist()[-1]
            timestamps=data['raw_signal']['timestamp'].tolist()
            return raw_json_data, f'Logging data uploaded. Initial signal acquisition options: {opts}', minval, maxval, opts, [minval, maxval], timestamps #, minval, maxval
        else:
            return False, 'No raw signal data in the logging data. Visualization for other data types is not implemented.',  0, 0, None, [0, 0], False #False # 0, 0 #, 0, 0

@callback(Output('signal_processing_step', 'value'),
              Input('signal_processing_step_is_whole_range', 'value'),
              Input('logdata_range_slider', 'value'),
              State('signal_processing_step', 'value'),
            #  State('signal_plotting','style'),
              prevent_initial_call=True)
def change_logdata_plotting_control_visibility(value, val, cmax):
    if len(value)>0:
        return max(val)-min(val)
    else:
        return cmax







@callback([Output(f'{ch}_raw_buffer', 'data', allow_duplicate=True) for ch in signal_chnames]+
              [Output(f'{ch}_fft_buffer', 'data', allow_duplicate=True) for ch in signal_chnames]+
              [Output(f'{ch}_raw_current', 'data', allow_duplicate=True) for ch in signal_chnames]+
              [Output(f'{ch}_fft_current', 'data', allow_duplicate=True) for ch in signal_chnames]+
              [Output(f'{ch}_bins_buffer', 'data', allow_duplicate=True) for ch in signal_chnames]+
              [Output(f'{ch}_bins_current', 'data', allow_duplicate=True) for ch in signal_chnames]+
              [Output(f'{ch}_maxfreq_buffer', 'data', allow_duplicate=True) for ch in signal_chnames]+
              [Output(f'{ch}_specfreq_current', 'data', allow_duplicate=True) for ch in signal_chnames]+
              [Output(f'{ch}_specfreq_buffer', 'data', allow_duplicate=True) for ch in signal_chnames]+
              [Output('current_split_logdata', 'data')]+
              [Output('message', 'children', allow_duplicate=True)]+
              [Output("signal_plotting_update_interval", 'value', allow_duplicate=True)]+
              [Output("signal_buffer_update_interval", 'value', allow_duplicate=True)]+
              [Output('logplots_upd_interval','n_intervals')]+
              [Output('logplots_upd_interval','max_intervals')]+
              [Output('logplots_upd_interval','interval')]+
#              [Output('logplots_upd_interval','disabled', allow_duplicate=True)]+
              [Output('start_log_plotting_switch','data')],


              Input("update_logplots_button", 'n_clicks'),
              State('logged_df', 'data'),
              State('logdata_range_slider', 'value'),
              State('signal_processing_step', 'value'),
              State('logdata_processing_interval', 'value'),
              State("update_logplots_button", 'children'),
            #  State('signal_plotting','style'),
              prevent_initial_call=True)
def clear_buffers(n_clicks, data, data_range, data_step, proc_step, button_text):
    if button_text=="Plot selected data...":
        data=pd.read_json(data, orient='split')
        wholelen=len(data)
        data=data.iloc[int(data_range[0]):int(data_range[1])]
        chunks = np.array_split(data, len(data) // data_step)
        dicts=[]
        for i, chunk in enumerate(chunks):
            chunk=chunk.to_dict(orient='list')
            dicts.append(chunk)
        return [[] for i in range(len(signal_chnames)*9)]+[dicts]+['Loaded the framed data to a buffer.']+[wholelen, wholelen]+[0, len(dicts), proc_step, True] #False
    else:
        raise PreventUpdate




@callback(Output('logplots_upd_interval','disabled', allow_duplicate=True),
              Output("update_logplots_button", 'children', allow_duplicate=True),
              Input("update_logplots_button", 'n_clicks'),
              State("update_logplots_button", 'children'),
              prevent_initial_call=True)
def stop_plotting(n_clicks, button_text):
    if button_text=='Stop plotting...':
        return True, "Plot selected data..."
    if button_text=="Plot selected data...":
        return False, 'Stop plotting...'


@callback(Output("update_logplots_button", 'children'),
              Input('logplots_upd_interval', 'n_intervals'),
              State('logplots_upd_interval', 'max_intervals'),
              State('logplots_upd_interval', 'disabled'),
              State("update_logplots_button", 'children'),
            #  State('signal_plotting','style'),
              prevent_initial_call=True)
def change_update_logplots_button_style(n_intervals, max_n_intervals, upd_disabled, cur_button_text):
    if upd_disabled:
        return "Plot selected data..."
    else:
        if n_intervals<max_n_intervals:
                return 'Stop plotting...'
        else:
            return "Plot selected data..."


@callback(Output('signal_joint_plot_container', 'children'),
              Output('signal_plot_container', 'children'),
              Input("clear_plots_button", 'n_clicks'),
              prevent_initial_call=True)
def clear_all_graphs(n_clicks):
    return [div_graph_fheatmap(), div_graph_fheatmap_max(), div_graph_fheatmap_spec(), div_graph_fastupdate_spec(), div_graph_rawheatmap()], [div_graph(chname) for chname in signal_chnames]

if __name__ == '__main__':
    app.run_server(debug=True, port=5002, dev_tools_hot_reload=False)



