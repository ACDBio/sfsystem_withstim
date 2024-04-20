import dash
import dash_bootstrap_components as dbc
import plotly.io as pio
import os
import threading
from dash import html
pio.templates.default = 'simple_white'
from flask import send_from_directory

# use_neuroplay=True
# if use_neuroplay:
#     neuroplay_loc='/home/biorp/NeuroPlayPro/NeuroPlayPro.sh'
#     def run_neuroplay():
#         neuroplay_loc = '/home/biorp/NeuroPlayPro/NeuroPlayPro.sh'
#         os.system(f'bash {neuroplay_loc}')
#     neuroplay_thread = threading.Thread(target=run_neuroplay)
#     neuroplay_thread.daemon = True
#     neuroplay_thread.start()

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.FLATLY])
server = app.server


navbar = dbc.NavbarSimple(
    dbc.DropdownMenu(
        [
            dbc.DropdownMenuItem(page["name"], href=page["path"])
            for page in dash.page_registry.values()
            if page["module"] != "pages.not_found_404"
        ],
        nav=True,
        label="Switch usage modes",
    ),
    brand="",
    color="BlueViolet",
    dark=True,
    className="mb-2",
)

app.layout = dbc.Container(
    [navbar, html.Div(), dash.page_container],
    fluid=True,
)



if __name__ == "__main__":
    app.run_server(debug=True)
@app.server.route('/suggestions/<path:filename>')
def serve_mp3(filename):
    return send_from_directory('suggestions', filename)