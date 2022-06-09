import io
import base64
from base64 import b64encode
from jupyter_dash import JupyterDash
from dash import dcc, html, Input, Output, no_update
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

from Helper_functions import load_dict
import numpy as np
from Helper_functions import interactive_session
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
from Helper_functions import density_scatter

from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from scipy.stats import spearmanr

basepath = r"C:\Users\Thibaut Goldsborough\Documents\Seth_BoneMarrow\Data\BoneMarrow_smallerfile2"
readpath = basepath + "\\Raw_Images"
outpath = basepath + "\\Outputs"

channel_dict={'Ch11': 'AF647 PDC (Photoreceptors)','Ch10': 'CA10-BV605', 'Ch07': "SYTO40 (DNA)"}

image_dim=64 #Dim of the final images
nuclear_channel="Ch7"
cellmask_channel="Ch1_mask"
outpath2=basepath+"//Outputs"
df=pd.read_csv(outpath2+"\\cell_info.csv")
cell_names=df["Cell_ID"].to_numpy()
Prediction_Channels=['Ch07']
image_dict=load_dict(outpath2,cell_names,image_dim)
Channels=['Ch1']  #Channel to be fed to the NN
images_with_index = []
for image_i in image_dict:
    if len(image_dict[image_i].keys())>=len(Channels):
        image=cv.merge([image_dict[image_i][i] for i in Channels])
        images_with_index.append((int(image_i),image))
    else:
        print(image_i)
images=np.array([image[1] for image in images_with_index])
names=np.array([image[0] for image in images_with_index])
DNA_pos=df["DNA_pos"].to_numpy()
Touches_Boundary=df["Touches_boundary"].to_numpy()
labels=df[["Intensity_MC_"+channel for channel in Prediction_Channels]].to_numpy()



latent_df=pd.read_csv(outpath+"\\Resnet0.35477188.csv")
namesdf=latent_df["Cell_ID"].to_numpy()
display_images=[image_dict[int(i)]['Ch1'] for i in namesdf]
display_images-=np.min(display_images)
display_images/=np.max(display_images)
display_images*=255
display_images*=1.5

def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url



def interactive_session(u,display_images,colors,namesdf):

    dim=np.shape(u)[1]

    if np.min(display_images)<0:
        display_images=np.array((np.array(display_images)-np.min(display_images)))

    if str(type(display_images[0][0][0]))=="<class 'numpy.float64'>":
        display_images=display_images.astype(np.uint8)


   
    buffer = io.StringIO()

    if dim==3:
        fig = go.Figure(data=[go.Scatter3d(
            x=u[:, 0],
            y=u[:, 1],
            z=u[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=colors,
            )
        )])

    if dim==2:
        fig = go.Figure(data=[go.Scatter(
            x=u[:, 0],
            y=u[:, 1],
            mode='markers',
            marker=dict(
                size=5,
                color=colors,
            )
        )])


    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )

    fig.update_layout(template='plotly_dark', title="Retinal cell clustering (t-SNE)")
    fig.show()

    #fig.update_layout(
    #    scene=dict(
    #        xaxis=dict(range=[-10,10]),
    #        yaxis=dict(range=[-10,10]), 
    #    )
    #)

    fig.write_html(buffer)

    html_bytes = buffer.getvalue().encode()
    encoded = b64encode(html_bytes).decode()

    app = JupyterDash(__name__)

    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(style={'height': '800px'}, id="graph-5", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
            ], style={'width': '80%' , 'display': 'inline-block', 'vertical-align': 'middle'}
    ) 


    @app.callback(
        Output("graph-tooltip-5", "show"),
        Output("graph-tooltip-5", "bbox"),
        Output("graph-tooltip-5", "children"),
        Input("graph-5", "hoverData"),
    )

    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update


        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        num = hover_data["pointNumber"]

        im_matrix = display_images[num]
        im_url = np_image_to_base64(im_matrix)
        children = [
            html.Div([
                html.Img(
                    src=im_url,
                    style={"width": "100px", 'display': 'block', 'margin': '0 auto'},
                ),
                html.P("Cell ID: " + str(int(namesdf[num])))#, style={'font-weight': 'bold'})
            ])
        ]

        return True, bbox, children

    if __name__ == "Helper_functions":
    #  app.run_server(mode='inline', debug=True)
        port_num=np.random.randint(100)+8000
        app.run_server(mode='external', debug=True,use_reloader=False,port=port_num)

def plot_scatter(df,channels,heatmap=False,cmap='magma_r',save=False,interactive=False,**kwargs):
    #fig, axs = plt.subplots(int(np.ceil(len(channels)/2)),2,figsize=(int(np.ceil(len(channels)/2))*10,2*10))
    fig, axs = plt.subplots(int(np.ceil(len(channels)/2)),2,figsize=(17.5,17.5))

    for i,channel in enumerate(channels):
        ax_index=(i//2,i%2)
        if len(channels)<=2:
            ax_index=(i)
        axs[ax_index].set_xlim(np.min(df["SCALED_Intensity_MC_"+channel]),np.max(df["SCALED_Intensity_MC_"+channel]))
        axs[ax_index].set_ylim(np.min(df["SCALED_Intensity_MC_"+channel]),np.max(df["SCALED_Intensity_MC_"+channel]))
        
        axs[ax_index].set_aspect('equal', adjustable='box')
        x=df["SCALED_Intensity_MC_"+channel].to_numpy()
        y=df["PRED_Intensity_MC_"+channel].to_numpy()
        #np.random.shuffle(x)
        #np.random.shuffle(y)



        if not heatmap:
            axs[ax_index].scatter(x,y,s=1,alpha=0.5)
        else:
            x,y,z=density_scatter( x , y, sort = False, bins =[40,40])
            axs[ax_index].scatter(x,y,c=z,s=1,cmap=cmap,**kwargs)
            norm = Normalize(vmin = np.min(z), vmax = np.max(z))
            
        if interactive:
            u=np.vstack((x,y)).T
            interactive_session(u,display_images,z,namesdf)

            break
