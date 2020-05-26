import plotly
import json
import plotly.graph_objects as go
import pandas


def classimbalance(x):
    x_val = ['Action Denied','Action Accepted']
    colors = ['lightslategray','crimson'] 

    x_0 = len(x[x['ACTION'] == 0])
    x_1 = len(x[x['ACTION'] == 1])
    fig = go.Figure([go.Bar(x=x_val, y = [x_0,x_1], text=[x_0,x_1],textposition='auto',  marker_color=colors)])
    fig.update_layout(title_text='Count of Action Denied and Action Accepted - Clearly Class Imbalance!')
    a = fig.to_json()
    return json.loads(json.dumps(a,cls=plotly.utils.PlotlyJSONEncoder))

def uniqueCategories(x):
    unique = pandas.DataFrame([(col,x[col].nunique()) for col in x.columns], 
                           columns=['Columns', 'Count of Category'])
    colors = ['lightslategray'] 
    fig = plotly.graph_objects.Figure([go.Bar(x=unique.Columns, y = unique['Count of Category'], text=unique['Count of Category'],textposition='auto',  marker_color=colors)])
    fig.update_layout(title_text='Unique categories - Train Data Set!')
    a = fig.to_json()
    return json.loads(json.dumps(a,cls=plotly.utils.PlotlyJSONEncoder))

def eda_train(train_data):

    graphs = {}
    class_imbalance = classimbalance(train_data)
    uniqueCategory = uniqueCategories(train_data)


    graphs['Class Imbalance'] = class_imbalance
    graphs['Unique Categories'] = uniqueCategory

    return graphs



