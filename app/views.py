from flask import Flask, render_template, request
from datetime import datetime  # add
import pandas
from eda import eda_train
from eda import eda_over
from sklearn.model_selection import StratifiedKFold,cross_validate,train_test_split
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score
import plotly.figure_factory as ff
import json
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import roc_curve, auc
import plotly
import random


app = Flask(__name__)

@app.route('/')
def index():
    train_data = pandas.read_csv("data/train.csv")

    # return "Hello World!"
    print(len(train_data))
    # print(train_data.head(1))
    return render_template('home.html',data=train_data.iloc[0])

@app.route('/submitRequest', methods=['GET', 'POST'] )
def test():
    testdata = {}
    testdata['RESOURCE'] = request.form.get("RESOURCE","")
    testdata['manager_id'] = request.form.get("MGR_ID","")
    testdata['ROLE_ROLLUP_1'] = request.form.get("ROLE_ROLLUP_1","")
    testdata['ROLE_ROLLUP_2'] = request.form.get("ROLE_ROLLUP_2","")
    testdata['ROLE_DEPTNAME'] = request.form.get("ROLE_DEPTNAME","")
    testdata['ROLE_TITLE'] = request.form.get("ROLE_TITLE","")
    testdata['ROLE_FAMILY_DESC'] = request.form.get("ROLE_FAMILY_DESC","")
    testdata['ROLE_CODE'] = request.form.get("ROLE_CODE","")
    testdata["Output"] = random.randint(0, 1)

    # Run ML algorithm
    return render_template('submitRequest.html',data = testdata)


@app.route('/train')
def train_data():
    train_data = pandas.read_csv("data/train.csv")
    print(len(train_data))
    return render_template('train_data.html',data=train_data)

def evaluate_model(train_data):
    plotlist = []
   

    model.fit(X_train_res, y_train_res)
    predicted_labels = model.predict(X_test)
    average_precision = average_precision_score(y_test, predicted_labels)
    tn, fp, fn, tp  = confusion_matrix(y_test, predicted_labels).ravel()
    fig = plt.figure(figsize=(10,8))
    average_precision = average_precision_score(y_test, predicted_labels)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()


    fpr[2], tpr[2], _ = roc_curve(y_test, predicted_labels)
    roc_auc[2] = auc(fpr[2], tpr[2])
    
    fig = plt.figure()
    lw = 2
    fig = plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    fig = plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    fig = plt.xlim([0.0, 1.0])
    fig = plt.ylim([0.0, 1.05])
    fig = plt.xlabel('False Positive Rate')
    fig = plt.ylabel('True Positive Rate')
    fig = plt.title('Receiver operating characteristic example')
    fig = plt.legend(loc="lower right")
    
    buf = io.BytesIO()
    fig.figure.savefig(buf, format = "png", dpi = 600, box_inches = "tight", pad_inches = 0)
    plotlist.append(buf.getvalue())

    return predicted_labels, tn, fp, fn, tp, plotlist



def plot_confusion_matrix(model_name):


    matrix = pandas.read_csv("Z"+model_name+".csv")
    tn = matrix['tn'][0]
    fp = matrix['fp'][0]
    fn = matrix['fn'][0]
    tp = matrix['tp'][0]

    z = [[tn,fp],[fn,tp]]

    x = [0,1]
    y = [0,1]

    z_text  = [['True Negative : ' + str(tn),'False Positive : ' + str(fp)],['False Negative : ' + str(fn),'True Positive : ' + str(tp)]]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                     )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            x=-0.1,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    a = fig.to_json()
    return json.loads(json.dumps(a,cls=plotly.utils.PlotlyJSONEncoder))

@app.route('/evaluation')
def evaluation():
    plotlist = plot_auc("LogisticRegression")
    plotly_p = plot_confusion_matrix("LogisticRegression") 
    plotlist_knn = plot_auc("KNN")
    plotly_p_knn = plot_confusion_matrix("KNN") 
    plotlist_random = plot_auc("RandomForestClassifier")
    plotly_p_random = plot_confusion_matrix("RandomForestClassifier") 

    data_mpl = {}
    data_mpl['LogisticRegression - AUC_ROC'] = base64.b64encode(plotlist[0]).decode('utf-8')
    data_mpl['KNN - AUC_ROC'] =  base64.b64encode(plotlist_knn[0]).decode('utf-8')
    data_mpl['RandomForestClassifier -  AUC_ROC'] =  base64.b64encode(plotlist_random[0]).decode('utf-8')
    data_mpl['LogisticRegression - ConfusionMatrix'] = plotly_p
    data_mpl['KNN - ConfusionMatrix'] =  plotly_p_knn
    data_mpl['RandomForestClassifier -  ConfusionMatrix'] =  plotly_p_random



    return render_template('evaluation.html' , plot_mpl = data_mpl)
   
    
def plot_auc(model_name):
    


    fpr = pandas.read_csv("FPR"+model_name+".csv")
    tpr = pandas.read_csv("TPR"+model_name+".csv")

    roc_auc = auc(fpr['0'], tpr['0'])
    
    fig = plt.figure()
    lw = 2
    fig = plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    fig = plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    fig = plt.xlim([0.0, 1.0])
    fig = plt.ylim([0.0, 1.05])
    fig = plt.xlabel('False Positive Rate')
    fig = plt.ylabel('True Positive Rate')
    fig = plt.title('Receiver operating characteristic example')
    fig = plt.legend(loc="lower right")

    plotlist = []
    buf = io.BytesIO()
    fig.figure.savefig(buf, format = "png", dpi = 600, box_inches = "tight", pad_inches = 0)
    plotlist.append(buf.getvalue())

    return plotlist


@app.route('/run_model')
def run_model():
    return render_template('run_model.html')


def grouping_and_onehot(train_data):

    train_data = train_data.drop(columns = 'ROLE_CODE')
    train_data['RESOURCE'] = train_data['RESOURCE'].astype('category')
    train_data['MGR_ID'] = train_data['MGR_ID'].astype('category')
    train_data['ROLE_ROLLUP_1'] = train_data['ROLE_ROLLUP_1'].astype('category')
    train_data['ROLE_ROLLUP_2'] = train_data['ROLE_ROLLUP_2'].astype('category')
    train_data['ROLE_DEPTNAME'] = train_data['ROLE_DEPTNAME'].astype('category')
    train_data['ROLE_FAMILY_DESC'] = train_data['ROLE_FAMILY_DESC'].astype('category')
    train_data['ROLE_FAMILY'] = train_data['ROLE_FAMILY'].astype('category')
    train_data['ROLE_TITLE'] = train_data['ROLE_TITLE'].astype('category')
    df = pandas.DataFrame(train_data.groupby('ROLE_TITLE'))
    dfa = pandas.DataFrame()
    for i in range(len(df)):
        df_new1 = pandas.get_dummies(df[1][i])
        dfa = dfa.append(df_new1)

    return dfa

def over_sampling(train_data):
    X = train_data.iloc[:,1:]
    y = train_data.iloc[:,:1]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 
    sm = SMOTE(random_state = 2) 
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train.values.ravel()) 
    df_new = pandas.DataFrame(X_train_res,y_train_res)
    df_new.columns = X_train.columns
    df_new.index = df_new.index.rename('ACTION') 
    df_new = df_new.reset_index(drop =False)
    return X_train_res, y_train_res,df_new


@app.route('/model_eda')
def model_eda():
    train_data = pandas.read_csv("data/train.csv")
    graphs = {}
    graphs = eda_train(train_data)
    print(len(graphs))
    X_train_res, y_train_res,df_new = over_sampling(train_data)
    graphs = eda_over(df_new,graphs)
    return render_template('model_eda.html', data=graphs)


if __name__ == '__main__':
    
    app.run(debug=True)
    




