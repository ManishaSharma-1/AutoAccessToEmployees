from flask import Flask, render_template, request
from datetime import datetime  # add
import pandas
from eda import eda_train
from eda import eda_over
from sklearn.model_selection import StratifiedKFold,cross_validate,train_test_split
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///train.db'  # add
# db = SQLAlchemy(app)  # add


# add
# class train_data(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     ACTION = db.Column(db.String(80), nullable=False)
#     RESOURCE = db.Column(db.String(80), nullable=False)
#     MGR_ID = db.Column(db.String(80), nullable=False)
#     ROLE_ROLLUP_1 = db.Column(db.String(80), nullable=False)
#     ROLE_ROLLUP_2 = db.Column(db.String(80), nullable=False)
#     ROLE_DEPTNAME = db.Column(db.String(80), nullable=False)
#     ROLE_TITLE = db.Column(db.String(80), nullable=False)
#     ROLE_FAMILY_DESC = db.Column(db.String(80), nullable=False)
#     ROLE_CODE = db.Column(db.String(80), nullable=False)
#     return app



@app.route('/')
def index():
    # return "Hello World!"
    train_data = pandas.read_csv("data/train.csv")
    print(len(train_data))
    # print(train_data.head(1))
    return render_template('home.html',data=train_data.iloc[0])

@app.route('/submitRequest', methods=['GET', 'POST'] )
def test():
    testdata = {}
    testdata['resources'] = request.form.get("RESOURCE","")
    testdata['manager_id'] = request.form.get("MGR_ID","")
    testdata['ROLE_ROLLUP_1'] = request.form.get("ROLE_ROLLUP_1","")
    testdata['ROLE_ROLLUP_2'] = request.form.get("ROLE_ROLLUP_2","")
    testdata['ROLE_DEPTNAME'] = request.form.get("ROLE_DEPTNAME","")
    testdata['ROLE_TITLE'] = request.form.get("ROLE_TITLE","")
    testdata['ROLE_FAMILY_DESC'] = request.form.get("ROLE_FAMILY_DESC","")
    testdata['ROLE_CODE'] = request.form.get("ROLE_CODE","")


    # Run ML algorithm
    return render_template('submitRequest.html',data = testdata)


@app.route('/train')
def train_data():
    # return "Hello World!"
    train_data = pandas.read_csv("data/train.csv")
    print(len(train_data))
    # print(train_data.head(1))
    return render_template('train_data.html',data=train_data)



@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html')




@app.route('/run_model')
def run_model():
    return render_template('run_model.html')

def over_sampling(train_data):
    train_data.head()
    df= train_data
    X= train_data.drop(['ROLE_CODE'], axis =1)
    y = train_data["ACTION"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 
    sm = SMOTE(random_state = 2) 
    X_train_res, y_train_res = sm.fit_sample(X_train, y_train.values.ravel()) 
    df_new = pandas.DataFrame(X_train_res, y_train_res)
    df_new.columns = X_train.columns
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




