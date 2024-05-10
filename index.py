import pandas as pd
import numpy as np

import dash
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from dash import dash_table

import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt

from datetime import timedelta, date
from datetime import datetime
import random 

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# #####################################
# # Add your data
# #####################################

# emp  = pd.read_csv("Data_app/Employee.csv")
# cells = pd.read_csv("Data_app/Cell.csv")
# performance = pd.read_csv("./Data_app/Performance.csv")
# headcount = pd.read_csv('Data_app/Headcount.csv')

emp  = pd.read_pickle("Data_app/Employee.pkl")
cells = pd.read_pickle("Data_app/Cell.pkl")
performance = pd.read_pickle("./Data_app/Performance.pkl")
headcount = pd.read_pickle("Data_app/Headcount.pkl")

##################################################
## Functions
##################################################
def create_rand_emps(DF_in, n_employees):
    random.seed(100)
    DF_out = DF_in.iloc[random.sample(range(DF_in.shape[0]), n_employees)]
    return DF_out

def initialize_emp(DFin, n_employees=60):
    random.seed(100)
    employee_sub = DFin.iloc[random.sample(range(len(DFin['employee_id'])), n_employees),]
    employee_sub = employee_sub.sort_values(by="employee_id")
    return employee_sub

def find_correct_line(n_afac, n_manual, n_workers, df):
    df = df.sort_values(by='total worker req')
    df['Next'] = df['total worker req'].shift(-1)
    max_workers = df['total worker req'].max()
    find_row = df[(df['total worker req'] <= n_workers) & (df['Next'] > n_workers)]
    if not find_row.empty:
        new_afac = find_row['num_afac'].values[0]
        new_manual = find_row['num_manual'].values[0]
        workers = find_row['total worker req'].values[0]
    elif n_workers >=  max_workers:
        new_afac = df['num_afac'].max()
        new_manual = df['num_manual'].max()
        workers = df['total worker req'].max()
    else:
        new_afac = 0
        new_manual = 0
        workers = 0
    return new_afac, new_manual, workers

#### Functions to create table for constraint tab
def constraint_tbl(emp, head_tbl, perf_tbl, afac_x, manual_x, workers_x):
    new_afac, new_manual, workers = find_correct_line(afac_x, manual_x, workers_x, head_tbl)
    if new_afac==0 or new_manual==0 or workers==0:
        workers = workers_x
    emp = emp.iloc[:workers,:]
    emp = emp.iloc[:,:3]
    perf_tbl = perf_tbl[perf_tbl['employee_id'].isin(emp['employee_id'])]
    perf_tbl = perf_tbl[['employee_id', 'cell_id', 'training']]
    perf_tbl_pivot = perf_tbl.pivot(index='employee_id', columns='cell_id', values='training')
    perf_tbl_pivot = perf_tbl_pivot.reset_index()
    
    correct_row = head_tbl[(head_tbl['num_afac'] == new_afac) & (head_tbl['num_manual'] == new_manual)]
    correct_row_headers = correct_row.columns.difference(['num_afac', 'num_manual', 'total worker req']).tolist()
    
    # perf_tbl_pivot.columns = [perf_tbl_pivot.columns[0]] + correct_row_headers
    perf_tbl_pivot.columns = [perf_tbl_pivot.columns[0]] + [str(i) for i in range(1,len(correct_row_headers)+1)]
    return perf_tbl_pivot

# def extract_first_two_columns(csv_file):
#     """
#     Load a CSV file and extract the first two columns.
#     """
#     # Load the CSV file
#     data_df = pd.read_csv(csv_file)
    
#     # Extract the first two columns
#     first_two_columns = data_df.iloc[:, :2]
    
#     return first_two_columns



def find_ergo_score(result_in, cell_in):
    """
    This function calculates the "erg_effort" score for a list of workers
    Inputs:
    result_in - dataframe of model results with var_name (name of variable) and val (1 or 0) 
    cell_in - dataframe with a column that indicates the erg_effort of a given station
    Output:
    result_out - dataframe with var_name (name of variable) and ergo score
    Assumption:
    low = 3
    medium = 2
    high = 1
    """
    def convert_erg_score(value):
        if value=='low':
            return 3
        elif value=='medium':
            return 2
        elif value=='high':
            return 1
        else:
            return 0
    convert_erg_score_vect_func = np.vectorize(convert_erg_score)
    cell_in['erg_score'] = convert_erg_score_vect_func(cell_in['erg_effort']) 
    
    result_in = result_in[result_in['val']!=0]
    result_in.loc[:,'var_name'] = result_in['var_name'].astype(str)

    def get_station(station_in):
        return station_in.split(',')[-1]
    get_station_vect_func = np.vectorize(get_station)
    result_in.loc[:,'cell_id'] = '0'
    result_in.loc[:,'cell_id'] = get_station_vect_func(result_in['var_name'])
    result_in.loc[:,'cell_id'] = result_in.loc[:,'cell_id'].astype(int)
    result_out = pd.merge(result_in, cell_in[['cell_id','erg_score']], on='cell_id', how='left')
    
    return result_out


def output_format(df):
    def get_station(station_in):
        return station_in.split(',')[-1]
    get_station_vect_func = np.vectorize(get_station)
    df['station'] = '0'
    df.loc[:, 'station'] = get_station_vect_func(df['var_name'])
    ##Rotation Index
    def get_rotation(rotation_in):
        return rotation_in.split(',')[-2]
    df['rotation'] = '0'
    get_rotation_vect_func = np.vectorize(get_rotation)
    df.loc[:, 'rotation'] = get_rotation_vect_func(df['var_name'])

    ##Employee Index
    def get_employee(employee_in):
       return employee_in.split('_')[1].split(',')[0]
    get_employee_vect_func = np.vectorize(get_employee)
    # df['employee'] = df['employee'].apply(get_employee)
    df['employee']='0'
    df.loc[:, 'employee'] = get_employee_vect_func(df['var_name'])

    df = df.drop(columns=['var_name', 'val'])
    df_tbl = df.pivot(index="employee", columns="rotation", values="station")
    df_tbl = df_tbl.reset_index()
    df_tbl.loc[:,'employee'] = df_tbl.loc[:,'employee'].astype(int)
    return df_tbl


###########################################################
### Optimization Function
###########################################################
import pulp

def run_opt_mod_ver1(n_afac, n_manual, n_workers, n_rotation, cell_sub, perf_sub, emp_sub, head_sub):
    """
    Approach:
    Main decision variable is X_{i,j,k} where 
    i: employee i
    j: rotation j
    k: station k for a given rotation j
    Start with j=1 for simplicity
    """ 
    # n_afac, n_manual, n_workers = find_correct_line(n_afac, n_manual, n_workers, head_sub)

    # emp_sub = emp_sub.iloc[:n_workers,:]
    perf_sub = perf_sub[perf_sub['employee_id'].isin(emp_sub['employee_id'])]

    ### Determine the number of workers necessary for the number of lines
    head_tbl = head_sub[(head_sub['num_afac']==n_afac) & (head_sub['num_manual']==n_manual)]
    head_tbl = head_tbl.drop(columns=['num_afac', 'num_manual','total worker req'])    
    head_tbl = head_tbl.melt(var_name='cell_name', value_name='total_cell_headcount')
    head_tbl['cell_id'] = range(1,len(head_tbl)+1)

    cell_col_list = cell_sub.columns
    cell_sub= cell_sub.drop(columns='total_cell_headcount')
    cell_sub = pd.merge(cell_sub, head_tbl[['cell_id','total_cell_headcount']], on='cell_id', how='left')
    cell_sub = cell_sub[cell_col_list]
    cell_sub['total_cell_headcount'] = cell_sub['total_cell_headcount'].fillna(0)
    n_workers_required = cell_sub['total_cell_headcount'].sum()

    cell_sub
    model = pulp.LpProblem("Sample-LP-Problem", pulp.LpMaximize)

    ### Create X variables for each worker i in rotation j, station k
    varX = []
    for emp in emp_sub['employee_id'].unique():
        for shift in range(n_rotation):
            for index in range(len(cell_sub['cellgroup'])):
                current_row = cell_sub.iloc[index,]
                j = shift
                k = current_row['cell_id']
                varX.append(str(emp)+","+str(j)+","+str(k))

    X_variables = pulp.LpVariable.matrix("X", varX, cat=pulp.LpBinary)
    n_stations = len(cell_sub['cell_id'].unique())
    X_var = np.array(X_variables).reshape(len(emp_sub['employee_id'].unique()), n_rotation, n_stations)

    perf_sub.columns = perf_sub.columns.map(lambda x: x.lower())
    cell_sub.columns = cell_sub.columns.map(lambda x: x.lower())

    ### Add constraints with respsect to Xijk's
    ## x[employee][rotation][station]
    # X[1][0][7]

    # Look at sequence in cell
    Seq_matrix = np.zeros(len(emp_sub['employee_id'].unique()) * n_rotation * n_stations).reshape(len(emp_sub['employee_id'].unique()), n_rotation, n_stations)
    for shift in range(n_rotation):
        for index in cell_sub['cell_id']:
            Seq_matrix_sub = Seq_matrix.copy()
            rotation = shift
            emp_count = 0
            for emp in emp_sub['employee_id']:
                emp_count += 1
                # Constraint 1: if employee is not trained, employee will not be assigned that station
                performance_sub_match = perf_sub[perf_sub['employee_id']==emp]
                if(not(performance_sub_match[(performance_sub_match['training']==1) & (performance_sub_match['cell_id']==index)].empty)):
                    Seq_matrix_sub[emp_count-1][shift][index-1] = 1
            # print(index)
            # print(pulp.lpSum(Seq_matrix_sub * X_var))
            model += pulp.lpSum(Seq_matrix_sub * X_var) >= cell_sub.loc[cell_sub['cell_id']==index,'total_cell_headcount'].values[0]

    ##### Every operator is assigned to a station
    for emp in range(len(emp_sub['employee_id'])):
        for shift in range(n_rotation):
            model += pulp.lpSum(X_var[emp][shift][seq] for seq in range(n_stations)) <= 1
    
    #### Ergo constraint
    ## First, convert 'low', 'medium', 'high' to integers
    def convert_erg_score(value):
            if value=='low':
                return 3
            elif value=='medium':
                return 2
            elif value=='high':
                return 1
            else:
                return 0

    ### Need to vectorize to use on a numpy object    
    convert_erg_score_vect_func = np.vectorize(convert_erg_score)
    cell_sub['erg_score'] = convert_erg_score_vect_func(cell_sub['erg_effort']) 

    ### make for the whole day
    ### create matrix for ergo constraint
    erg_ef = pd.merge(perf_sub, cell_sub[['cell_id', 'erg_score']], on="cell_id", how="left")
    erg_ef = erg_ef.sort_values(by=["employee_id","cell_id"])
    min_erg_score = 2      
    #max_erg_score  = 2
    erg_ef = erg_ef['erg_score']
    erg_ef = np.array(erg_ef)
    erg_ef = erg_ef.reshape(n_workers,n_stations)
    erg_ef_vect = erg_ef[0]

    for emp in range(len(emp_sub['employee_id'])):
        model += pulp.lpSum([X_var[emp][shift]*erg_ef_vect for shift in range(n_rotation)]) >=  n_rotation*min_erg_score

    ### blocks for "high" stations
    high_block_size = 2
    ### blocks for "medium" stations
    medium_block_size = 2

    for index in cell_sub['cell_id']:
        emp_count = 0
        for emp in emp_sub['employee_id']:
            emp_count += 1
            for shift in range(n_rotation):
                shift_left = n_rotation - shift
                if cell_sub.loc[index-1,'erg_effort']=='high':
                    shift_left = min(high_block_size, shift_left)
                    print(pulp.lpSum([X_var[emp_count-1, shift_i, index-1] for shift_i in range(shift, shift+shift_left)]) <= 1)
                    model += pulp.lpSum([X_var[emp_count-1, shift_i, index-1] for shift_i in range(shift, shift+shift_left)]) <= 1
                elif cell_sub.loc[index-1,'erg_effort']=='medium':
                    shift_left = min(medium_block_size, shift_left)
                    print(pulp.lpSum([X_var[emp_count-1, shift_i, index-1] for shift_i in range(shift, shift+shift_left)]) <= 1)
                    model += pulp.lpSum([X_var[emp_count-1, shift_i, index-1] for shift_i in range(shift, shift+shift_left)]) <= 1

    cost_df = pd.merge(perf_sub, cell_sub[['cell_id', 'hourly_rate']], on="cell_id", how="outer")
    cost_df = cost_df.sort_values(by=["employee_id","cell_id"])
    cost_df['perf_avg'] = cost_df['output_factor'] * cost_df['hourly_rate']
    cost_vector = cost_df['perf_avg'].to_numpy()
    cost_vector = np.tile(cost_vector, n_rotation)
    ## Obj Function
    cost_vector = cost_vector.reshape(len(emp_sub['employee_id'].unique()), n_rotation, n_stations)
    obj_func = pulp.lpSum(cost_vector*X_var)

    model += obj_func

    # model.writeLP("version1.lp")

    model.solve(pulp.PULP_CBC_CMD())
    status = pulp.LpStatus[model.status]

    if status == "Optimal":
        total_obj_val = model.objective.value()
    else:
        total_obj_val = "Infeasible"
        
    varsdict = {}
    for v in model.variables():
        varsdict[v.name] = v.varValue
        
    result_df = pd.DataFrame(list(varsdict.items()), columns=['var_name', 'val'])
    result_df = result_df[result_df['val']>0]

    return (result_df, total_obj_val)


employee_sub = initialize_emp(emp)
performance_sub = performance[performance['employee_id'].isin(employee_sub['employee_id'])]

AFAC_options = [6]
Manual_options = [0,1,2,3,4,5,6,7,8]
max_rotation = 10
max_workers = 60


#####################################
# Styles & Colors
#####################################

NAVBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "12rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "top":0,
    "margin-top":'2rem',
    "margin-left": "14rem",
    "margin-right": "2rem",
}

SIDEBAR_STYLE = {
    # "position": "fixed",  # comment this out
    "top": 42,
    "left": 0,
    "bottom": 0,
    "background-color": "#f8f9fa",
    "overflowY": "auto",
}

#####################################
# Create Auxiliary Components Here
#####################################


def side_bar():
    """
    Create side bar
    """
    side_controls = dbc.Card(
    [
        html.H5("Operator Assignment Inputs", className="display-10",style={'textAlign':'center'}),
#       html.Hr(),
        html.Div(
            [
                dbc.Label("AFAC Lines:"),
                dcc.Dropdown(
                    id="AFAC_input_value",
                    options=AFAC_options,
                    value=6,
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Manual Lines:"),
                dcc.Dropdown(
                    id="Manual_input_value",
                    options=Manual_options,
                    value=6,
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Number of Rotations:"),
                dbc.Input(id="Rotation_input_value", type="number", value=10, min=1),
            ]
        ),
        html.Div(
            [
                html.H6(["Current max workers: ",str(max_workers)]),
                dbc.Label("Number of workers present (min=16):"),
                dbc.Input(id="Workers_input_value", type="number", value=46),
            ]
        ),
        html.Hr(),
        html.Button('Populate', id='Populate_side_button', n_clicks=0)
    ],
    body=True,
    style=SIDEBAR_STYLE,
    )
    return side_controls

#####################################
# Create Page Layouts Here
#####################################

### Layout 1
layout1 = html.Div( 
    dbc.CardBody(
        html.Div([
            html.H1("Workers Present"),
            html.Hr(),
            html.H5("Workers who have checked into the floor."),
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.Div(
                            [
                            html.Div(id="worker_table_tab1"), 
                            dcc.Store(id="worker_table_active"),
                            ]
                        ),
                    ], width=6),
                ]),
            ])
        ])
    )
)

layout2 = html.Div([
    html.H1("Set contraints for optimization"),
    html.Hr(),
    html.H5("Nonzero value indicates that the worker is eligible to work at the station/cell."),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div(
                    [
                        html.Button('Update Constraints', id='update_const_button', n_clicks=0),
                        html.Hr(),
                        html.Div(id="const_table_tab2"), 
                        dcc.Store(id="const_table_active"),
                        dcc.Store(id="perf_table_active"),
                    ]
                )
            ], width=10),
        ])
    ])
])

layout3 = html.Div([
    html.H1("Optimization Results"),
    html.Hr(),
    html.H5("This tab shows the results of the optimization model."),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4(children=["AFAC:   ", html.Span(id='n_afac_mod')]),
                    dcc.Store('n_afac_mod_active'),
                    html.H4(children=["Manual: ", html.Span(id='n_manual_mod')]),
                    dcc.Store('n_manual_mod_active'),
                    html.H4(children=["Assigned Workers: ", html.Span(id='n_workers_mod')]),
                    dcc.Store('n_workers_mod_active'),
                    html.H4(children=["DexFlex Workers: ", html.Span(id='n_dexflex')]),
                    dcc.Store('n_dexflex_active'),
                ]), 
            ], width=6),
            dbc.Col([
                html.Div([
                    html.H4(children=["Missed Capacity:           ", html.Span(id='missed_capacity')]),  
                    html.H4(children=["Additional Workers Needed: ", html.Span(id='add_worker_need')]),
                ])
            ], width=6),
        ]),
        html.Hr(),
        dbc.Row([ 
            dbc.Col([
                html.Div([
                    html.H4(children=["Obj Val: ", html.H4(id="result_obj_val_tab3")]),
                    html.H4("Result Table:"),
                    html.Div(id="result_table_tab3"), 
                    dcc.Store(id="result_table_active"),
                    dcc.Store(id="result_obj_val_active")
                ])
            ], width=6),
            dbc.Col([
                html.Div([
                ]),
            ], width=1,),
            dbc.Col([  
                 html.H4("Station Name Key:"),
                # Replace the placeholder with the actual table generated from cell data
                html.Div([
                    dash_table.DataTable(
                        id='cell_tbl',
                        columns=[{"name": i, "id": i} for i in cells.columns[:2]],
                        data=cells.to_dict('records')
                    )
                ])
            ], width=5),
        ]),
    ])
])

test_tab1 = html.Div([
    html.H1("show worker table"),
    html.Hr(),
    dbc.Container([
        html.Div(
            [
                html.H5("Hello"),
                html.Div(id="table_check_col1"),
            ]
        ),
    ]),
])
test_tab2 = html.Div([
    html.H1("Check for AFAC and Manual Lines"),
    html.Hr(),
    dbc.Container([
        html.Div(
            [
                html.Div(id="perf_table_check")
            ]
        ),
    ]),
])


app.layout = html.Div([
    dbc.Row([
        dbc.Col(side_bar(), width=3),
        dbc.Col(
            html.Div(
                dcc.Tabs(id="tabs-main", value="tab1", children=[
                    dcc.Tab(layout1,
                            label = "Workers",
                            value = 'tab1',
                            ),
                    dcc.Tab(layout2,
                            label = "Constraints",
                            value = 'tab2',
                            ),
                    dcc.Tab(layout3,
                            label = "Results",
                            value = 'tab3',
                            ),
                    # dcc.Tab(test_tab1,
                    #         label = "show worker table",
                    #         value = 'test-tab-1',
                    #         ),
                    # dcc.Tab(test_tab2,
                    #         label = "Check for AFAC and Manual Lines",
                    #         value = "test-tab-2",
                    #         )     
                ])
            ),
            width=9,
        )
    ])
])

@app.callback(Output("content", "children"), Input("tabs-main", 'value'))
def render_content(tabX):
    if tabX=="tab-1":
        return layout1
    elif tabX=="tab-2":
        return layout2
    elif tabX=="tab-3":
        return layout3
    elif tabX=="test-tab-1":
        return test_tab1
    elif tabX=="test-tab-2":
        return test_tab2
    else:
        return html.P("Error")


######################################################################
## Callbacks
######################################################################

@app.callback(Output('worker_table_active', 'data'),
              Input('Workers_input_value', 'value'))
def update_emp_table_table(value):
    df = create_rand_emps(employee_sub , value)
    df.sort_values(by='employee_id', inplace=True)
    return df.to_json(date_format='iso', orient='split')


@app.callback(Output("worker_table_tab1", 'children'),
              Input('worker_table_active', 'data'))
def show_emp_table_tabl(df):
    df = pd.read_json(df, orient='split')
    df = df.iloc[:, 0:3]
    return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], 
                                row_selectable="multi",editable=False)


@app.callback(Output("table_check_col1", 'children'),
              [Input('worker_table_active', 'data')],
            #   [Input('worker_table_tab_alt', 'derived_virtual_data')]
            )
def show_emp_table_check(df):
    df = pd.read_json(df, orient='split')
    if df.empty:
        return "No data"
    else:
        return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], 
                                row_selectable="multi",editable=False)



@app.callback(Output("const_table_active", 'data'),
              [Input('worker_table_active', 'data'),
               Input('AFAC_input_value', 'value'),
               Input('Manual_input_value', 'value'),
               Input('Workers_input_value', 'value')])
def update_const_table(data, afac_x, manual_x, workers_x):
    df_in = pd.read_json(data, orient='split')
    # perf_in = pd.read_json(perf_in, orient='split')
    df = constraint_tbl(df_in, headcount, performance_sub, afac_x, manual_x, workers_x)
    return df.to_json(date_format='iso', orient='split')


@app.callback(Output("const_table_tab2", 'children'),
                Input('const_table_active', 'data'),)
def show_const_table_tab2(df):
    df = pd.read_json(df, orient='split')
    if df.empty:
         return "No data"
    else:
        return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], 
                                    editable=True)

@app.callback(
    Output('n_afac_mod', 'children'),
    Output('n_manual_mod', 'children'),
    Output('n_workers_mod', 'children'),
    Output('n_dexflex', 'children'),
    [Input('n_afac_mod_active', 'data'),
     Input('n_manual_mod_active', 'data'),
     Input('n_workers_mod_active', 'data'),
     Input('n_dexflex_active', 'data')]
)
def show_lines(val1, val2, val3, val4):
    return val1, val2, val3, val4


@app.callback(
    Output('n_afac_mod_active', 'data'),
    Output('n_manual_mod_active', 'data'),
    Output('n_workers_mod_active', 'data'),
    Output('n_dexflex_active', 'data'),
    [Input('AFAC_input_value', 'value'),
     Input('Manual_input_value', 'value'),
     Input('Workers_input_value', 'value')]
)
def update_lines(value1, value2, value3):
    afac, manual, workers = find_correct_line(value1, value2, value3, headcount)
    dexflex = value3 - workers
    return afac, manual, workers, dexflex


@app.callback(Output("result_table_active", 'data'),
              Output("result_obj_val_active", 'data'),
              [Input('Populate_side_button', 'n_clicks'),
               Input('worker_table_active', 'data'),
            #    Input('perf_table_active', 'data'),
               Input('AFAC_input_value', 'value'),
               Input('Manual_input_value', 'value'),
               Input('Rotation_input_value', 'value'),
               Input('Workers_input_value', 'value')])
def update_result_table(n_clicks, data, afac_x, manual_x, rotation_x, workers_x):
    if n_clicks>0:
        emp = pd.read_json(data, orient='split')
        new_afac, new_manual, workers = find_correct_line(afac_x, manual_x, workers_x, headcount)
        if new_afac==0 or new_manual==0 or workers==0:
            workers = workers_x
        emp = emp.iloc[:workers,:]
        # perf_in = pd.read_json(perf_in, orient='split')
        perf_in = performance_sub
        (df_result, obj_val) = run_opt_mod_ver1(new_afac, new_manual, workers, rotation_x, cells, perf_in, emp, headcount)
        # return obj_val, df_result.to_json(date_format='iso', orient='split')
        return df_result.to_json(date_format='iso', orient='split'), obj_val
       
@app.callback(Output("result_table_tab3", 'children'),
                Input('result_table_active', 'data'))
def show_result_table_tab3(df):
    df = pd.read_json(df, orient='split')
    if df.empty:
         return "No data"
    else:        
        ergo_result = find_ergo_score(df, cells)
        ergo_result['var_name'] = ergo_result['var_name'].astype('str')
        def get_var(var_in):
            var_in = var_in.split(',')[0]
            return var_in.split('_')[1]
        get_var_vect_func = np.vectorize(get_var)
        ergo_result.loc[:,'employee'] = '0'
        ergo_result.loc[:,'employee'] = get_var_vect_func(ergo_result['var_name']) 
        ergo_result = ergo_result.groupby(['employee'])['erg_score'].mean()
        ergo_result = ergo_result.reset_index()
        ergo_result['employee']  = ergo_result['employee'].astype('int')
        ergo_result = ergo_result.sort_values(by='employee')

        df = output_format(df)
        df = df.merge(ergo_result, on="employee", how="left")
        df = df.sort_values(by="employee")

        return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], 
                                    editable=False)

@app.callback(Output("result_obj_val_tab3", 'children'),
              Input('result_obj_val_active', 'data'))
def show_obj_val_tab3(val):
    if isinstance(val, (int, float)):
        return round(val,2)
    else:
        return "Infeasible"

## Station Callback
@app.callback(
    Output('cell_tbl', 'children'),
    [Input('cells', 'data')]
)

def calculate_ergo_score(df):
    df = pd.read_json(df, orient='split')
    if df.empty:
         return "No data"
    else:
        ergo_result = find_ergo_score(df, cells)
        ergo_result['var_name'] = ergo_result['var_name'].astype('str')
        def get_var(var_in):
            var_in = var_in.split(',')[0]
            return var_in.split('_')[1]
        get_var_vect_func = np.vectorize(get_var)
        ergo_result.loc[:,'employee'] = '0'
        ergo_result.loc[:,'employee'] = get_var_vect_func(ergo_result['var_name']) 
        ergo_result = ergo_result.groupby(['employee'])['erg_score'].mean()
        ergo_result = ergo_result.reset_index()
        ergo_result['employee']  = ergo_result['employee'].astype('int')
        ergo_result = ergo_result.sort_values(by='employee')
        return dash_table.DataTable(ergo_result.to_dict('records'), [{"name": i, "id": i} for i in ergo_result.columns], 
                                    editable=False)


@app.callback(Output('missed_capacity', 'children'),
              Output('add_worker_need', 'children'),
              [Input('worker_table_active', 'data'),
               Input('result_table_active', 'data'),
               Input('result_obj_val_active', 'data'),
               Input('n_afac_mod_active', 'data'),
               Input('n_manual_mod_active', 'data'),
               Input('n_dexflex_active', 'data'),
               Input('Workers_input_value', 'value'),
               Input('Rotation_input_value', 'value')])
def update_missed_capacity(data, result_df, obj_val, n_afac, n_manual, n_dexflex, n_workers, n_rotation):
    emp = pd.read_json(data, orient='split')
    results = pd.read_json(result_df, orient='split')
    perf_in = performance_sub
    if isinstance(obj_val, (int, float)):
        if n_afac==max(AFAC_options) and n_manual==max(Manual_options):
            update_missed_capacity = "No missed capacity"
        else:
            next_manual = n_manual + 1
            df = headcount
            df = df.sort_values(by='total worker req')
            find_row = df[(df['num_afac'] == n_afac) & (df['num_manual'] == next_manual)]
            if not find_row.empty:
                new_afac = find_row['num_afac']
                new_manual = find_row['num_manual']
                workers = find_row['total worker req']
                add_workers_req = workers - n_workers
                update_missed_capacity = 159*n_rotation 
            else:
                update_missed_capacity = "No missed capacity"
                add_workers_req = 0
        return update_missed_capacity, add_workers_req
    else:
        return "Unavailable", 0


server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)

# if __name__ == '__main__':
#     app.run_server(port=5000, host= '127.0.01',debug=True)