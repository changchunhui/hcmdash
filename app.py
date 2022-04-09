# Imports
import pandas as pd
import datetime as dt
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Load data and parse all date columns
df = pd.read_csv('data/hr_dataset_v5.csv', parse_dates=['DOB','DateofHire','DateofTermination','LastPerformanceReview_Date'])

######################################## DATA PREP AND CLEAUP ########################################

# Create column for year of birth (to use in lambda function)
df['DOB_year'] = df['DOB'].dt.year

# Create column for age generation using lambda function
df['Generation'] = df['DOB_year'].apply(lambda x: 'Silent Generation' if x >= 1928 and x <= 1945 else ('Baby Boomers' if x >= 1946 and x <= 1964 else ('Generation Jokes' if x >= 1955 and x <= 1965 else ('Generation X' if x >= 1965 and x <= 1980 else ('Xennials' if x >= 1977 and x <= 1983 else ('Millenials' if x >= 1981 and x <= 1996 else ('Generation Z' if x >= 1997 and x <= 2009 else ('Generation Alpha' if x >= 2010 else np.NaN))))))))

# Drop 'DOB_year'
df.drop(columns='DOB_year', inplace=True)

# Rename gender column
df.rename(columns={'Sex':'Gender'}, inplace=True)

# Rename race description column
df.rename(columns={'RaceDesc':'Ethnicity'}, inplace=True)

# Remove trailing spaces in department values
df['Department'] = df['Department'].str.strip()

# Remove trailing space in gender values
df['Gender'] = df['Gender'].str.strip()

# Create list of departments for checklist
dept_list = df['Department'].unique().tolist()

# Rearrange column order
df = df[['Employee_Name', 'EmpID', 'MarriedID', 'MaritalStatusID', 'GenderID', 'EmpStatusID', 'DeptID', 'PerfScoreID', 'FromDiversityJobFairID', 'Salary', 'Termd', 'PositionID', 'Position', 'PositionLevel', 'State', 'Zip', 'DOB', 'Age', 'Generation', 'Gender', 'MaritalDesc', 'CitizenDesc', 'HispanicLatino', 'Ethnicity', 'DateofHire', 'DateofTermination', 'TermReason', 'EmploymentStatus', 'Department', 'ManagerName', 'ManagerID', 'RecruitmentSource', 'PerformanceScore', 'EngagementSurvey', 'EmpSatisfaction', 'SpecialProjectsCount', 'LastPerformanceReview_Date', 'DaysLateLast30', 'Absences']]

######################################## DATAFRAME PREPROCESSING ########################################

# Create dataframe for active count - this is the master dataframe
active_ees = df[df['EmploymentStatus']=='Active']

######################################## HELPER FUNCTIONS ########################################

### All functions used in callbacks

# Enterprise Wide Helper Functions

# Function to create pie charts for enterprise wide overview - percentage values (graph1)
def plot_pct_category_enterprise_wide(category):
    vcount = active_ees[category].value_counts()
    df = pd.DataFrame(vcount).reset_index().rename(columns={'index':category, category:'Count'})
    fig = px.pie(df, values='Count', names=category, template='seaborn')
    fig.update_layout(title='Representation by {}, Percentage'.format(category), title_x=0.5)
    return fig

# Function to create bar charts for enteprise wide overview - count values (graph2)
def plot_cnt_category_enterprise_wide(category):
    vcount = active_ees[category].value_counts()
    df = pd.DataFrame(vcount).reset_index().rename(columns={'index':category, category:'Count'})
    fig = px.bar(df, x=category, y='Count', text='Count', template='seaborn')
    fig.update_layout(title='Representation by {}, Headcount'.format(category), title_x=0.5)
    fig.update_traces(textposition='outside', cliponaxis=False)
    return fig

# Function to create graphs for category representation at a specified position level - percentage values (graph1)
def plot_pct_category_poslevel_enterprise_wide(category, poslevel):
    vcount = active_ees[category][active_ees['PositionLevel']==poslevel].value_counts()
    df = pd.DataFrame(vcount).reset_index().rename(columns={'index':category, category:'Count'})
    fig = px.pie(df, names=category, values='Count', template='seaborn')
    fig.update_layout(title='Representation by {} for {}, Percentage'.format(category, poslevel), title_x=0.5)
    return fig

# Function to create graphs for category representation at a specified position level - count values (graph2)
def plot_cnt_category_poslevel_enterprise_wide(category, poslevel):
    vcount = active_ees[category][active_ees['PositionLevel']==poslevel].value_counts()
    df = pd.DataFrame(vcount).reset_index().rename(columns={'index':category, category:'Count'})
    fig = px.bar(df, x=category, y='Count', text='Count', template='seaborn')
    fig.update_layout(title='Representation by {} for {}, Headcount'.format(category, poslevel), title_x=0.5)
    fig.update_traces(textposition='outside', cliponaxis=False)
    return fig

# Function to create graphs for salary range based on user inputs - percentage values (graph1)
def plot_pct_salary_range_enterprise_wide(category, salary_input1, salary_input2):
    df = active_ees[[category,'Salary']][(active_ees['Salary'] >= salary_input1) & (active_ees['Salary'] <= salary_input2)].reset_index(drop=True)
    df_vcount = df[category].value_counts().reset_index().rename(columns={'index':category, category:'Count'})
    fig = px.pie(df_vcount, values='Count', names=category, template='seaborn')
    fig.update_layout(title='Salary Range and {}, Percentage: {} - {}'.format(category, salary_input1, salary_input2))
    return fig

# Function to create graphs for salary range based on user inputs - count values (graph2)
def plot_cnt_salary_range_enterprise_wide(category, salary_input1, salary_input2):
    df = active_ees[[category,'Salary']][(active_ees['Salary'] >= salary_input1) & (active_ees['Salary'] <= salary_input2)].reset_index(drop=True)
    df_vcount = df[category].value_counts().reset_index().rename(columns={'index':category, category:'Count'})
    fig = px.bar(df_vcount, y='Count', x=category, text='Count', template='seaborn')
    fig.update_layout(title='Salary Range and {}, Headcount: {} - {}'.format(category, salary_input1, salary_input2))
    fig.update_traces(textposition='outside', cliponaxis=False)
    return fig

# By Department Helper Functions

# Create function to build horizontal stacked 100% bar graph for category overview by department - percentage values (graph3)
def plot_pct_category_dept(category, dept_selected):
    df = active_ees.groupby(['Department', category]).size().reset_index().rename(columns={0:'Count'})
    df['Percentage'] = active_ees.groupby(['Department', category]).size().groupby(level=0).apply(lambda x: (x / x.sum()) * 100).values.round(2)
    fig = px.bar(df[df['Department'].isin(dept_selected)], x='Percentage', y='Department', color=category, text='Percentage', barmode='stack', template='seaborn')
    fig.update_layout(title='{} by Department, Percentage'.format(category), title_x=0.5, uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_yaxes(categoryorder='category descending')
    fig.update_traces(textposition='inside', insidetextanchor='middle', texttemplate='%{text:.2f}%')
    return fig

# Create function to build grouped vertical bar graph for category overview by department - count values (graph4)
def plot_cnt_category_dept(category, dept_selected):
    df = active_ees.groupby(['Department', category]).size().reset_index().rename(columns={0:'Count'})
    df['Percentage'] = active_ees.groupby(['Department', category]).size().groupby(level=0).apply(lambda x: (x / x.sum()) * 100).values.round(2)
    fig = px.bar(df[df['Department'].isin(dept_selected)], x='Department', y='Count', color=category, text='Count', barmode='group', template='seaborn')
    fig.update_layout(title='{} by Department, Headcount'.format(category), title_x=0.5)
    fig.update_xaxes(categoryorder='category ascending')
    fig.update_traces(textposition='outside', cliponaxis=False)
    return fig

# Create function to build horizontal stacked 100% bar graph for category representation at specified position level and by department - percentage values (graph3)
def plot_pct_category_poslevel_dept(category, poslevel, dept_selected):
    df = active_ees[active_ees['PositionLevel']==poslevel].groupby(['Department',category]).size().reset_index().rename(columns={0:'Count'})
    df['Percentage'] = active_ees[active_ees['PositionLevel']==poslevel].groupby(['Department',category]).size().groupby(level=0).apply(lambda x: (x / x.sum()) * 100).values.round(2)
    fig = px.bar(df[df['Department'].isin(dept_selected)], x='Percentage', y='Department', color=category, barmode='stack', text='Percentage', template='seaborn')
    fig.update_layout(title='{} by Department for {}, Percentage'.format(category, poslevel), title_x=0.5, uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_yaxes(categoryorder='category descending')
    fig.update_traces(textposition='inside', insidetextanchor='middle', texttemplate='%{text:.2f}%')
    return fig

# Create function to build grouped vertical bar graph for category reprsentation at a specified position level and by department - count values (graph4)
def plot_cnt_category_poslevel_dept(category, poslevel, dept_selected):
    df = active_ees[active_ees['PositionLevel']==poslevel].groupby(['Department',category]).size().reset_index().rename(columns={0:'Count'})
    df['Percentage'] = active_ees[active_ees['PositionLevel']==poslevel].groupby(['Department',category]).size().groupby(level=0).apply(lambda x: (x / x.sum()) * 100).values.round(2)
    fig = px.bar(df[df['Department'].isin(dept_selected)], x='Department', y='Count', color=category, barmode='group', text='Count', template='seaborn')
    fig.update_layout(title='{} by Department for {}, Headcount'.format(category, poslevel), title_x=0.5)
    fig.update_xaxes(categoryorder='category ascending')
    fig.update_traces(textposition='outside', cliponaxis=False)
    return fig

# Function to create graphs for salary range and department based on user inputs - percentage values (graph3)
def plot_pct_salary_range_dept(category, salary_input1, salary_input2, dept_selected):
    df = active_ees[(active_ees['Salary'] >= salary_input1) & (active_ees['Salary'] <= salary_input2)].groupby(['Department', category]).size().reset_index().rename(columns={0:'Count'})
    df['Percentage'] = active_ees[(active_ees['Salary'] >= salary_input1) & (active_ees['Salary'] <= salary_input2)].groupby(['Department', category]).size().groupby(level=0).apply(lambda x: (x / x.sum()) * 100).values.round(2)
    fig = px.bar(df[df['Department'].isin(dept_selected)], x='Percentage', y='Department', color=category, text='Percentage', barmode='stack', template='seaborn')
    fig.update_layout(title='Salary Range and {} by Department, Percentage: {} - {}'.format(category, salary_input1, salary_input2), title_x=0.5, uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_yaxes(categoryorder='category descending')
    fig.update_traces(textposition='inside', insidetextanchor='middle', texttemplate='%{text:.2f}%')
    return fig

# Function to create graphs for salary range and department based on user inputs - count values (graph4)
def plot_cnt_salary_range_dept(category, salary_input1, salary_input2, dept_selected):
    df = active_ees[(active_ees['Salary'] >= salary_input1) & (active_ees['Salary'] <= salary_input2)].groupby(['Department', category]).size().reset_index().rename(columns={0:'Count'})
    df['Percentage'] = active_ees[(active_ees['Salary'] >= salary_input1) & (active_ees['Salary'] <= salary_input2)].groupby(['Department', category]).size().groupby(level=0).apply(lambda x: (x / x.sum()) * 100).values.round(2)
    fig = px.bar(df[df['Department'].isin(dept_selected)], x='Department', y='Count', color=category, text='Count', barmode='group', template='seaborn')
    fig.update_layout(title='Salary Range and {} by Department, Headcount: {} - {}'.format(category, salary_input1, salary_input2), title_x=0.5)
    fig.update_xaxes(categoryorder='category ascending')
    fig.update_traces(textposition='outside', cliponaxis=False)
    return fig

######################################## DASHBOARD ########################################

######################################## INITALIZATION AND LAYOUT ########################################

# Define tab style formatting
tab_styles = {
    'display': 'inlineBlock',
    'width': '500px',
    'height': '40px',
}

tab_style = {
    'fontSize': 13,
    'padding': '0px',
    'line-height': '40px',
}

active_tab_style = {
    'fontWeight': 'bold',
    'fontSize': 15,
    'padding': '0px',
    'line-height': '40px',
    'border-top': '3px solid #7c7c9c',
    #'background-color': '#8e9599',
    #'color':'white',
}

# Custom CSS
external_stylesheets = ['https://changchunhui.github.io/ref/stylesheet.css']

# Instantiate dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

# Set option for all departments selection
options=[{'label':d, 'value':d} for d in sorted(dept_list)]
all_depts = [option['value'] for option in options]

# Layout
app.layout = html.Div([
    # Header
    html.H4('HCM Diversity Dashboard'),

    # Link object for Github page
    html.Div([
        html.A('See the project documentation my Github repo', href='https://github.com/changchunhui/hcmdash', target='blank'),
        html.Br(),
        html.Br()
    ]),
    
    html.Div([
        dcc.Markdown('''_**Graph options:** ** Options and filters do not affect the Employee Data tab_''')
    ]),
    
    html.Div([
        dcc.Markdown('''_Select category:_''')
    ], style={'display':'inline-block', 'margin-left':'20px'}),

    # Radio buttons for category
    html.Div([
        dcc.RadioItems(
            id='category',
            options=[
                {'label':'Gender', 'value':'Gender'}, 
                {'label':'Ethnicity', 'value':'Ethnicity'}, 
                {'label':'Generation', 'value':'Generation'}, 
            ],
            value='Gender',
            style={'display':'inline-block'}, 
            labelStyle={'display':'inline-block'}, 
            inputStyle={'margin-left':'30px'}
        )
    ], style={'display':'inline-block'}),

    html.Br(),

    html.Div([
        dcc.Markdown('''_Select level of detail:_''')
    ], style={'display':'inline-block', 'margin-left':'20px'}),
    
    # Radio buttons for level of detail
    html.Div([
        dcc.RadioItems(
            id='detail_level', 
            options=[
                {'label':'Overview', 'value':'Overview'},
                {'label':'Position Level', 'value':'Position Level'},
                {'label':'Salary Range', 'value':'Salary Range'}
            ],
            value='Overview', 
            style={'display':'inline-block'}, 
            labelStyle={'display':'inline-block'}, 
            inputStyle={'margin-left':'30px'}, 
        )
    ], style={'display':'inline-block'}),
    
    html.Br(),
    
    # Markdown text
    html.Div([
        dcc.Markdown('''_Select position level:_''')
    ], id='select_poslevel_text', style={'display':'inline-block', 'margin-left':'20px'}),
    
    # Radio butons for position level
    html.Div([
        dcc.RadioItems(
            id='poslevel',
            options=[
                {'label':'Senior Management', 'value':'Senior Management'},
                {'label':'Middle Management', 'value':'Middle Management'}, 
                {'label':'Individual Contributor', 'value':'Individual Contributor'}
            ],
            value='Senior Management',
            style={'display':'inline-block'},
            labelStyle={'display':'inline-block'},
            inputStyle={'margin-left':'30px'},
        ),
    ], style={'display':'inline-block'}),

    # Empty div to allow next elements to display on next line - to display or hide based on condition in callback
    html.Div([
    ], id='row_space', style={'display':'block'}),

    html.Div([
        dcc.Markdown('''_Input salary range:_''')
    ], id='input_salary_range_text', style={'display':'inline-block', 'margin-left':'20px'}),

    # Input field - lower boundary of salary range
    html.Div([
        dcc.Input(
            id='salary_input1', 
            type='number', 
            placeholder='Enter lower boundary', 
        )
    ], style={'display':'inline-block'}),

    # Input field - upper boundary of salary range
    html.Div([
        dcc.Input(
            id='salary_input2', 
            type='number', 
            placeholder='Enter upper boundary'
        )
    ], style={'display':'inline-block'}),

    # Markdown note for max salary range
    html.Div([
        dcc.Markdown('''_** Max range: $45,000 - $250,000_''')
    ], id='max_range_text', style={'display':'inline-block', 'margin-left':'20px'}),

    # Empty div to allow next elements to display on next line - to display or hide based on condition in callback
    html.Div([
    ], id='row_space2', style={'display':'block'}),

    html.Br(),

    # Tabs
    html.Div([
        dcc.Tabs(
            id='tabs',
            value='enterprise_wide',
            parent_className='custom_tabs',
            className='custom_tabs_container',
            children=[
                dcc.Tab(
                    label='Enterprise Wide',
                    value='enterprise_wide',
                    className='custom_tab',
                    selected_className='custom_tab--selected',
                    style=tab_style,
                    selected_style=active_tab_style
                ),
                dcc.Tab(
                    label='By Department',
                    value='by_department',
                    className='custom_tab',
                    selected_className='custom_tab--selected',
                    style=tab_style,
                    selected_style=active_tab_style
                ),
                dcc.Tab(
                    label='Employee Data',
                    value='employee_data',
                    className='custom_tab',
                    selected_className='custom_tab--selected',
                    style=tab_style,
                    selected_style=active_tab_style
                ),
            ], style=tab_styles
        ), html.Div(id='page') 
    ]),
    
])

######################################## CALLBACKS ########################################

# Callback - changing conditional visibility of position level radio buttons
@app.callback(
    [Output(component_id='select_poslevel_text', component_property='style'), 
     Output(component_id='poslevel', component_property='style'), 
     Output(component_id='row_space', component_property='style')], 
    Input(component_id='detail_level', component_property='value')
)
def change_component_visibility(detail_level):
    if detail_level == 'Position Level':
        return [{'display':'inline-block', 'margin-left':'20px'}, {'display':'inline-block'}, {'display':'block'}]
    else:
        return [{'display':'none'}, {'display':'none'}, {'display':'none'}]

# Callback - changing conditional visibility of salary range input fields
@app.callback(
    [Output(component_id='input_salary_range_text', component_property='style'), 
     Output(component_id='salary_input1', component_property='style'), 
     Output(component_id='salary_input2', component_property='style'), 
     Output(component_id='max_range_text', component_property='style'), 
     Output(component_id='row_space2', component_property='style')], 
    Input(component_id='detail_level', component_property='value')
)
def change_component_visibility(detail_level):
    if detail_level == 'Salary Range':
        return [{'display':'inline-block', 'margin-left':'20px'}, {'display':'inline-block', 'margin-left':'30px'}, {'display':'inline-block', 'margin-left':'30px'}, {'display':'inline-block', 'margin-left':'20px'}, {'display':'block'}]
    else:
        return [{'display':'none'}, {'display':'none'}, {'display':'none'}, {'display':'none'}, {'display':'none'}]

# Callback - changing conditional visibility of departments checklists
@app.callback(
    [Output(component_id='select_dept_text', component_property='style'), 
     Output(component_id='checklist_all_dept', component_property='style'), 
     Output(component_id='checklist_dept', component_property='style'), 
     Output(component_id='row_space3', component_property='style')], 
    Input(component_id='tabs', component_property='value')
)
def change_component_visibility(tab):
    if tab == 'enterprise_wide':
        return [{'display':'none'}, {'display':'none'}, {'display':'none'}, {'display':'none'}]
    elif tab == 'by_department':
        return [{'display':'inline-block'}, {'display':'inline-block'}, {'display':'inline-block'}, {'display':'block'}]
    elif tab == 'employee_data':
        return [{'display':'none'}, {'display':'none'}, {'display':'none'}, {'display':'none'}]

# Callback for "all" departments checklist (first checklist)
# Controls selection on the main departments checklist (second checklist)
@app.callback(
    Output(component_id='checklist_dept', component_property='value'),
    Output(component_id='checklist_all_dept', component_property='value'),
    Input(component_id='checklist_dept', component_property='value'),
    Input(component_id='checklist_all_dept', component_property='value'),
)
def sync_checklists(dept_selected, all_selected):
    ctx = dash.callback_context
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if input_id == 'checklist_dept':
        all_selected = ['All'] if set(dept_selected) == set(all_depts) else []
    else:
        dept_selected = all_depts if all_selected else []
    return dept_selected, all_selected

# Callback for tabs
@app.callback(
    Output(component_id='page', component_property='children'),
    Input(component_id='tabs', component_property='value')
)
def render_page(tab):
    if tab == 'enterprise_wide':
        return html.Div([
            html.Br(),
            
            # Instantiate graphs 1 and 2
            html.Div([
                dcc.Graph(id='graph1'), 
                dcc.Graph(id='graph2'), 
            ], style={'columnCount':2}),
        ])

    elif tab == 'by_department':
        return html.Div([
            html.Br(),

            # Markdown text
            html.Div([
                dcc.Markdown('''_Select department(s):_'''),
            ], id='select_dept_text', style={'display':'inline-block'}),

            # Checklist - all departments
            html.Div([
                dcc.Checklist(
                    id='checklist_all_dept',
                    options=[{'label': 'All', 'value': 'All'}],
                    value=['All'],
                    labelStyle={'display': 'inline-block'},
                    inputStyle={'margin-left':'30px'},
                ),
            ], style={'display':'inline-block'}),
            
            # Checklist - individual departments
            html.Div([
                dcc.Checklist(
                    id='checklist_dept',
                    options=options,
                    value=[],
                    labelStyle={'display': 'inline-block'},
                    inputStyle={'margin-left':'30px'},
                ),        
            ], style={'display':'inline-block'}),

            # Empty div to allow next elements to display on next line - to display or hide based on condition in callback
            html.Div([
                html.Br()
            ], id='row_space3', style={'display':'block'}),

            # Instantiate graphs 3 and 4
            html.Div([
                dcc.Graph(id='graph3'), 
                dcc.Graph(id='graph4'), 
            ], style={'columnCount':1}),

        ])
    
    elif tab == 'employee_data':
        return html.Div([
            dash_table.DataTable(
                id='table',
                columns=[{'id':c, 'name':c} for c in active_ees.columns],
                data=active_ees.to_dict('records'),
                page_size=20,
                style_table={'height':'500px', 'overflowY':'auto'}
            )
        ])

# Callback - graph1 - enterprise wide category representation, percertange values
@app.callback(
    Output(component_id='graph1', component_property='figure'), 
    [Input(component_id='category', component_property='value'), 
     Input(component_id='detail_level', component_property='value'), 
     Input(component_id='poslevel', component_property='value'), 
     Input(component_id='salary_input1', component_property='value'), 
     Input(component_id='salary_input2', component_property='value')]
)
def update_figure(category, detail_level, poslevel, salary_input1, salary_input2):
    if detail_level == 'Overview':
        fig = plot_pct_category_enterprise_wide(category)
        return fig

    elif detail_level == 'Position Level':
        fig = plot_pct_category_poslevel_enterprise_wide(category, poslevel)
        return fig
    
    elif detail_level == 'Salary Range':
        fig = plot_pct_salary_range_enterprise_wide(category, salary_input1, salary_input2)
        return fig

# Callback - graph2 - enterprise wide category representation, count values
@app.callback(
    Output(component_id='graph2', component_property='figure'), 
    [Input(component_id='category', component_property='value'), 
     Input(component_id='detail_level', component_property='value'), 
     Input(component_id='poslevel', component_property='value'), 
     Input(component_id='salary_input1', component_property='value'), 
     Input(component_id='salary_input2', component_property='value')]
)
def update_figure(category, detail_level, poslevel, salary_input1, salary_input2):
    if detail_level == 'Overview':
        fig = plot_cnt_category_enterprise_wide(category)
        return fig
    
    elif detail_level == 'Position Level':
        fig = plot_cnt_category_poslevel_enterprise_wide(category, poslevel)
        return fig

    elif detail_level == 'Salary Range':
        fig = plot_cnt_salary_range_enterprise_wide(category, salary_input1, salary_input2)
        return fig

# Callback - graph3 - departmental category representation, percentage values
@app.callback(
    Output(component_id='graph3', component_property='figure'), 
    [Input(component_id='category', component_property='value'), 
     Input(component_id='detail_level', component_property='value'), 
     Input(component_id='poslevel', component_property='value'), 
     Input(component_id='salary_input1', component_property='value'), 
     Input(component_id='salary_input2', component_property='value'), 
     Input(component_id='checklist_dept', component_property='value')]
)
def update_figure(category, detail_level, poslevel, salary_input1, salary_input2, dept_selected):
    if detail_level == 'Overview':
        fig = plot_pct_category_dept(category, dept_selected)
        return fig

    elif detail_level == 'Position Level':
        fig = plot_pct_category_poslevel_dept(category, poslevel, dept_selected)
        return fig

    elif detail_level == 'Salary Range':
        fig = plot_pct_salary_range_dept(category, salary_input1, salary_input2, dept_selected)
        return fig

# Callback - graph4 - departmental category representation, count values
@app.callback(
    Output(component_id='graph4', component_property='figure'), 
    [Input(component_id='category', component_property='value'), 
     Input(component_id='detail_level', component_property='value'), 
     Input(component_id='poslevel', component_property='value'), 
     Input(component_id='salary_input1', component_property='value'), 
     Input(component_id='salary_input2', component_property='value'), 
     Input(component_id='checklist_dept', component_property='value')]
)
def update_figure(category, detail_level, poslevel, salary_input1, salary_input2, dept_selected):
    if detail_level == 'Overview':
        fig = plot_cnt_category_dept(category, dept_selected)
        return fig

    elif detail_level == 'Position Level':
        fig = plot_cnt_category_poslevel_dept(category, poslevel, dept_selected)
        return fig

    elif detail_level == 'Salary Range':
        fig = plot_cnt_salary_range_dept(category, salary_input1, salary_input2, dept_selected)
        return fig

# Run app
# if __name__ == '__main__':
#     app.run_server(debug=True)