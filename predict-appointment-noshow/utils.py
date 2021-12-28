import pandas as pd
from bokeh.models import HoverTool
from bokeh.models.sources import ColumnDataSource
from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from bokeh.layouts import gridplot
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve


def load_data(path):
    """ After loading the data with pandas, fix some
        column names and make appointment_day the end
        of the day. Then, print some information about
        the dataframe.
    """
    # Data Wrangling
    # After loading the data with pandas, Fix typos in some column names
    # but we change others as well to suit personal preference.
    data = pd.read_csv(path, parse_dates=['AppointmentDay', 'ScheduledDay'])
    data.index = data['AppointmentID']
    data.rename(columns={'Hipertension': 'hypertension',
                         'Handcap': 'handicap',
                         'PatientId': 'patient_id',
                         'AppointmentID': 'appointment_id',
                         'ScheduledDay': 'scheduled_time',
                         'AppointmentDay': 'appointment_day',
                         'Neighbourhood': 'neighborhood',
                         'No-show': 'no_show'}, inplace=True)
    for column in data.columns:
        data.rename(columns={column: column.lower()}, inplace=True)
    data['appointment_day'] = data['appointment_day'] + pd.Timedelta('1d') - pd.Timedelta('1s')


    data["neighborhood"] = data["neighborhood"].astype('category')
    data["patient_id"] = data["patient_id"].astype('int')

    for bool_col in ["scholarship", "hypertension", "diabetes", "alcoholism", "handicap", "sms_received"]:
        data[bool_col] = data[bool_col].astype(bool)

    data['no_show'] = data['no_show'].map({'No': False, 'Yes': True})

    # Show the size of the data in a print statement
    print('{} Appointments, {} Columns'.format(data.shape[0], data.shape[1]))
    print('Appointments: {}'.format(data.shape[0]))
    print('Schedule times: {}'.format(data.scheduled_time.nunique()))
    print('Patients: {}'.format(data.patient_id.nunique()))
    print('Neighborhoods: {}'.format(data.neighborhood.nunique()))
    pd.options.display.max_columns = 100
    pd.options.display.float_format = '{:.2f}'.format
    return data


def plot_roc_auc(y_test, probs, pos_label=1):
    fpr, tpr, thresholds = roc_curve(y_test,
                                     probs[:, 1],
                                     pos_label=pos_label)

    output_notebook()
    p = figure(height=400, width=400)
    p.line(x=fpr, y=tpr)
    p.title.text = 'Receiver operating characteristic'
    p.xaxis.axis_label = 'False Positive Rate'
    p.yaxis.axis_label = 'True Positive Rate'

    p.line(x=fpr, y=fpr, color='red', line_dash='dashed')
    return(p)


def plot_f1(y_test, probs, nprecs):
    threshes = [x/1000. for x in range(50, nprecs)]
    precisions = [precision_score(y_test, probs[:, 1] > t) for t in threshes]
    recalls = [recall_score(y_test, probs[:, 1] > t) for t in threshes]
    fones = [f1_score(y_test, probs[:, 1] > t) for t in threshes]

    output_notebook()
    p = figure(height=400, width=400)
    p.line(x=threshes, y=precisions, color='green', legend_label='precision')
    p.line(x=threshes, y=recalls, color='blue', legend_label='recall')
    p.line(x=threshes, y=fones, color='red', legend_label='f1')
    p.xaxis.axis_label = 'Threshold'
    p.title.text = 'Precision, Recall, and F1 by Threshold'
    return(p)


def plot_kfirst(ytest, probs, firstk=500):
    A = pd.DataFrame(probs)
    A['y_test'] = ytest.values
    krange = range(firstk)
    firstk = []
    for K in krange:
        a = A[1][:K]
        a = [1 for prob in a]
        b = A['y_test'][:K]
        firstk.append(precision_score(b, a))

    output_notebook()
    p = figure(height=400, width=400)
    p.step(x=krange, y=firstk)
    p.xaxis.axis_label = 'Predictions sorted by most likely'
    p.yaxis.axis_label = 'Precision'
    p.title.text = 'K-first'
    p.yaxis[0].formatter.use_scientific = False
    return p


def plot_locations(fm):
    tmp = fm.groupby('neighborhood').apply(lambda df: df.tail(1))['locations.COUNT(appointments)'].sort_values().reset_index().reset_index()
    hover = HoverTool(tooltips=[
        ("Count", "@{locations.COUNT(appointments)}"),
        ("Place", "@neighborhood"),
    ])
    source = ColumnDataSource(tmp)
    p4 = figure(width=400,
                height=400,
                tools=[hover, 'box_zoom', 'reset', 'save'])
    p4.scatter('index', 'locations.COUNT(appointments)', alpha=.7, source=source, color='teal')
    p4.title.text = 'Appointments by Neighborhood'
    p4.xaxis.axis_label = 'Neighborhoods (hover to view)'
    p4.yaxis.axis_label = 'Count'
    return p4


def plot_noshow_by_loc(fm):
    tmp = fm.groupby('neighborhood').apply(lambda df: df.tail(1))[['locations.COUNT(appointments)',
                                                                   'locations.PERCENT_TRUE(appointments.no_show)']].sort_values(
        by='locations.COUNT(appointments)').reset_index().reset_index()
    hover = HoverTool(tooltips=[
        ("Prob", "@{locations.PERCENT_TRUE(appointments.no_show)}"),
        ("Place", "@neighborhood"),
    ])
    source = ColumnDataSource(tmp)
    p5 = figure(width=400,
                height=400,
                tools=[hover, 'box_zoom', 'reset', 'save'])
    p5.scatter('index', 'locations.PERCENT_TRUE(appointments.no_show)', alpha=.7, source=source, color='maroon')
    p5.title.text = 'Probability of no-show by Neighborhood'
    p5.xaxis.axis_label = 'Neighborhoods (hover to view)'
    p5.yaxis.axis_label = 'Probability of no-show'
    return p5


def plot_ages(fm):
    tmp = fm.tail(5000).groupby('age').apply(lambda df: df.tail(1))[['ages.COUNT(appointments)']].sort_values(
        by='ages.COUNT(appointments)').reset_index().reset_index()
    hover = HoverTool(tooltips=[
        ("Count", "@{ages.COUNT(appointments)}"),
        ("Age", "@age"),
    ])
    source = ColumnDataSource(tmp)
    p6 = figure(width=400,
                height=400,
                tools=[hover, 'box_zoom', 'reset', 'save'])
    p6.scatter('age', 'ages.COUNT(appointments)', alpha=.7, source=source, color='magenta')
    p6.title.text = 'Appointments by Age'
    p6.xaxis.axis_label = 'Age'
    p6.yaxis.axis_label = 'Count'
    return p6


def plot_noshow_by_age(X):
    source = ColumnDataSource(X.tail(5000).groupby('age').apply(lambda x: x.tail(1)))

    hover = HoverTool(tooltips=[
        ("Prob", "@{ages.PERCENT_TRUE(appointments.no_show)}"),
        ("Age", "@age"),
    ])

    p7 = figure(title="Probability no-show by Age",
                x_axis_label='Age',
                y_axis_label='Probability of no-show',
                width=400,
                height=400,
                tools=[hover, 'box_zoom', 'reset', 'save']
                )

    p7.scatter('age', 'ages.PERCENT_TRUE(appointments.no_show)',
               alpha=.7,
               source=source)
    return p7
