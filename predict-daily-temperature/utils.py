import numpy as np
import matplotlib.pyplot as plt
import featuretools as ft

plt.rcdefaults()


def plot_model_performances(baseline_evalml_score, random_forest_evalml_score, baseline_score, featuretools_score):
    # data to plot
    n_groups = 2
    evalml_runs = (baseline_evalml_score, random_forest_evalml_score)
    manual_runs = (baseline_score, featuretools_score)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    plt.bar(index, evalml_runs, bar_width,
            alpha=opacity,
            color='#9300ff',
            label='EvalML Runs')

    plt.bar(index + bar_width, manual_runs, bar_width,
            alpha=opacity,
            color='gray',
            label='Manual Runs')

    plt.xlabel('Models')
    plt.ylabel('Median Absolute Error')
    plt.title('Model Performance')
    plt.xticks(index + bar_width/2,
               ("Baseline Runs", "Feature Engineering Runs"))
    plt.legend()

    plt.tight_layout()
    plt.show()


def feature_importances(X, reg, feats=5):
    feature_imps = [(imp, X.columns[i])
                    for i, imp in enumerate(reg.feature_importances_)]
    feature_imps.sort()
    feature_imps.reverse()
    for i, f in enumerate(feature_imps[0:feats]):
        print('{}: {} [{:.3f}]'.format(i + 1, f[1], f[0]))
    print('-----\n')
    return [f[1] for f in feature_imps[:feats]]


def set_up_entityset(df, id='es', time_index=None):
    es = ft.EntitySet(id=id)
    es.add_dataframe(df.copy(),
                     dataframe_name='temperatures',
                     index='id',
                     make_index=True,
                     time_index=time_index)
    return es
