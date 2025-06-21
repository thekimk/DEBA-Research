### KK ###
from import_KK import *
from data_KK import *
from preprocessing_KK import *
from description_KK import *
from evaluation_KK import *



### Date and Author: 20250611, Kyungwon Kim ###
### SHAP explanation
def explanation_SHAP_values(model, X_train, X_test, X_colname,
                            model_type='Linear'): # 'Linear', 'Tree', 'Deep'
    # Format Transformation
    if type(X_train) == pd.DataFrame:
        X_train = X_train.values
    if type(X_test) == pd.DataFrame:
        X_test = X_test.values

    # SHAP model
    if model_type.lower() == 'tree':
        explainer = shap.TreeExplainer(model, data=X_train, feature_names=X_colname, approximate=True)
        shap_values_train = explainer(X_train, check_additivity=False)
        shap_values_test = explainer(X_test, check_additivity=False)
        ## Sampling for Interaction Calculation

        ## Feature Interaction Calculation
        explainer_inter = shap.TreeExplainer(model, data=None, feature_names=X_colname, approximate=False)
        shap_intervalues_train = explainer_inter.shap_interaction_values(X_train)
        shap_intervalues_test = explainer_inter.shap_interaction_values(X_test)
    else:
        explainer = shap.Explainer(model, X_train, 
                                   algorithm=model_type.lower(), feature_names=X_colname)
        shap_values_train = explainer(X_train)
        shap_values_test = explainer(X_test)
        shap_intervalues_train = None
        shap_intervalues_test = None

    return shap_values_train, shap_values_test, shap_intervalues_train, shap_intervalues_test


### Date and Author: 20250611, Kyungwon Kim ###
### SHAP individual explanation
def explanation_SHAP_individual(shap_values, 
                                output_type='identity',    # 'identity', 'logit'
                                X_colname=None,
                                max_display=10):
    print('Individual Explanation(1 Decision Plot -> 1 Force Plot -> 1000 Force Plot)...')
    # Dimension Rearrange
    if shap_values.values.ndim == 3:
        shap_values = shap_values[:,:,-1]
        
    # Visualization
    random_index = np.random.randint(shap_values.values.shape[0])
    shap.decision_plot(base_value=shap_values.base_values[random_index],
                       shap_values=shap_values.values[random_index],
                       features=shap_values.data[random_index],
                       feature_names=X_colname,
                       feature_display_range=slice(None, -max_display, -1),
                       link=output_type, highlight=0, show=False)
    fig = plt.gcf()
    ax = plt.gca()
    for label in ax.get_yticklabels():
        label.set_fontsize(20)          # Y축 폰트 크기
    for label in ax.get_xticklabels():
        label.set_fontsize(20)          # X축 폰트 크기
    fig.tight_layout(pad=-10) 
    plt.show()
    shap.initjs()
    display(shap.force_plot(base_value=shap_values.base_values[random_index],
                            shap_values=shap_values.values[random_index],
                            features=shap_values.data[random_index],
                            feature_names=X_colname,
                            link=output_type))
    shap.initjs()
    shap_sample = shap_values.sample(1000)
    display(shap.force_plot(base_value=shap_sample.base_values,
                            shap_values=shap_sample.values,
                            features=shap_sample.data,
                            feature_names=X_colname,
                            link=output_type))


### Date and Author: 20250611, Kyungwon Kim ###
### SHAP total explanation
def explanation_SHAP_total(shap_values, X_colname=None, max_display=10):
    print('Total Explanation(Beeswarm Plot)...')
    # Dimension Rearrange
    if shap_values.values.ndim == 3:
        shap_values = shap_values[:,:,-1]
        
    # Beeswarm Plot
    shap.plots.beeswarm(shap_values=shap_values, max_display=max_display, show=False) # order=shap_values_train.abs.max(0)
    fig = plt.gcf()
    ax = plt.gca()
    for label in ax.get_yticklabels():
        label.set_fontsize(20)          # Y축 폰트 크기
    for label in ax.get_xticklabels():
        label.set_fontsize(16)          # X축 폰트 크기
    fig.tight_layout(pad=-10) 
    plt.subplots_adjust(top=1.5, bottom=0.8)    # Y축 레이블 간격
    plt.show()

def explanation_SHAP_change(shap_values, X, n_bins=10, X_colname=[]):
    print('Feature Contribution Direction...')
    slope_quantile, slope_total = [], []
    for i in tqdm(range(shap_values.shape[1])):
        # Y & X Setting
        Y = shap_values[:,i].copy()
        X_sub = X[:,i].copy()
        
        # Binning
        ## 모든 값이 같으면 더미 bin 반환
        if np.all(X_sub == X_sub[0]):
            bins = np.array([X_sub[0], X_sub[0]+1e-9])
        ## 값의 고유값의 수가 n_bins+1 미만이면 bin 갯수 축소
        X_unique = np.unique(X_sub)
        if len(X_unique) <= n_bins:
            n_bins_resize = len(X_unique)-1
            bins = np.quantile(X_sub, q=np.linspace(0, 1, n_bins_resize+1))
        else:
            bins = np.quantile(X_sub, q=np.linspace(0, 1, n_bins+1))
        bins = np.unique(bins)
        if len(bins) == 1:
            bins = np.array([bins[0], bins[0]+1e-9])
        bins[-1] = bins[-1] + 1e-9
    
        # Binning Statistics
        row_binning = pd.cut(X_sub, bins, include_lowest=True, duplicates='drop')
        df = pd.DataFrame({
            'bin': row_binning,
            'X': X_sub,
            'SHAP': Y
        })
        df_stat = df.groupby('bin', observed=True).agg(
            X_mean=('X', 'mean'),
            SHAP_mean=('SHAP', 'mean'),
        ).reset_index()
        df_stat = df_stat.dropna(subset=["X_mean", "SHAP_mean"])
    
        # Slope Calulation
        ## Total Regression
        model = LinearRegression(fit_intercept=True).fit(X_sub.reshape(-1, 1), Y)
        slope_total.append(model.coef_[0])
        ## Quantile Regression
        slope = []
        for idx, row in df_stat.iterrows():
            mask = row_binning == row['bin']
            model = None
            model = LinearRegression(fit_intercept=True).fit(X_sub[mask].reshape(-1,1), Y[mask])
            slope.append(model.coef_[0])
        slope_quantile.append(np.median(slope))
    contribution_direction = pd.concat([pd.Series(slope_total), pd.Series(slope_quantile)], axis=1)
    contribution_direction.columns = ['Contribution Change (Linear Regression)', 'Contbituion Change (Quantile Regression)']
    
    return contribution_direction

def explanation_SHAP_contribution(shap_values, X, n_bins=10, X_colname=[]):
    print('Feature Total Contribution...')
    # Dimension Rearrange
    if shap_values.values.ndim == 3:
        shap_values = shap_values[:,:,-1]
    shap_values = shap_values.values
    
    # Positive and Negative Average Importance
    importance = np.median(shap_values, axis=0)
    importance_posi = np.sum(np.where(shap_values>0, shap_values, 0), axis=0) / np.sum(shap_values>0, axis=0)
    importance_nega = np.sum(np.where(shap_values<0, shap_values, 0), axis=0) / np.sum(shap_values<0, axis=0)
    importance_range = importance_posi + np.abs(importance_nega)
    ## Rearrange
    df_contribution = pd.DataFrame({
        'Feature': X_colname,
        'Importance': importance,
        # 'Positive': importance_posi,
        # 'Negative': importance_nega,
        'Contribution Range': importance_range,
    })

    # Change Importance
    df_change = explanation_SHAP_change(shap_values, X, n_bins=n_bins, X_colname=X_colname)
    df_contribution = pd.concat([df_contribution, df_change], axis=1)
    ## Rearrange
    df_result = pd.concat([df_contribution[df_contribution['Importance'] > 0].sort_values(by='Contribution Range', ascending=False).reset_index().iloc[:,1:],
                               df_contribution[df_contribution['Importance'] < 0].sort_values(by='Contribution Range', ascending=False).reset_index().iloc[:,1:]], axis=1)
    ## Save
    folder_location = os.path.join(os.getcwd(), 'Result', '')
    if not os.path.exists(folder_location):
        os.makedirs(folder_location)
    save_name = os.path.join(folder_location,'FeatureImportance_Total.csv')
    df_result.to_csv(save_name, index=False, encoding='utf-8-sig')
    display(df_result.head())
    ## Rearrange
    df_result = pd.concat([df_contribution[df_contribution['Contribution Change (Linear Regression)'] > 0].sort_values(by='Contribution Range', ascending=False).reset_index().iloc[:,1:],
                               df_contribution[df_contribution['Contribution Change (Linear Regression)'] < 0].sort_values(by='Contribution Range', ascending=False).reset_index().iloc[:,1:]], axis=1)
    ## Save
    folder_location = os.path.join(os.getcwd(), 'Result', '')
    if not os.path.exists(folder_location):
        os.makedirs(folder_location)
    save_name = os.path.join(folder_location,'FeatureImportance_TotalDirectional.csv')
    df_result.to_csv(save_name, index=False, encoding='utf-8-sig')
    display(df_result.head())


### Date and Author: 20250611, Kyungwon Kim ###
### SHAP total explanation with interaction
def explanation_SHAP_interaction(shap_values, shap_intervalues, X_colname, max_display=10):
    print('Feature Interactive Contribution...')
    # Feature Order
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    feature_order = np.argsort(shap_importance)[::-1][:max_display]

    # Calculate Feature Interaction Importance
    feature_interimportance  = dict()
    for idx_feature in tqdm(feature_order):
        interaction = np.mean(shap_intervalues[:,idx_feature,:], axis=0)
        interaction_feature = [X_colname[i] for i in np.argsort(-np.abs(interaction)) if i != idx_feature]
        interaction_contribution = [interaction[i] for i in np.argsort(-np.abs(interaction)) if i != idx_feature]
        result_interaction = pd.DataFrame([interaction_feature, interaction_contribution]).T
        result_interaction.columns = ['Interaction Features', 'Contribution Importance']
        feature_interimportance[X_colname[idx_feature]] = result_interaction
    ## Arrange as DF
    feature_interimportance_df = []
    for key, df in feature_interimportance.items():
        df_sub = df[df['Contribution Importance'] != 0]
        df_sub.insert(0, 'Feature', key)
        feature_interimportance_df.append(df_sub)
    feature_interimportance_df = pd.concat(feature_interimportance_df, ignore_index=True)
    ## Save
    folder_location = os.path.join(os.getcwd(), 'Result', '')
    if not os.path.exists(folder_location):
        os.makedirs(folder_location)
    save_name = os.path.join(folder_location,'FeatureImportance_Interaction.csv')
    feature_interimportance_df.to_csv(save_name, index=False, encoding='utf-8-sig')   
    
    # Visualization
    feature_order = [X_colname[i] for i in feature_order]
    for col in feature_order[:int(max_display/2)]:
        shap.dependence_plot(ind=col, shap_values=shap_values.values,
                             features=shap_values.data, feature_names=X_colname)


### Date and Author: 20250611, Kyungwon Kim ###
### SHAP main
def explanation_SHAP(model, X_train, X_test, X_colname, 
                     model_type='Linear',    # 'Linear', 'Tree', 'Deep'
                     output_type='identity',    # 'identity', 'logit'
                     max_display=10, dependency=False):
    # Calculate SHAP
    shap_values_train, shap_values_test,\
    shap_intervalues_train, shap_intervalues_test = explanation_SHAP_values(model, X_train, X_test, 
                                                                            X_colname, model_type=model_type)
    # Individual
    print('Train Dataset:')
    explanation_SHAP_individual(shap_values_train, output_type, X_colname,
                                max_display=max_display)
    print('Test Dataset:')
    explanation_SHAP_individual(shap_values_test, output_type, X_colname,
                                max_display=max_display)
    
    # Total
    print('Train Dataset:')
    explanation_SHAP_total(shap_values_train, X_colname, max_display=max_display)
    print('Test Dataset:')
    explanation_SHAP_total(shap_values_test, X_colname, max_display=max_display)

    # Directional Feature Importance
    print('Train Dataset:')
    explanation_SHAP_contribution(shap_values_train, X_train, n_bins=5, X_colname=X_colname)
    print('Test Dataset:')
    explanation_SHAP_contribution(shap_values_test, X_test, n_bins=5, X_colname=X_colname)

    # Feature Interactional Importance
    if dependency:
        print('Train Dataset:')
        explanation_SHAP_interaction(shap_values_train, shap_intervalues_train, X_colname, max_display)
        print('Test Dataset:')
        explanation_SHAP_interaction(shap_values_test, shap_intervalues_train, X_colname, max_display)


    