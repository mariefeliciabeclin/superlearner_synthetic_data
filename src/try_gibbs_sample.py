from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model


def step(df, reg="RF"):
    n_col = len(X.columns)
    new_df = dict({})
    for i, col in df.columns:
        print(i)
        Y = df[col]
        col_ = list(df.columns)
        col_.pop(i)
        X = df[col_]
        if reg =="RF":
            r = RandomForestRegressor(n_estimators=10)

        if reg =="norm":
            r = linear_model.LinearRegression()
            
        r = r.fit(X, Y)
        new_df[col] = r.predict
    return pd.DataFrame(new_df)


df gibbs_sampler(df):
    df_step=df
    for i in range(5):
        df = step(df)
    return df

    

