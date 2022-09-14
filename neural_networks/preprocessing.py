#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 09:12:37 2022

@author: Marieke_Wesselkamp
"""
import sqlite3
import zipfile
import pandas as pd
import matplotlib.pyplot as plt


def get_gpp():
    # %% The Data
    # zf = zipfile.ZipFile('/Users/Marieke_Wesselkamp/ProfoundData.zip')
    # db = zf.read('ProfoundData.sqlite')

    con = sqlite3.connect('/Users/Marieke_Wesselkamp/ProfoundData.sqlite')

    df = pd.read_sql_query(''.join(('SELECT * FROM ', 'FLUX')), con)
    df.columns.values
    df = df[['site', 'date', 'year', 'day', 'gppDtVutRef_umolCO2m2s1', 'gppDtVutSe_umolCO2m2s1']]

    df['date'] = pd.to_datetime(df['date']).dt.normalize()

    df_grouped = df.groupby(['site', 'date', 'year']).agg(
        {'gppDtVutRef_umolCO2m2s1': ['mean'], 'gppDtVutSe_umolCO2m2s1': ['mean']})
    df_grouped.columns = ['GPP_ref', 'GPP_se']
    df_grouped = df_grouped.reset_index()
    df_grouped['date'] = pd.to_datetime(df_grouped['date']).dt.normalize()

    dfs = df_grouped.groupby(['site'])['year'].unique().to_dict()
    common_years = list(set.intersection(*(set(v) for v in dfs.values())))
    dfs = [df_grouped.loc[df_grouped['year'] == y] for y in common_years]
    dfs = pd.concat(dfs)

    names = dfs['site'].unique().tolist()
    fig, axs = plt.subplots(5)
    for i in range(len(names)):
        axs[i].plot(dfs.loc[dfs['site'] == names[i]]['GPP_ref'], label=names[i])
        axs[i].legend()
    fig.show

    # drop years without data
    # dfs['site'] = dfs['site'].astype('string')
    dfs = dfs[(dfs['site'] != 'le_bray') & (dfs['year'] != 2002)]
    dfs = dfs[(dfs['site'] != 'hyytiala') & (dfs['year'] != 2007)]

    return dfs