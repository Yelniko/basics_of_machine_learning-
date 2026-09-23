from pandas.plotting import scatter_matrix
from ucimlrepo import fetch_ucirepo
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings



def main():
    warnings.filterwarnings('ignore')
    room_occupancy_estimation = fetch_ucirepo(id=864)

    X = room_occupancy_estimation.data.features
    y = room_occupancy_estimation.data.targets

    print(room_occupancy_estimation.metadata)
    print(room_occupancy_estimation.variables)
    target = list(y.columns)[0]
    print(target)

    df = pd.concat([X, y], axis=1)
    print(df)

    print("The shape of the train data is (row, column):" + str(df.shape))
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    print(f"Кількість дублів: {df.duplicated().sum()}")

    print(pd.DataFrame(df.groupby('S6_PIR')[target].value_counts()))
    print(pd.DataFrame(df.groupby('S7_PIR')[target].value_counts()))

    print('Values = 0')
    for i in range(1, 5):
        mask = df['S' + str(i) + '_Light']
        print(f'S{i}_Light: {mask[mask == 0].count()}')

    print('0 < Values =< 20')
    for i in range(1, 5):
        mask = df['S' + str(i) + '_Light']
        print(f'S{i}_Light: {mask[mask <= 20].count()}')

    fig, ax = plt.subplots(1, 4, figsize=(14, 5))

    plt.subplots_adjust(wspace=0.5)

    sns.boxplot(data=df['S1_Light'], ax=ax[0], color='brown')
    ax[0].set_xlabel('S1_Light')

    sns.boxplot(data=df['S2_Light'], ax=ax[1], color='g')
    ax[1].set_xlabel('S2_Light')

    sns.boxplot(data=df['S3_Light'], ax=ax[2])
    ax[2].set_xlabel('S3_Light')

    sns.boxplot(data=df['S4_Light'], ax=ax[3], color='y')
    ax[3].set_xlabel('S4_Light')

    plt.suptitle('Light"s outliers')

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    plt.subplots_adjust(wspace=0.5)

    sns.boxplot(data=df['S5_CO2'], ax=ax[0], color='b')
    ax[0].set_xlabel('S5_CO2')

    sns.boxplot(data=df['S5_CO2_Slope'], ax=ax[1], color='g')
    ax[1].set_xlabel('S5_CO2_Slope')

    plt.suptitle('CO2"s outliers')

    fig, ax = plt.subplots(1, 4, figsize=(14, 5))

    plt.subplots_adjust(wspace=0.5)

    sns.boxplot(data=df['S1_Sound'], ax=ax[0], color='brown')
    ax[0].set_xlabel('S1_Sound')

    sns.boxplot(data=df['S2_Sound'], ax=ax[1], color='g')
    ax[1].set_xlabel('S2_Sound')

    sns.boxplot(data=df['S3_Sound'], ax=ax[2])
    ax[2].set_xlabel('S3_Sound')

    sns.boxplot(data=df['S4_Sound'], ax=ax[3], color='y')
    ax[3].set_xlabel('S4_Sound')

    plt.suptitle('Sound"s outliers')

    columns = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp', 'S1_Light',
               'S2_Light', 'S3_Light', 'S4_Light', 'S1_Sound', 'S2_Sound', 'S3_Sound',
               'S4_Sound', 'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR']
    sns.set_theme(style='whitegrid', rc={'figure.figsize': (16, 16)})

    fig, axs = plt.subplots(4, 4)
    fig.suptitle("Features Density | Room_Occupancy_Count", y=.9)
    count = 0
    for x in range(4):
        for y in range(4):
            ax = sns.kdeplot(data=df, x=columns[count], hue="Room_Occupancy_Count", ax=axs[x, y])
            ax.set(ylabel=None)
            ax.tick_params(labelsize=9)
            sns.move_legend(ax, "upper right", ncol=1, title=None, frameon=False)
            count += 1

    df_notime = df.drop(['Date', 'Time'], axis=1)
    corr = df_notime.corr(method='spearman', numeric_only=True)

    mask = np.zeros_like(corr, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True
    sns.set_style('whitegrid')
    plt.subplots(figsize=(15, 12))
    sns.heatmap(corr, annot=True, mask=mask, cmap='RdBu_r',
                linewidths=.9, linecolor='white', fmt='.2g', center=0, square=True)
    plt.title('Paired correlation between features', y=1, fontsize=20)

    plt.show()
    plt.close()

    print(pd.DataFrame(df[target].value_counts(normalize=True) * 100))
    sns.set_theme(style='whitegrid', palette='vlag', rc={'figure.figsize': (10, 6)})
    ax = sns.countplot(x=df.Room_Occupancy_Count)
    ax.set(title='Distribution of Room_Occupancy_Count');
    print(df_notime.columns)
    scatter_matrix(df_notime, alpha=0.8, figsize=(8, 8), diagonal='kde')

    plt.show()
    plt.clf()
    df['Room_Occupancy_Count'].value_counts().plot(kind='pie')

    plt.show()

    params = (df.select_dtypes(include=np.number).columns.drop('Room_Occupancy_Count'))

    for i, j in combinations(params, 2):
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.scatter(df[i], df[j], s=10, alpha=0.5)

        ax.set_title(f'{i} vs {j}')
        ax.set_xlabel(i)
        ax.set_ylabel(j)

        plt.show()
        plt.close(fig)

    print(df['Date'][0])
    df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')
    print(type(df['Date'][0]))

    selected_year = 2017
    selected_month = 12

    filtered_df = df[(df['Date'].dt.year == selected_year) & (df['Date'].dt.month == selected_month)]
    print(filtered_df)

    filtered_df = df[df['Room_Occupancy_Count'] > 0]
    print(filtered_df)

    filtered_df = df[df['S5_CO2'] > 700]
    print(filtered_df)

    filtered_df = df[
        (df['S1_Temp'] >= 22) &
        (df['S1_Temp'] <= 25)
        ]
    print(filtered_df)

    filtered_df = df[
        (df['Date'].dt.year == 2017) &
        (df['Date'].dt.month == 12) &
        (df['Room_Occupancy_Count'] > 0) &
        (df['S5_CO2'] > 700)
        ]
    print(filtered_df)

    filtered_df = df[df['Room_Occupancy_Count'].isin([2, 3])]
    print(filtered_df)


if __name__ == "__main__":
    main()