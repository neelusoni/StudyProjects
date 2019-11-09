import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
from PIL import Image
import folium
from folium import plugins
import webbrowser

def save_bar_graph_to_figure(df,col_list,fig_path, title):
    '''
    Generates Horizontal Stacked Bar graph using columns of dataframe and saves the figure for future use
    Args:
        df: Data to be used for plotting
        col_list: List of dataframe columns for plot
        fig_path: Location to save the generated plot
        title: Title for the plot
    Returns: None, Saves the generated plot in specified location
    '''
    df = df[col_list]

    fig = plt.figure(figsize=(15.5, 7.5))
    y_pos = np.arange(len(df[col_list[0]]))
    ax = fig.add_subplot(111)

    ax.set_title(title)

    df.plot(kind='barh',stacked=True,ax=ax)
    list_values = (df[col_list[1]].tolist()+
                   df[col_list[2]].tolist())

    for rect, value in zip(ax.patches, list_values):
        h = rect.get_height() / 2.
        w = rect.get_width() / 2.
        x, y = rect.get_xy()
        ax.text(x + w, y + h, value, horizontalalignment='center', verticalalignment='center')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df[col_list[0]])

    ax.legend(loc='upper center',bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True,ncol=2)
    plt.tight_layout()
    fig.savefig(fig_path)

if __name__ == '__main__':

    df_can = pd.read_excel('Canada.xlsx',
                       sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                       skipfooter=2
                      )

    df_can.drop(['AREA', 'REG', 'DEV', 'Type', 'Coverage'], axis=1, inplace=True)
    df_can.rename(columns={'OdName': 'Country', 'AreaName': 'Continent', 'RegName': 'Region'}, inplace=True)
    all(isinstance(column, str) for column in df_can.columns)
    all(isinstance(column, str) for column in df_can.columns)

    df_vis = df_can.copy()

    df_vis.set_index('Country', inplace=True)
    df_vis['Total'] = df_vis.sum(axis=1)

    years = list(map(int, range(1980, 2014)))

    mpl.style.use('ggplot')  # optional: for ggplot-like style

    df_vis.sort_values(['Total'], ascending=False, axis=0, inplace=True)

    # get the top 5 entries
    df_top5 = df_vis.head()

    # transpose the dataframe
    df_top5 = df_top5[years].transpose()
    df_top5.index = df_top5.index.map(int)  # let's change the index values of df_top5 to type integer for plotting

    plot_type = 'choropleth'

    if plot_type == 'area':
        #Area Plot
        df_top5.plot(kind='area',
                     alpha=0.35,  # transparency value: 0-1, default value a= 0.5
                     stacked=False,
                     figsize=(20, 10),  # pass a tuple (x, y) size
                     )

        plt.title('Immigration Trend of Top 5 Countries')
        plt.ylabel('Number of Immigrants')
        plt.xlabel('Years')
        plt.show()

    if plot_type == 'hist':
        df_vis.loc[['Denmark', 'Norway', 'Sweden'], years]
        df_t = df_vis.loc[['Denmark', 'Norway', 'Sweden'], years].transpose()

            # let's get the x-tick values
        count, bin_edges = np.histogram(df_t, 15)
        xmin = bin_edges[0] - 10  # first bin value is 31.0, adding buffer of 10 for aesthetic purposes
        xmax = bin_edges[-1] + 10  # last bin value is 308.0, adding buffer of 10 for aesthetic purposes

            # un-stacked histogram
        df_t.plot(kind='hist',
                      figsize=(10, 6),
                      bins=15,
                      alpha=0.6,
                      xticks=bin_edges,
                      color=['coral', 'darkslateblue', 'mediumseagreen'],
                      stacked=True,
                      xlim=(xmin, xmax),
                      )

        plt.title('Histogram of Immigration from Denmark, Norway, and Sweden from 1980 - 2013')
        plt.ylabel('Number of Years')
        plt.xlabel('Number of Immigrants')
        plt.show()

    if plot_type == 'bar':
        df_iceland = df_vis.loc['Iceland', years]
        df_iceland.plot(kind='bar', figsize=(10, 6), rot=90)# rotate the bars by 90 degrees, use 'barh' for horizontal bars

        plt.xlabel('Year')  # add to x-label to the plot
        plt.ylabel('Number of immigrants')  # add y-label to the plot
        plt.title('Icelandic immigrants to Canada from 1980 to 2013')  # add title to the plot

            # Annotate arrow
        plt.annotate('',  # s: str. Will leave it blank for no text
                         xy=(32, 70),  # place head of the arrow at point (year 2012 , pop 70)
                         xytext=(28, 20),  # place base of the arrow at point (year 2008 , pop 20)
                         xycoords='data',  # will use the coordinate system of the object being annotated
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
                         )

            # Annotate Text
        plt.annotate('2008 - 2011 Financial Crisis',  # text to display
                         xy=(28, 30),  # start the text at at point (year 2008 , pop 30)
                         rotation=72.5,  # based on trial and error to match the arrow
                         va='bottom',  # want the text to be vertically 'bottom' aligned
                         ha='left',  # want the text to be horizontally 'left' algned.
                         )
        plt.show()

    if plot_type == 'line':
        df_top5 = df_vis.nlargest(5, 'Total')
        df_top5 = df_top5[years].transpose()

        df_top5.head()

        df_top5.plot(kind='line')
        plt.title('Immigration from top 5 countries')
        plt.ylabel('Number of Immigrants')
        plt.xlabel('Years')
        plt.show()

    if plot_type == 'pie':
        df_continents = df_vis.groupby('Continent',axis=0).sum()
        colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
        explode_list = [0.1, 0, 0, 0, 0.1, 0.1]  # ratio for each continent with which to offset each wedge.

        df_continents['Total'].plot(kind='pie',
                          figsize=(5, 6),
                          autopct='%1.1f%%', # add in percentages
                          startangle=90,     # start angle 90° (Africa)
                          shadow=True,       # add shadow
                          labels=None,
                          pctdistance = 1.12,
                          colors=colors_list,
                          explode = explode_list,
                        )

        plt.title('Immigration to Canada by Continent [1980 - 2013]')
        plt.axis('equal') # Sets the pie chart to look like a circle.
        plt.legend(labels=df_continents.index, loc='upper left')
        plt.show()

    if plot_type == 'box':
        df_CI = df_vis.loc[['China','India'], years].transpose()
        df_CI.plot(kind='box', figsize=(8, 6),color='blue', vert=False)

        plt.title('Box plot of India vs China Immigrants from 1980 - 2013')
        plt.xlabel('Number of Immigrants')
        plt.show()

    if plot_type == 'scatter':
        # we can use the sum() method to get the total population per year
        df_tot = pd.DataFrame(df_vis[years].sum(axis=0))
        # change the years to type int (useful for regression later on)
        df_tot.index = map(int, df_tot.index)
        # reset the index to put in back in as a column in the df_tot dataframe
        df_tot.reset_index(inplace=True)
        # rename columns
        df_tot.columns = ['year', 'total']

        x = df_tot['year']  # year on x-axis
        y = df_tot['total']  # total on y-axis
        fit = np.polyfit(x, y, deg=1)

        df_tot.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='darkblue')
        plt.plot(x, fit[0] * x + fit[1], color='red')  # recall that x is the Years
        plt.annotate('y={0:.0f} x + {1:.0f}'.format(fit[0], fit[1]), xy=(2000, 150000))

        plt.title('Total Immigration to Canada from 1980 - 2013')
        plt.xlabel('Year')
        plt.ylabel('Number of Immigrants')
        plt.show()

    if plot_type == 'bubble':
        fig = plt.figure()  # create figure

        ax0 = fig.add_subplot(1, 2, 1)  # add subplot 1 (1 row, 2 columns, first plot)
        ax1 = fig.add_subplot(1, 2, 2)  # add subplot 2 (1 row, 2 columns, second plot)

        df_can_t = df_vis[years].transpose()  # transposed dataframe
        # cast the Years (the index) to type int
        df_can_t.index = map(int, df_can_t.index)
        # let's label the index. This will automatically be the column name when we reset the index
        df_can_t.index.name = 'Year'
        # reset index to bring the Year in as a column
        df_can_t.reset_index(inplace=True)
        # normalize India data
        norm_india = (df_can_t['India'] - df_can_t['India'].min()) / (
                    df_can_t['India'].max() - df_can_t['India'].min())
        # normalize India data
        norm_china = (df_can_t['China'] - df_can_t['China'].min()) / (
                df_can_t['China'].max() - df_can_t['China'].min())

        ax0 = df_can_t.plot(kind='scatter',
                            x='Year',
                            y='India',
                            figsize=(14, 8),
                            alpha=0.5,  # transparency
                            color='green',
                            s=norm_india * 2000 + 10,  # pass in weights
                            xlim=(1975, 2015)
                            )

        # China
        ax1 = df_can_t.plot(kind='scatter',
                            x='Year',
                            y='China',
                            alpha=0.5,
                            color="blue",
                            s=norm_china * 2000 + 10,
                            ax=ax0
                            )

        ax0.set_ylabel('Number of Immigrants')
        ax0.set_title('Immigration from India and China from 1980 - 2013')
        ax0.legend(['India','China'], loc='upper left', fontsize='x-large')

        plt.show()

    if plot_type == 'waffle':
        width = 40  # width of chart
        height = 10  # height of chart
        value_sign = '%' #default is blank string
        df_dsn = df_vis.loc[['Denmark', 'Norway', 'Sweden'], :]

        categories = df_dsn.index.values  # categories
        values = df_dsn['Total']  # correponding values of categories

        colormap = plt.cm.coolwarm  # color map class

        # compute the proportion of each category with respect to the total
        total_values = sum(values)
        category_proportions = [(float(value) / total_values) for value in values]

        # compute the total number of tiles
        total_num_tiles = width * height  # total number of tiles
        print('Total number of tiles is', total_num_tiles)

        # compute the number of tiles for each catagory
        tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]

        # print out number of tiles per category
        for i, tiles in enumerate(tiles_per_category):
            print(df_dsn.index.values[i] + ': ' + str(tiles))

        # initialize the waffle chart as an empty matrix
        waffle_chart = np.zeros((height, width))

        # define indices to loop through waffle chart
        category_index = 0
        tile_index = 0

        # populate the waffle chart
        for col in range(width):
            for row in range(height):
                tile_index += 1

                # if the number of tiles populated for the current category
                # is equal to its corresponding allocated tiles...
                if tile_index > sum(tiles_per_category[0:category_index]):
                    # ...proceed to the next category
                    category_index += 1

                    # set the class value to an integer, which increases with class
                waffle_chart[row, col] = category_index

        # use matshow to display the waffle chart
        colormap = plt.cm.coolwarm
        plt.matshow(waffle_chart, cmap=colormap)
        plt.colorbar()

        # get the axis
        ax = plt.gca()

        # set minor ticks
        ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
        ax.set_yticks(np.arange(-.5, (height), 1), minor=True)

        # add dridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

        plt.xticks([])
        plt.yticks([])

        # compute cumulative sum of individual categories to match color schemes between chart and legend
        values_cumsum = np.cumsum(values)
        total_values = values_cumsum[len(values_cumsum) - 1]

        # create legend
        legend_handles = []
        for i, category in enumerate(categories):
            if value_sign == '%':
                label_str = category + ' (' + str(values[i]) + value_sign + ')'
            else:
                label_str = category + ' (' + value_sign + str(values[i]) + ')'

            color_val = colormap(float(values_cumsum[i]) / total_values)
            legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

        # add legend to chart
        plt.legend(
            handles=legend_handles,
            loc='lower center',
            ncol=len(categories),
            bbox_to_anchor=(0., -0.2, 0.95, .1)
        )

        plt.show()

    if plot_type == "word":
        # open the file and read it into a variable alice_novel
        alice_novel = open('alice_novel.txt', 'r').read()

        stopwords = set(STOPWORDS)
        stopwords.add('said')  # add the words said to stopwords

        # save mask to alice_mask
        alice_mask = np.array(Image.open('alice_mask.jpg'))

        # instantiate a word cloud object
        alice_wc = WordCloud(background_color='white', max_words=2000, mask=alice_mask, stopwords=stopwords)

        # generate the word cloud
        alice_wc.generate(alice_novel)

        # display the cloud
        fig = plt.figure()
        fig.set_figwidth(14)
        fig.set_figheight(18)

        plt.imshow(alice_wc, interpolation='bilinear')
        plt.axis('off')
        plt.show()

    if plot_type == "regression":
        # we can use the sum() method to get the total population per year
        df_tot = pd.DataFrame(df_vis[years].sum(axis=0))

        # change the years to type float (useful for regression later on)
        df_tot.index = map(float, df_tot.index)

        # reset the index to put in back in as a column in the df_tot dataframe
        df_tot.reset_index(inplace=True)

        # rename columns
        df_tot.columns = ['year', 'total']

        # view the final dataframe
        df_tot.head()

        plt.figure(figsize=(15, 10))

        sns.set(font_scale=1.5)
        sns.set_style('whitegrid')

        ax = sns.regplot(x='year', y='total', data=df_tot, color='green', marker='+', scatter_kws={'s': 200})
        ax.set(xlabel='Year', ylabel='Total Immigration')
        ax.set_title('Total Immigration to Canada from 1980 - 2013')

        plt.show()

    if plot_type == "folium":
        # define the world map
        world_map = folium.Map(location=[56.130, -106.35], zoom_start=4, tiles='Stamen Toner') #other values of tiles 'Stamen Terrain', 'Mapbox Bright'

        df_incidents = pd.read_csv('Police_Department_Incidents_-_Previous_Year__2016_.csv')
        limit = 100
        df_incidents = df_incidents.iloc[0:limit, :] #first 100 crimes

        # San Francisco latitude and longitude values
        latitude = 37.77
        longitude = -122.42

        sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)

        # instantiate a mark cluster object for the incidents in the dataframe
        incidents = plugins.MarkerCluster().add_to(sanfran_map)

        # loop through the dataframe and add each data point to the mark cluster
        for lat, lng, label, in zip(df_incidents.Y, df_incidents.X, df_incidents.Category):
            folium.Marker(
                location=[lat, lng],
                icon=None,
                popup=label,
            ).add_to(incidents)

        sanfran_map.save("/Users/neelu/StudyProjects/StudyProjects/sanfranmap.html")
        webbrowser.open("file:///Users/neelu/StudyProjects/StudyProjects/sanfranmap.html")

    if plot_type == "choropleth":
        df_can.columns = list(map(str, df_can.columns))
        # add total column
        df_can['Total'] = df_can.sum(axis=1)

        # years that we will be using in this lesson - useful for plotting later on
        years = list(map(str, range(1980, 2014)))

        # let's rename the columns so that they make sense
        world_geo = r'world_countries.json'  # geojson file

        # create a numpy array of length 6 and has linear spacing from the minium total immigration to the maximum total immigration
        threshold_scale = np.linspace(df_can['Total'].min(),
                                      df_can['Total'].max(),
                                      6, dtype=int)
        threshold_scale = threshold_scale.tolist()  # change the numpy array to a list
        threshold_scale[-1] = threshold_scale[
                                  -1] + 1  # make sure that the last value of the list is greater than the maximum immigration

        # let Folium determine the scale.
        world_map = folium.Map(location=[0, 0], zoom_start=2, tiles='Mapbox Bright')
        world_map.choropleth(
            geo_data=world_geo,
            data=df_can,
            columns=['Country', 'Total'],
            key_on='feature.properties.name',
            threshold_scale=threshold_scale,
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Immigration to Canada',
            reset=True
        )
        world_map.save("/Users/neelu/StudyProjects/StudyProjects/worldmap.html")
        webbrowser.open("file:///Users/neelu/StudyProjects/StudyProjects/worldmap.html")
