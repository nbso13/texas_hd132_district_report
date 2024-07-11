## Author: Nicholas Ornstein
## Date: November 1st 2022
## Title: Analyzing demographics and party/turnout scores within Texas HD 132 precincts
## N.B.: set up to run from 'scripts' directory 
import this
import pandas as pd
from shapely import geometry
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import And
import os
import time

# print(os.getcwd())
# defining parameters
file_dir = '../data/'
file_stem = r'fiftyplusdemscorefinal_scrambled'
file_type = '.csv'
df_1 = pd.read_csv(file_dir+file_stem+file_type) #,  sep='\t', lineterminator='\r', encoding = 'utf-16le')

file_dir = '../data/'
file_stem = r'lessthanfiftydemscorefinal_scrambled'
file_type = '.csv'
df_2 = pd.read_csv(file_dir+file_stem+file_type)  #sep='\t', lineterminator='\r', encoding = 'utf-16le')

mid_dem_range = [30, 70]
mid_turnout_range = [30, 70]
high_dem_cutoff = 70
high_turnout_cutoff = 70
top_district_number = 3
timestr = time.strftime("%Y%m%d-%H%M%S")

# concat dfs -

df = pd.concat([df_1, df_2], axis=0)

def calc_stats(df, race_names):
    # calculate population
    pop = len(df['Voter File VANID'])
    # calculate median age
    median_age = np.nanmedian(df['Age'])
    age_bins = np.arange(0,110,10)
    age_frq, age_edges = np.histogram([age for age in df['Age'] if not np.isnan(age)], bins=age_bins)

    # calc sd age
    sd_age = np.std(df['Age'])

    # catch nan's:
    for name in df['RaceName']:
        if type(name) is not str:
            name = str(name)

    # calc race counts and percent
    race_counts = [np.sum(df['RaceName']==race) for race in race_names]
    race_percents = [100*race_count/pop for race_count in race_counts]
    race_percent_names = ["Percent " + race for race in race_names]
    female_count = np.sum(df['Sex'] == 'F')
    male_count = np.sum(df['Sex'] == 'M')
    percent_female = 100* female_count/(male_count+female_count)
    turnout_scores = df["TO2022"]
    turnout_bins = np.arange(0,110,10)
    turnout_frq, turnout_edges = np.histogram([toscore for toscore in df['TO2022'] if not np.isnan(toscore)], bins= turnout_bins)
    dem_party_support_scores = df["DNCDemPartySupportV1"]
    score_bins =  np.arange(0,110,10)
    dem_support_frq, dem_support_edges = np.histogram([score for score in dem_party_support_scores if not np.isnan(score)], bins= score_bins)
    gop_supp_scores = [100-sco for sco in dem_party_support_scores]
    gop_support_frq, gop_support_edges = np.histogram([score for score in gop_supp_scores if not np.isnan(score)], bins=score_bins)
    unit_vector = np.ones_like(dem_party_support_scores)
    pop_mid_turnout_high_dem = np.sum(((mid_turnout_range[0]*unit_vector <= turnout_scores) & (turnout_scores <= mid_turnout_range[1]*unit_vector)) & (dem_party_support_scores > high_dem_cutoff*unit_vector))
    
    high_dems = df[(dem_party_support_scores > high_dem_cutoff*unit_vector)]
    turnout_high_dem_frq, turnout_high_dem_edges = np.histogram([toscore for toscore in high_dems['TO2022'] if not np.isnan(toscore)])
    scaled_gotv = 100*pop_mid_turnout_high_dem/pop
    pop_high_turnout_mid_dem = np.sum(((mid_dem_range[0]*unit_vector <= dem_party_support_scores) & (dem_party_support_scores <= mid_dem_range[1]*unit_vector)) & (turnout_scores > high_turnout_cutoff*unit_vector))
    
    high_TO = df[(turnout_scores > high_turnout_cutoff*unit_vector)]
    dem_support_high_TO_frq, dem_support_high_TO_edges = np.histogram([toscore for toscore in high_TO['DNCDemPartySupportV1'] if not np.isnan(toscore)], bins=score_bins)

    scaled_flip = 100*pop_high_turnout_mid_dem/pop
    predicted_dem_votes = np.sum((dem_party_support_scores/(100*unit_vector))*(turnout_scores/(100*unit_vector)))
    pred_gop_votes = np.sum((gop_supp_scores/(100*unit_vector))*(turnout_scores/(100*unit_vector)))
    likely_dem_pop = np.sum(dem_party_support_scores>high_dem_cutoff)
    likely_gop_pop = np.sum(dem_party_support_scores<(100-high_dem_cutoff))
    median_dem_score = np.nanmedian(dem_party_support_scores)
    sd_dem_score = np.nanstd(dem_party_support_scores)
    median_turnout_score = np.nanmedian(turnout_scores)
    sd_turnout_score = np.nanstd(turnout_scores)
    # + race_percents
    stat_list = [pop, median_age, sd_age, percent_female, pop_mid_turnout_high_dem, scaled_gotv, pop_high_turnout_mid_dem, \
        scaled_flip, predicted_dem_votes, pred_gop_votes, likely_dem_pop,likely_gop_pop, median_dem_score, sd_dem_score,\
        median_turnout_score, sd_turnout_score]  + [[age_frq, age_edges], [turnout_high_dem_frq, turnout_high_dem_edges], \
        [dem_support_high_TO_frq, dem_support_high_TO_edges], [dem_support_frq, dem_support_edges], [turnout_frq, turnout_edges]]
    return stat_list


# figure out list of possible races
races_list = list(set(df['RaceName']))
races = [str(race) for race in races_list]

# race_count_names = [race + " Count" for race in races]
race_percent_names = ["Percent " + race  for race in races]
all_precincts_list = list(set(df['PrecinctName']))
all_precincts = []

# populate precinct list
for i in all_precincts_list:
    if not np.isnan(i):
        all_precincts.append(i)

all_precincts_strs = [str(int(prec)) for prec in all_precincts]


print(f"Pop without precinct recorded: {np.sum(np.isnan(df['PrecinctName']))}")

# race_percent_names +
all_stats  = ["Population", "Median Age", "S.D. Age", "Percent Female",\
        "Pop mid turnout high dem", "mid turnout high dem scaled by pop (concentration)", "Pop mid dem high TO", "Mid dem high TO scaled by pop (concentration)" ,\
        "Predicted Dem Votes", "Predicted GOP votes", "Likely Dem Pop", "Likely GOP Pop",  \
        "Median Dem Score", "S.D. Dem Score", "Median Turnout Score", "S.D. Turnout Score"] + \
        ["Age", "Turnout in High Dem", "Dem Support in High Turnout", "Dem Support Scores", "Turnout Scores"]
    # + len(race_percent_names)*["Greens"]
color_maps_by_stat = ["Greens", "Purples", "Purples", "Oranges", "Greens", "Greens", "Purples", "Purples", "Blues", "Reds", "Blues", "Reds", "Blues", "Blues", "Purples", "Purples"] 
# + len(race_percent_names)*[0]
hist_flags_by_stat =  [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0] # statistics for which we generate a histogram
stat_data = {}

for stat in all_stats:
    stat_data[stat] = list()

for prec in all_precincts: #breaking down by precinct
    new_df = df[df['PrecinctName'] == prec]
    stat_list = calc_stats(new_df, races )
    for i in range(len(all_stats)):
        target_list = stat_data[all_stats[i]]
        target_list.append(stat_list[i])

indices = all_precincts_strs + ["House District"]
stat_list = calc_stats(df, races)
for i in range(len(all_stats)):
    target_list = stat_data[all_stats[i]]
    target_list.append(stat_list[i])

stats_df_old = pd.DataFrame(stat_data, index=indices)
# print(stats_df_old)
house_district_series = stats_df_old.loc["House District", :]
stats_df = stats_df_old.drop(index=["House District"])


# final_file_stem = "whole_district"
# stats_df.to_excel(f"summary_stats_for_{final_file_stem}.xlsx") 
    # av age
    
# take off last row which is just summary stats
# stats_df = stats_df.drop(index=["House District"], axis=0)


# read csv
fp = "132nd_Shapfile-point.shp"
os.chdir("../data/mygeodata")
# Read file using gpd.read_file()
data = gpd.read_file(fp)
data_crs = data.crs

polygons = []
for id in all_precincts_strs:
    polygons.append(geometry.Polygon([[data['geometry'][p].x, data['geometry'][p].y] for p in range(len(data)) if data['id'][p] == id ]))
geopd_df = gpd.GeoDataFrame(stats_df, crs=data_crs, geometry=polygons)

histogram_stat_counter = 0

# for each stat that is not a histogram stat - main loop
for stat_index in range(len(all_stats[0:-6])): 
    # get the column of results for each precinct for this stat
    this_stat = geopd_df[all_stats[stat_index]]
    # sort by this stat
    this_stat = this_stat.sort_values(ascending=False)
    sorted_inds= this_stat.index
    # grab the top n precincts
    top_precincts_names = list(sorted_inds[0:top_district_number])
    top_precincts = geopd_df.loc[top_precincts_names, :]
    # grab the median precinct
    # median_precinct_name = this_stat.index[int(len(all_precincts_strs)/2)]
    #if this is a stat that gets a histogram
    if hist_flags_by_stat[stat_index]:
        # four plots - the first three are the top ones the last one is the median district for comparison
        # find column index into data frame
        # the last six are histogram columns
        hist_vals_for_this_stat = geopd_df[all_stats[-5 + histogram_stat_counter]]
        # check that the histograms are for the right statistic
        print(all_stats[-5 + histogram_stat_counter] + " and " + all_stats[stat_index] + " are being plotted together.")
        #increment counter to keep track of where on the histogram list you are

        fig, axes = plt.subplots(2,2)
        plt.suptitle(all_stats[-5 + histogram_stat_counter])
        axes_list = axes.ravel()
        max_y = np.max(np.hstack(hist_vals_for_this_stat[top_precincts_names][0]))
        # for every top precinct, plot the histogram.
        for top_precinct_index in range(len(top_precincts_names)):
            freqs = hist_vals_for_this_stat[top_precincts_names[top_precinct_index]][0]
            edges = hist_vals_for_this_stat[top_precincts_names[top_precinct_index]][1]
            cmap =  plt.cm.get_cmap(color_maps_by_stat[stat_index])
            axes_list[top_precinct_index].bar(edges[:-1], freqs, width=np.diff(edges), facecolor=cmap(0.5), edgecolor="black", align="edge")
            axes_list[top_precinct_index].set_ylim([0, max_y])
            precinct_stat_value = this_stat[top_precincts_names[top_precinct_index]]
            axes_list[top_precinct_index].set_title(f"Precinct: {top_precincts_names[top_precinct_index]}, {all_stats[stat_index]}: {precinct_stat_value} ")
            if precinct_stat_value < np.max(edges):
                axes_list[top_precinct_index].axvline(x=precinct_stat_value, color='red')

        # do last one as median for comparison
        freqs_house = house_district_series[all_stats[-5 + histogram_stat_counter]][0]
        edges_house = house_district_series[all_stats[-5 + histogram_stat_counter]][1]
        axes_list[top_precinct_index+1].bar(edges_house[:-1], freqs_house, width=np.diff(edges), facecolor=cmap(0.5), edgecolor="black", align="edge")
        district_stat_value = house_district_series[all_stats[stat_index]]
        # axes_list[top_precinct_index+1].set_ylim([0, max_y])
        if district_stat_value > np.max(edges):
            district_stat_value = district_stat_value/len(all_precincts_strs)
        if district_stat_value < np.max(edges):
            axes_list[top_precinct_index+1].axvline(x=district_stat_value, color='red')
            axes_list[top_precinct_index+1].set_title(f"District: {all_stats[stat_index]} ")
        else:
            axes_list[top_precinct_index+1].set_title(f"District: {all_stats[stat_index]}/#precincts: {district_stat_value} ")
        
        histogram_stat_counter = histogram_stat_counter+1
        fig.set_size_inches(13, 7)
        plt.savefig(f'../../results/{timestr}_{all_stats[stat_index]}_histogram.png', bbox_inches='tight')


    # get locations of each annotation
    # minx, miny, maxx, maxy = df.geometry.total_bounds
    
    print('making map plot')
    # make the map plot
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(13, 7)
    plt.axis('off')
    # set the title of the map
    ax.set_title(all_stats[stat_index])
    # adjust plot colorbar stuff
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    # plot it with the desired colormap
    geopd_df.plot(column=all_stats[stat_index], ax=ax, legend=True, cax = cax, cmap=color_maps_by_stat[stat_index])

    top_precincts['coords'] = top_precincts['geometry'].apply(lambda x: x.representative_point().coords[:])
    top_precincts['coords'] = [coords[0] for coords in top_precincts['coords']]

    for idx, row in top_precincts.iterrows():
        ax.annotate(text=idx, xy=row['coords'],
                    horizontalalignment='center', color = 'white')
    plt.savefig(f'../../results/{timestr}_{all_stats[stat_index]}.png', bbox_inches='tight')
n=1

