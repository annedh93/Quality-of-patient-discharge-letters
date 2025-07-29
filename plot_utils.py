# Load packages
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap


def make_barplot(group_indices, data, outcome, shift, save_name):
    colors = ['#66ffb3','#6699CC','#FF6666']
    barplot_tables = {}
    for i in group_indices:
        barplot_tables[i] = data.loc[group_indices[i], outcome].value_counts()
    width = 0.4-0.05*len(group_indices)
    location_shift = shift*(len(group_indices)-1)*0.5*width 
    for index, i in enumerate(barplot_tables):
        plt.bar(x=barplot_tables[i].keys()+location_shift[index], 
            height=100*barplot_tables[i]/barplot_tables[i].sum(), 
            width=width, color=colors[index], label=i)
    plt.legend(loc='best')
    plt.ylabel('%')
    plt.ylim((0,100))
    #plt.title(title)
    plt.savefig('figures/'+save_name, bbox_inches='tight')
    plt.show()


def get_number_of_annotations_per_letter(annotations, base):
    grouped_annotations = pd.DataFrame(annotations.groupby('evalphase2_id').type.value_counts())
    # Build a new dataframe with counts
    annotations_per_letter = pd.DataFrame()
    annotations_per_letter['evalphase2_id'] = grouped_annotations.index.get_level_values(0)
    annotations_per_letter['type'] = grouped_annotations.index.get_level_values(1)
    annotations_per_letter['count'] = np.array(grouped_annotations['type'])
    # Pivot the dataframe so that all letters have an entry for omission/hallucination/trivial fact
    annotations_per_letter = annotations_per_letter.pivot(index='evalphase2_id', columns='type', values='count').reset_index()
    # Check which evaluation IDs are in the base dataset, but not in the new subset
    evaluation_id_not_in_annotations = list(set(base.id)-set(annotations_per_letter['evalphase2_id']))
    # Make a df with the number of omissions/hallucinations/trivial facts for all included letters
    annotations_per_letter_not_in_annotations = pd.DataFrame({'evalphase2_id':evaluation_id_not_in_annotations, 'omission':np.nan, 'hallucination':np.nan, 'trivial':np.nan})
    annotations_per_letter_all = annotations_per_letter.append(annotations_per_letter_not_in_annotations, ignore_index=True)
    annotations_per_letter_all = annotations_per_letter_all.fillna(0)
    # Transform wide back to long dataframe
    annotations_per_letter_all = pd.melt(annotations_per_letter_all, id_vars='evalphase2_id', value_vars=['omission','hallucination','trivial'])
    # Merge with medical specialty and type of letter information from base
    return annotations_per_letter_all.merge(base.loc[:,['id','evaluated_letter','medical_specialty']], how='left', left_on='evalphase2_id', right_on='id').sort_values('evalphase2_id')

def get_statistics_per_letter(annotations_per_letter, letter_type, type_annotations):
    statistics_per_letter = pd.DataFrame(columns=['mean','std','median','25percentile','75percentile','min','max'])
    for type_annotation in type_annotations:
        variable = annotations_per_letter.loc[
            (annotations_per_letter.variable==type_annotation) &
            (annotations_per_letter.evaluated_letter==letter_type), 'value']
        statistics_per_letter = statistics_per_letter.append({
            'mean':np.round(np.mean(variable),1), 
            'std':np.round(np.std(variable),1), 
            'median':np.median(variable),
            '25percentile':np.percentile(variable,25),
            '75percentile':np.percentile(variable,75),
            'min':np.min(variable),
            'max':np.max(variable)}, ignore_index=True)
    statistics_per_letter.index = type_annotations
    return statistics_per_letter

def make_om_hal_triv_barplot(group_indices, data, outcome, shift, title, save_name, cutoff=False):
    colors = ['#6490FF','#FFB001','#DC267F']
    line_shift = [0,-0.03,0.03]
    barplot_tables = {}
    for i in group_indices:
        barplot_table = data.loc[group_indices[i], outcome].value_counts()
        if cutoff!=False:
            barplot_tables[i] = barplot_table[barplot_table.keys()<=cutoff].append(pd.Series({cutoff+1:barplot_table[barplot_table.keys()>cutoff].sum()}))
        else: barplot_tables[i] = barplot_table
    width = 0.4-0.05*len(group_indices)
    location_shift = shift*(len(group_indices)-1)*0.5*width
    for index, i in enumerate(barplot_tables):
        plt.bar(x=barplot_tables[i].keys()+location_shift[index], 
            height=100*barplot_tables[i]/barplot_tables[i].sum(), 
            width=width, color=colors[index], label=i)
    for index, i in enumerate(barplot_tables):
        plt.vlines(np.median(data.loc[group_indices[i],outcome])+line_shift[index], ymin=0, ymax=100, color=colors[index], ls='--')
    plt.legend(loc='best')
    if cutoff!=False: plt.xticks(ticks=range(0,cutoff+2), labels=list(range(0,cutoff+1))+['>='+str(cutoff+1)])
    plt.xlabel('Frequency per letter')
    plt.ylabel('%')
    plt.title(title)
    plt.savefig('figures/'+save_name, bbox_inches='tight')
    plt.show()

def make_om_hal_triv_barplot_importance(group_indices, data, outcome, shift, title, save_name, cutoff=False):
    colors = [['#6490FF','#FFB001','#DC267F'],['#CCCCFF','#FFCC99','#CC99CC']]
    line_shift = [0,-0.03,0.03]
    barplot_tables = {}
    for i in group_indices:
        group_tables = {}
        for dataset in data:
            barplot_table = data[dataset].loc[data[dataset][group_indices[i][0]]==group_indices[i][1], outcome].value_counts().sort_index()
            if cutoff!=False:
                group_tables[dataset] = barplot_table[barplot_table.keys()<=cutoff].append(pd.Series({cutoff+1:barplot_table[barplot_table.keys()>cutoff].sum()}))
            else: group_tables[dataset] = barplot_table
        barplot_tables[i] = group_tables
    width = 0.4-0.05*len(group_indices)
    horizontal_shift = shift*(len(group_indices)-1)*0.5*width
    for index, i in enumerate(barplot_tables):
        for index_d, dataset in enumerate(data):
            if dataset=='less important':
                    bottom = pd.Series([0]*(cutoff+2))
                    bottom_barplot = 100*barplot_tables[i]['important']/(barplot_tables[i]['important'].sum() + barplot_tables[i]['less important'].sum())
                    bottom.loc[bottom.index.isin(bottom_barplot.index)] = bottom_barplot
            else: bottom=0
            height = pd.Series([0]*(cutoff+2))
            height_barplot = 100*barplot_tables[i][dataset]/(barplot_tables[i]['important'].sum() + barplot_tables[i]['less important'].sum())
            height.loc[height.index.isin(height_barplot.index)] = height_barplot
            plt.bar(x=range(0,cutoff+2)+horizontal_shift[index], 
                height=height, 
                bottom=bottom,
                width=width, color=colors[index_d][index], label=dataset + ' ' + i)
    for index, i in enumerate(barplot_tables):
        for index_d, dataset in enumerate(data):
            plt.vlines(np.median(data[dataset].loc[data[dataset][group_indices[i][0]]==group_indices[i][1],outcome])+line_shift[index], ymin=0, ymax=100, color=colors[index_d][index], ls='--')
    plt.legend(loc='best')
    if cutoff!=False: plt.xticks(ticks=range(0,cutoff+2), labels=list(range(0,cutoff+1))+['>='+str(cutoff+1)])
    plt.xlabel('Frequency per letter')
    plt.ylabel('%')
    plt.title(title)
    plt.savefig('figures/'+save_name, bbox_inches='tight')
    plt.show()

def GPT_letter_with_hallucinations(hallucinations, raw_letter):
    GPT_letter_list = ast.literal_eval(raw_letter)
    GPT_letter = ''
    wrapper = textwrap.TextWrapper(width=100)
    for chunk in GPT_letter_list:
        for item in chunk:
            new_item = item + ': ' + chunk[item]
            for hallucination in hallucinations:
                new_item = new_item.replace(hallucination, '\033[44;33m'+hallucination+'\033[0m')
            newline = wrapper.fill(new_item) + '\n'
            GPT_letter = GPT_letter + newline
    return(GPT_letter)