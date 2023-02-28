#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
To rank the predicted structures of drugs using a ranking method.
To compare whether any predicted structure is consistent with the real structure.
"""
import numpy as np
import pandas as pd


def rank_central(drug, generated_samples, SHIFT_DENSITY):
    generated_samples['Density bias'] = abs(generated_samples['Calculated density'] - (drug['Predicted density'] + SHIFT_DENSITY))
    generated_samples['Rank central'] = generated_samples['Density bias'].rank(ascending=True)
    ranked_structures = generated_samples.sort_values(by=['Rank central'], ascending=True, ignore_index=True)
    return ranked_structures

def compare(structure, generated_samples, CRITERION_EDGE, CRITERION_ANGLE):
    # Compares given structure to a generated sample
    structures = pd.concat([structure.to_frame().T]*6).reset_index(drop=True)
    structures.iloc[1, :][['B', 'C']] = structures.iloc[1, :][['C', 'B']]
    structures.iloc[1, :][['Beta', 'Gamma']] = structures.iloc[1, :][['Gamma', 'Beta']]
    structures.iloc[2, :][['A', 'B']] = structures.iloc[2, :][['B', 'A']]
    structures.iloc[2, :][['Alpha', 'Beta']] = structures.iloc[2, :][['Beta', 'Alpha']]
    structures.iloc[3, :][['A', 'B', 'C']] = structures.iloc[3, :][['B', 'C', 'A']]
    structures.iloc[3, :][['Alpha', 'Beta', 'Gamma']] = structures.iloc[3, :][['Beta', 'Gamma', 'Alpha']]
    structures.iloc[4, :][['A', 'B', 'C']] = structures.iloc[4, :][['C', 'A', 'B']]
    structures.iloc[4, :][['Alpha', 'Beta', 'Gamma']] = structures.iloc[4, :][['Gamma', 'Alpha', 'Beta']]
    structures.iloc[5, :][['A', 'C']] = structures.iloc[5, :][['C', 'A']]
    structures.iloc[5, :][['Alpha', 'Gamma']] = structures.iloc[5, :][['Gamma', 'Alpha']]

    # Compare the given structure with the generated samples
    matched_structures = pd.DataFrame()
    for i, structure in structures.iterrows():
        new_matched_structures = generated_samples[generated_samples.apply(lambda x: ((x['Space group number'] == structure['Space group number']) &
                                                                               (x['Z value'] == structure['Z value']) &
                                                                               (abs(x['A'] - structure['A']) <= (structure['A'] * CRITERION_EDGE)) &
                                                                               (abs(x['B'] - structure['B']) <= (structure['B'] * CRITERION_EDGE)) &
                                                                               (abs(x['C'] - structure['C']) <= (structure['C'] * CRITERION_EDGE)) &
                                                                               (abs(x['Alpha'] - structure['Alpha']) <= (structure['Alpha'] * CRITERION_ANGLE)) &
                                                                               (abs(x['Beta'] - structure['Beta']) <= (structure['Beta'] * CRITERION_ANGLE)) &
                                                                               (abs(x['Gamma'] - structure['Gamma']) <= (structure['Gamma'] * CRITERION_ANGLE))), axis=1)]
        if not new_matched_structures.empty:
            matched_structures = pd.concat([matched_structures, new_matched_structures], axis=0)

    # Drop duplicates and sort the dataframe
    matched_structures = matched_structures.drop_duplicates()
    matched_structures.sort_index(inplace=True)

    # Return the matched structures
    if not matched_structures.empty:
        return 1, matched_structures
    else:
        return 0, matched_structures


def main():
    # constants
    N_DRUGS = 264 # 264
    CRITERION_EDGE = 0.1
    CRITERION_ANGLE = 0.1
    RANK_CENTRAL = True
    SHIFT_DENSITY = 0.0 # 0.0
    N_SAMPLES = 1000

    # files
    source_path = '../../data/monomorph/drugs_monomorph_compounds.csv'
    source_file_path = '../../data/monomorph/drugs_monomorph.csv'
    result_file_path = '../../results/monomorph_pregenerated/'

    # Read files into dataframes
    drugs = pd.read_csv(source_path, encoding="ISO-8859-1", engine='python')
    structures_drugs = pd.read_csv(source_file_path, encoding="ISO-8859-1", engine='python')

    # start
    l, l_hit, ls_rank, l_fail = [], [], [], []
    # Loop through all drugs
    for i in range(N_DRUGS):
        drug = drugs.iloc[i, :]
        print('#%d drug:' % (i+1))
        gb = structures_drugs.groupby('InChI string')
        structures_drug = gb.get_group(drug['InChI string'])

        # Open generated crystal structure file
        try:
            generated_samples = pd.read_csv(result_file_path + '%s_generated.csv' % (i+1), encoding="ISO-8859-1", engine='python')
        except:
            print('#%d drug: the generated crystal structure file cannot be opened.' % (i+1))
            continue

        # Rank
        if RANK_CENTRAL:
            generated_samples = rank_central(drug, generated_samples, SHIFT_DENSITY)

        # Evaluate
        num_polymorph = structures_drug.shape[0]
        l_recall, l_rank, matched_structures = [], [], pd.DataFrame()
        for j, structure in structures_drug.iterrows():
            res, new_matched_structures = compare(structure, generated_samples, CRITERION_EDGE, CRITERION_ANGLE)
            l_recall.append(res)

            if not new_matched_structures.empty:
                l_rank.append(new_matched_structures.index[0]+1)
                matched_structures = pd.concat([matched_structures, new_matched_structures], axis=0)
            else:
                l_rank.append(0)

        # Check for results
        if not matched_structures.empty:
            matched_structures = matched_structures.drop_duplicates()
            # pd.DataFrame(matched_structures).to_csv(result_file_path + '%d_matched.csv' % (i+1), index=False, na_rep='None')
        else:
            l_fail.append((i+1))
        print('#%d drug: %d matched structures.' % ((i+1), matched_structures.shape[0]))

        ls_rank.append(l_rank)

        if sum(l_recall) == len(l_recall):
            l.append(1)
            l_hit.append(1)
        elif sum(l_recall) > 0:
            l.append(0)
            l_hit.append(1)
        else:
            l.append(0)
            l_hit.append(0)        

    # Calculate accuracy, hit rate, and averaged rank
    accuracy = round(sum(l)/len(l), 4)
    hit = round(sum(l_hit)/len(l_hit), 4)
    rank = round(sum([sum(x) for x in ls_rank])/sum([len(x) - x.count(0) for x in ls_rank]), 4)
    print('%d drugs Accuracy: ' % len(l), accuracy)
    print('%d drugs Hit rate: ' % len(l_hit), hit)
    print('%d drugs Averaged rank: ' % len(ls_rank), rank)
    # print('Rank:', ls_rank)
    # print('Fails:', l_fail)

if __name__ == "__main__":
    main()