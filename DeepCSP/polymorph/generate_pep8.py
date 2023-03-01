#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Validate the DeepCSP framework using listed drugs.
"""
import threading
from multiprocessing import Pool
import tqdm
import pandas as pd
from sdv.tabular import CTGAN

from postprocess_code.divide import *
from postprocess_code.calculateVolume import *
from postprocess_code.calculateDensityScreen import *


def augment(structure):
    # Augment the given structure
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
    return structures



def generate(i):
    # Generate crystal structures for drugs
    # Constants
    CRITERION_DENSITY = 0.05
    CRITERION_EDGE = 1
    CRITERION_ANGLE = 5
    N_GENERATED = 1000
    N_SAMPLES = 30000

    # Pathes
    SOURCE_PATH = '../../data/polymorph/drugs_polymorph_compounds.csv'
    MODEL_PATH = '../../models/OCGAN/crystal_model_epochs400.pkl'
    RESULT_FILE_PATH = '../../results/polymorph/'

    # Read drugs data and CTGAN model
    drugs = pd.read_csv(SOURCE_PATH, encoding="ISO-8859-1", engine='python')
    model = CTGAN.load(MODEL_PATH)

    # Get the drug for which to generate crystal structures
    drug = drugs.iloc[i, :]
    # generate structures
    j = 0
    temp_new_samples = pd.DataFrame()
    temp_structures = pd.DataFrame()
    result = pd.DataFrame()
    while True:
        j += 1
        print('#%d drug: %d attempt(s) at crystal strcuture generation.' % ((i+1), j))
        # generation condition
        condition_dic = {
                        'Molecular weight_category': drug['Molecular weight_category'],
                        'S_L_category': drug['S_L_category'],
                        'Dipole moment_category': drug['Dipole moment_category'],
                        }

        print('Sampling...')
        try:
            new_samples = model.sample(
                                        N_SAMPLES, 
                                        conditions=condition_dic, 
                                        max_rows_multiplier=100, 
                                        max_retries=10, 
                                        graceful_reject_sampling=True
                                        )
        except:
            print('Cannot generate samples for the drug.')
            break
        print('Conditionally generated samples:', new_samples.shape[0])

        # Integer edge lengths and angle angles
        print('Integer')
        new_samples[['A', 'B', 'C']] = new_samples[['A', 'B', 'C']].round(2)
        new_samples[['Alpha', 'Beta', 'Gamma']] = new_samples[['Alpha', 'Beta', 'Gamma']].round(2)
        new_samples.loc[new_samples['Alpha_90']==True, 'Alpha'] = 90
        new_samples.loc[new_samples['Beta_90']==True, 'Beta'] = 90
        new_samples.loc[new_samples['Gamma_90']==True, 'Gamma'] = 90

        # Filter non-crystallographic structures
        print('Filter the structures that do not meet the crystallographic standards')
        new_samples = new_samples[new_samples.apply(lambda x: 
                ((x['Space group number'] in [2, 1]) & 
                    (x['A'] != x['B']) & (x['A'] != x['C']) & (x['B'] != x['C']) & 
                    (x['Alpha'] != 90) & (x['Beta'] != 90) & (x['Gamma'] != 90) & 
                    (x['Alpha'] != x['Beta']) & (x['Alpha'] != x['Gamma']) & (x['Beta'] != x['Gamma']) & 
                    ((x['Alpha'] + x['Beta'] + x['Gamma']) < 360))
                | ((x['Space group number'] in [14, 4, 15, 9, 5]) & 
                    (x['A'] != x['B']) & (x['A'] != x['C']) & (x['B'] != x['C']) & 
                    (x['Alpha'] == 90) & (x['Gamma'] == 90) & (x['Beta'] != 90) & 
                    ((x['Alpha'] + x['Beta'] + x['Gamma']) < 360))
                | ((x['Space group number'] in [19, 61, 33, 29]) & 
                    (x['A'] != x['B']) & (x['A'] != x['C']) & (x['B'] != x['C']) & 
                    (x['Alpha'] == 90) & (x['Beta'] == 90) & (x['Gamma'] == 90)), 
                axis=1)]

        # Calculate volume
        print('Calculate volume')
        new_samples['Calculated volume'] = new_samples.apply(
          lambda x: triclinic(x['A'], x['B'], x['C'], x['Alpha'], x['Beta'], x['Gamma']) 
            if x['Space group number'] in [2, 1]
          else monoclinic(x['A'], x['B'], x['C'], x['Beta']) 
            if x['Space group number'] in [14, 4, 15, 9, 5]
          else orthorhombic(x['A'], x['B'], x['C']) 
            if x['Space group number'] in [19, 61, 33, 29]
          else 10000000, 
          axis=1).round(2)

        # Calculate density
        print('Calculate density')
        new_samples['Calculated density'] = new_samples.apply(lambda x: density(x['Z value'], drug['Molecular weight'], Avogadro, x['Calculated volume']), axis=1).round(3)
        
        # Screen by density
        print('Screen the structures that do not meet the predicted unit cell density')
        new_samples = new_samples[new_samples.apply(lambda x: (abs(x['Calculated density'] - drug['Predicted density']) <= CRITERION_DENSITY), axis=1)]
        print('Finally screened samples:', new_samples.shape[0])

        # Cluster the similar structures
        print('Cluster the similar structures')
        start = temp_new_samples.shape[0]
        for index, row in new_samples.iterrows():
            # check if temp_new_samples is empty
            if temp_new_samples.empty:
                temp_new_samples = pd.concat([temp_new_samples, row.to_frame().T], axis=0)
                temp_structures = pd.concat([temp_structures, augment(row)], axis=0)
                continue
            else:
                # check for matches
                matched_s = temp_structures[
                    temp_structures.apply(lambda x: 
                        ((x['Space group number'] == row['Space group number']) & 
                        (x['Z value'] == row['Z value']) & 
                        (abs(x['A'] - row['A']) <= CRITERION_EDGE) & 
                        (abs(x['B'] - row['B']) <= CRITERION_EDGE) & 
                        (abs(x['C'] - row['C']) <= CRITERION_EDGE) & 
                        (abs(x['Alpha'] - row['Alpha']) <= CRITERION_ANGLE) & 
                        (abs(x['Beta'] - row['Beta']) <= CRITERION_ANGLE) & 
                        (abs(x['Gamma'] - row['Gamma']) <= CRITERION_ANGLE)), 
                    axis=1
                )]

                # if matches found, continue
                if not matched_s.empty:
                    continue
                # if no matches found, concatenate and augment
                else:
                    temp_new_samples = pd.concat([temp_new_samples, row.to_frame().T], axis=0)
                    temp_structures = pd.concat([temp_structures, augment(row)], axis=0)
                    continue
        else:
            stop = temp_new_samples.shape[0]
        print('Finally cluster samples:', (stop-start))

        # Concatenate
        print('Current generate samples:', temp_new_samples.shape[0])
        if temp_new_samples.shape[0] >= N_GENERATED or j >= 100:
            if temp_new_samples.shape[0] >= N_GENERATED:
                result = temp_new_samples.iloc[:N_GENERATED, :].reset_index(drop=True)
            else:
                result = temp_new_samples.reset_index(drop=True)
            # Output
            print('#%d drug: %d crystal structures generated.' % ((i+1), result.shape[0]))
            print(result)
            pd.DataFrame(result).to_csv(RESULT_FILE_PATH + '%d_generated.csv' % (i+1), index=False, na_rep='None')
            break

def main():
    # Constants
    N_DRUGS = 159 # 159 polymorph drugs

    pool = Pool(processes=8)
    pool.map(generate, tqdm.tqdm([i for i in range(N_DRUGS)]))
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
