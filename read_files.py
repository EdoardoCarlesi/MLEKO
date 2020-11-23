"""
    MLEKO
    Machine Learning Ecosystem for KOsmology

    (C) Edoardo Carlesi 2020
    https://github.com/EdoardoCarlesi/MLEKO
"""

import pandas as pd
import numpy as np


def read_lg_rs_fullbox(file_base='/home/edoardo/CLUES/PyRCODIO/output/lg_fullbox_rs', lgf_data=False, lgf_hires_data=False, files=[0, 20], TA=False):
    """
        Read all the fullbox LG data for each box, without VWeb information
    """

    all_data = []
    for i in range(files[0], files[1]):
        this_number = '%04d' % i
        this_data_file = file_base + '_' + this_number + '.csv'

        try:
            this_data = pd.read_csv(this_data_file)
            all_data.append(this_data)
        except:
            'Skip this data'

    data = pd.concat(all_data) 
    return data


def read_lg_lgf(TA=False):
    """
        Read all the LGF / Hestia simulation LG data
    """

    if TA == True:
        data_file = '/home/edoardo/CLUES/PyRCODIO/output/lg_pairs_512_TA.csv'
    else:
        data_file = '/home/edoardo/CLUES/PyRCODIO/output/lg_pairs_512.csv'
    data = pd.read_csv(data_file)

    return data


def read_lg_fullbox(file_base='/home/edoardo/CLUES/PyRCODIO/output/lg_fullbox', TA=False, radii=False):
    """
        Read all the fullbox LG data for each box, without VWeb information
    """

    if TA == True:
        data_ta = file_base + '_TA.csv'
        data = pd.read_csv(data_ta)

    else:
        if radii == True:
            extension = '_radii.csv'
        else:
            extension = '.csv'

        data_00 = file_base + '_00' + extension
        train_00 = pd.read_csv(data_00)
        data_01 = file_base + '_01' + extension
        train_01 = pd.read_csv(data_01)
        data_02 = file_base + '_02' + extension
        train_02 = pd.read_csv(data_02)
        data_03 = file_base + '_03' + extension
        train_03 = pd.read_csv(data_03)
        data_04 = file_base + '_04' + extension
        train_04 = pd.read_csv(data_04)

        data = pd.concat([train_00, train_01, train_02, train_03, train_04])
    
    return data


def read_lg_vweb(grid_size=64, file_base='/home/edoardo/CLUES/PyRCODIO/output/lg_fullbox_vweb_'):
    """
        Read vweb ONLY at the LG position
    """
    
    grid = '%03d' % grid_size
    file_base = file_base + grid

    data_00 = file_base + '_00.csv'
    train_00 = pd.read_csv(data_00)
    data_01 = file_base + '_01.csv'
    train_01 = pd.read_csv(data_01)
    data_02 = file_base + '_02.csv'
    train_02 = pd.read_csv(data_02)
    data_03 = file_base + '_03.csv'
    train_03 = pd.read_csv(data_03)
    data_04 = file_base + '_04.csv'
    train_04 = pd.read_csv(data_04)

    data = pd.concat([train_00, train_01, train_02, train_03, train_04])

    return data


def read_lg_fullbox_vweb(grids = [64], TA=False):
    """
        Read both vweb and lg data, concatenate the sets   
    """

    # First read the full data for each LG
    data = read_lg_fullbox(TA=TA)

    # Read in the cosmic web at different scales
    for grid in grids:
        this_data_web = read_lg_vweb(grid_size = grid)

        # Read all the columns, skip the first two that only contain IDs
        these_cols = this_data_web.columns[2:]

        for col in these_cols:
            new_col = col + '_' + str(grid)
            data[new_col] = this_data_web[col]

    return data



