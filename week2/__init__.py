from enum import Enum
import os
import logging
import platform

log = logging.getLogger('Week2')

if platform.system() == 'Windows':
    han_base = "D:/Dataset/Unihan"
    raw_base = "D:/Dataset/icwb2-data"
    feat_base = "D:/Dataset/icwb2"
else:
    home = os.path.expanduser('~')
    han_base = f'{home}/data/Unihan'
    raw_base = f'{home}/data/icwb2-data'
    feat_base = f'{home}/data/icwb2'


class Tag(Enum):
    FIN = 0
    BEG = 1
    MID = 2
    END = 3
