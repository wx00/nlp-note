from enum import Enum
import logging
import platform

log = logging.getLogger('Week2')

if platform.system() == 'Windows':
    raw_base = "D:/Dataset/icwb2-data/"
    feat_base = "D:/Dataset/icwb2/"
else:
    raw_base = '~/data/icwb2-data/'
    feat_base = '~/data/icwb2/'


class Tag(Enum):
    FIN = 0
    BEG = 1
    MID = 2
    END = 3
