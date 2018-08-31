import logging
# import nltk
import platform

log = logging.getLogger('Week1')

# nltk.download('wordnet')
# nltk.download('stopwords')

if platform.system() == 'Windows':
    raw_base = "D:/Dataset/aclImdb/"
    feat_base = "D:/Dataset/imdb/"
else:
    raw_base = '~/aclImdb/'
    feat_base = '~/imdb/'
