import logging
import nltk

log = logging.getLogger('Week1')

if not hasattr(nltk.corpus, 'stopwords'):
    log.info('downloading corpus ...')
    nltk.download('stopwords')
    log.info('downloading corpus done')
