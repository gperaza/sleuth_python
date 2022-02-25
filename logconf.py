import logging
import logging.config

# Setup Logging
log_dict = {
    'version': 1,
    'formatters': {
        'simple': {
            'format': ('%(asctime)s - %(levelname)s - %(message)s')
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
            'filename': 'log',
            'mode': 'w'
        }
    },
    'loggers': {
        'default': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': 'no',
        }
    },
    'root': {
        'level': 'DEBUG',
    }
}
logging.config.dictConfig(log_dict)

logger = logging.getLogger('default')
