'''
@module     qcommodel.logging.__init__.py
@info       Logging module wrapper to instantiate and share common logger
            settings across all subclasses
'''

# Python libraries
import logging
import logging.config
import sys

# Setup logger with desired settings
logging.basicConfig()
_logger = logging.getLogger('qualcomm')

# Disable sklearn complaints about dataframes in models
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

## Helper class for stdout/err redirects
class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.strip())

    def flush(self):
        pass

def Logger(**kwargs):
    ''' Logger instantiation function to abstract away the actual module
    :param redirect: Boolean flag for redirecting stdout/err to log
    :param name: String name of the logger to be used as a child in the log
    :return logging.Logger: Logger class instance
    '''
    # Get the parent logger
    result = _logger
    
    # Name the child logging module if desired
    if 'name' in kwargs:
        result = _logger.getChild(kwargs['name'])
        del kwargs['name']

    # Pass through the rest of the logging settings
    logging.config.dictConfig(kwargs)

    # Redirect handling
    if 'redirect' in kwargs:
        assert isinstance(kwargs['redirect'], bool), f'Redirect flag must be of type bool'
        setRedirect(result, kwargs['redirect'])
    result.debug('Logger initialized')
    return result


## Configuration functions
def setRedirect(logger, enable: bool):
    ''' stdout/err redirection
    :param logger: The logger to redirect to
    :param enable: Boolean on/off flag for stdout/err redirection
    '''
    if enable:
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)
    else:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__