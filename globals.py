"""
Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
from ConfigParser import SafeConfigParser
import logging

logger = logging.getLogger(__name__)

config = None

def read_configuration(configfile):
    """Read configuration and set variables.

    :return:
    """
    global config
    # if config:
    #     return config
    logger.info("Reading configuration from: " + configfile)
    parser = SafeConfigParser()
    parser.read(configfile)
    config = parser
    return parser
