#!/usr/bin/env python
"""
Infrequent Variant Filter for VCF files

Usage:
    ifvfilter [--debug|--info] <src> [<dst>]
    ifvfilter --version
    ifvfilter -h|--help

Options:
    --debug, --info     Execute a command with debug|info messages
    --version           Print version and exit
    -h, --help          Print help and exit

Arguments:
    <src>               Path to an input file
    <dst>               Path to an output file
"""

import logging
import os

from docopt import docopt

from . import __version__
from .ifvfilter import infreqent_variant_filter


def main():
    args = docopt(__doc__, version='ifvfilter {}'.format(__version__))
    _set_log_config(debug=args['--debug'], info=args['--info'])
    logger = logging.getLogger(__name__)
    logger.debug('args:{0}{1}'.format(os.linesep, args))
    infreqent_variant_filter(
        in_vcf_path=args['<src>'], out_vcf_path=args['<dst>']
    )


def _set_log_config(debug=None, info=None):
    if debug:
        lv = logging.DEBUG
    elif info:
        lv = logging.INFO
    else:
        lv = logging.WARNING
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=lv
    )
