#!/usr/bin/env python
"""
GMM-based Infrequent Variant Filter for VCF data

Usage:
    vgmmfilter [--debug|--info] [--altdp-cutoff=<int>] [--target-pass]
               [--fig-pdf=<path>] <src> [<dst>]
    vgmmfilter --version
    vgmmfilter -h|--help

Options:
    --debug, --info       Execute a command with debug|info messages
    --altdp-cutoff=<int>  Set ALT depth cutoff for GMM clusters [default: 10]
    --target-pass         Target only passing variants in a VCF file
    --fig-pdf=<path>      Write a figure into a PDF file
    --version             Print version and exit
    -h, --help            Print help and exit

Arguments:
    <src>                 Path to an input file
    <dst>                 Path to an output file
"""

import logging
import os
import signal

from docopt import docopt
from pdbio.vcfdataframe import VcfDataFrame

from . import __version__
from .vgmm import VariantGMMFilter


def main():
    args = docopt(__doc__, version='vgmmfilter {}'.format(__version__))
    _set_log_config(debug=args['--debug'], info=args['--info'])
    logger = logging.getLogger(__name__)
    logger.debug('args:{0}{1}'.format(os.linesep, args))
    _vgmm_filter(
        in_vcf_path=args['<src>'], out_vcf_path=args['<dst>'],
        out_fig_pdf_path=args['--fig-pdf'],
        altdp_cutoff=args['--altdp-cutoff'],
        target_pass=args['--target-pass']
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


def _vgmm_filter(in_vcf_path, out_vcf_path, out_fig_pdf_path=None,
                 altdp_cutoff=10, target_pass=False):
    logger = logging.getLogger(__name__)
    logger.info('Execute VariantGMMFilter: {}'.format(in_vcf_path))
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    vcfdf = VcfDataFrame(path=in_vcf_path)
    logger.debug('Execute Variant GMM Filter.')
    vgmmf = VariantGMMFilter(
        altdp_cutoff=float(altdp_cutoff),
        target_filtered_variants=('PASS' if target_pass else None)
    )
    vcfdf = vgmmf.run(vcfdf=vcfdf, out_fig_pdf_path=out_fig_pdf_path)
    logger.info('Write a VCF file: {}'.format(out_vcf_path))
    vcfdf.output_table(path=out_vcf_path)
