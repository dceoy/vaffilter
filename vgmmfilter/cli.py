#!/usr/bin/env python
"""
GMM-based Infrequent Variant Filter for VCF data

Usage:
    vgmmfilter [--debug|--info] [--af-cutoff=<int>] [--min-salvaged-af=<float>]
               [--alpha-of-mvalue=<float>] [--target-pass] [--seed=<int>]
               [--fig-pdf=<path>] <src> [<dst>]
    vgmmfilter --version
    vgmmfilter -h|--help

Options:
    --debug, --info       Execute a command with debug|info messages
    --af-cutoff=<float>   Set AF cutoff for GMM clusters [default: 0.02]
    --min-salvaged-af=<float>
                          Salvage variants of high AF [default: 0.2]
    --alpha-of-mvalue=<float>
                          Specify alpha of M-value [default: 1]
    --target-pass         Target only passing variants in a VCF file
    --seed=<int>          Set random seed
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

import numpy as np
from docopt import docopt
from pdbio.vcfdataframe import VcfDataFrame

from . import __version__
from .vgmm import VariantGMMFilter


def main():
    args = docopt(__doc__, version='vgmmfilter {}'.format(__version__))
    _set_log_config(debug=args['--debug'], info=args['--info'])
    logger = logging.getLogger(__name__)
    logger.debug('args:{0}{1}'.format(os.linesep, args))
    if args['--seed']:
        np.random.seed(seed=int(args['--seed']))
    _vgmm_filter(
        in_vcf_path=args['<src>'], out_vcf_path=args['<dst>'],
        out_fig_pdf_path=args['--fig-pdf'], af_cutoff=args['--af-cutoff'],
        min_salvaged_af=args['--min-salvaged-af'],
        alpha_of_mvalue=args['--alpha-of-mvalue'],
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


def _vgmm_filter(in_vcf_path, out_vcf_path, out_fig_pdf_path, af_cutoff,
                 min_salvaged_af, alpha_of_mvalue, target_pass):
    logger = logging.getLogger(__name__)
    logger.info('Execute VariantGMMFilter: {}'.format(in_vcf_path))
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    vcfdf = VcfDataFrame(path=in_vcf_path)
    logger.debug('Execute Variant GMM Filter.')
    vgmmf = VariantGMMFilter(
        af_cutoff=float(af_cutoff), min_salvaged_af=float(min_salvaged_af),
        alpha_of_mvalue=float(alpha_of_mvalue),
        target_filtered_variants=('PASS' if target_pass else None)
    )
    vcfdf = vgmmf.run(vcfdf=vcfdf, out_fig_pdf_path=out_fig_pdf_path)
    logger.info('Write a VCF file: {}'.format(out_vcf_path))
    vcfdf.output_table(path=out_vcf_path)
