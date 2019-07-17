#!/usr/bin/env python

import logging
import signal

from ..util.vcfdataframe import VcfDataFrame


def infreqent_variant_filter(in_vcf_path, out_vcf_path):
    logger = logging.getLogger(__name__)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    logger.info('Load a VCF file: {}'.format(in_vcf_path))
    vcfdf = VcfDataFrame(path=in_vcf_path)
    default_cols = vcfdf.df.columns
    df_xvcf = vcfdf.expanded_df(by_info=True, by_samples=False, drop=False)
    vcfdf.df = df_xvcf[default_cols]
    logger.info('Write a VCF file: {}'.format(out_vcf_path))
    vcfdf.output_table(path=out_vcf_path)
