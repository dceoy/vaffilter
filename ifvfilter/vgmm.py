#!/usr/bin/env python
"""Infrequent Variant Filter for VCF
https://github.com/dceoy/ifvfilter
"""

import logging
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pylab import rcParams
from sklearn.mixture import GaussianMixture


class VariantGMMFilter(object):
    def __init__(self, af_cutoff=0.01, dp_cutoff=100, target_filters=None,
                 id='VGMM'):
        self.__logger = logging.getLogger(__name__)
        self.__af_co = af_cutoff
        self.__dp_co = dp_cutoff
        if not target_filters:
            self.__target_filters = None
        elif isinstance(target_filters, str):
            self.__target_filters = {target_filters}
        else:
            self.__target_filters = set(target_filters)
        self.__target_filters = target_filters
        self.__id = id

    def run(self, vcfdf, out_fig_pdf_path=None, covariance_type='full',
            peakout_iter=5):
        vcf_cols = vcfdf.df.columns
        axes = ['AF', 'DP', 'INSLEN', 'DELLEN']
        df_xvcf = vcfdf.expanded_df(
            df=(
                vcfdf.df.pipe(
                    lambda d: d[d['FILTER'].isin(self.__target_filters)]
                ) if self.__target_filters else vcfdf.df
            ),
            by_info=True, by_samples=False, drop=False
        ).assign(
            INDELLEN=lambda d: (d['ALT'].apply(len) - d['REF'].apply(len))
        ).assign(
            AF=lambda d: d['INFO_AF'].astype(float),
            DP=lambda d: d['INFO_DP'].astype(int),
            INSLEN=lambda d: d['INDELLEN'].clip(lower=0),
            DELLEN=lambda d: (-d['INDELLEN']).clip(lower=0)
        )
        self.__logger.debug('df_xvcf:{0}{1}'.format(os.linesep, df_xvcf))
        rvn = ReversibleNormalizer(
            df=df_xvcf,
            columns=df_xvcf[axes].pipe(
                lambda d: d.columns[d.nunique() > 1].tolist()
            )
        )
        x_train = rvn.normalized_df[rvn.columns]
        self.__logger.debug('x_train:{0}{1}'.format(os.linesep, x_train))
        best_gmm_dict = dict()
        for k in range(1, x_train.shape[0]):
            gmm = GaussianMixture(
                n_components=k, covariance_type=covariance_type
            )
            gmm.fit(X=x_train)
            bic = gmm.bic(X=x_train)
            self.__logger.debug('k: {0}, bic: {1}'.format(k, bic))
            if not best_gmm_dict or bic < best_gmm_dict['bic']:
                best_gmm_dict = {'k': k, 'bic': bic, 'gmm': gmm}
            elif k >= (best_gmm_dict['k'] + peakout_iter):
                break
        best_gmm = best_gmm_dict['gmm']
        self.__logger.debug('best_gmm:{0}{1}'.format(os.linesep, best_gmm))
        df_gmm_mu = rvn.denormalize(
            df=pd.DataFrame(best_gmm.means_, columns=x_train.columns)
        ).assign(
            **{c: df_xvcf[c][0] for c in axes if c not in x_train.columns}
        )[axes]
        self.__logger.debug('df_gmm_mu:{0}{1}'.format(os.linesep, df_gmm_mu))
        df_cl = rvn.df.assign(
            CL_INT=best_gmm.predict(X=x_train)
        ).merge(
            df_gmm_mu.reset_index().rename(
                columns={'index': 'CL_INT', **{k: ('CL_' + k) for k in axes}}
            ),
            on='CL_INT', how='left'
        ).assign(
            CL_FILTER=lambda d: np.where(
                ((d['CL_AF'] < self.__af_co) | (d['CL_DP'] < self.__dp_co)),
                self.__id, d['FILTER']
            )
        )
        if out_fig_pdf_path:
            self._draw_fig(df=df_cl, out_fig_path=out_fig_pdf_path)
        vcfdf.df = vcfdf.df.join(
            df_cl[['CL_FILTER']], how='left'
        ).assign(
            FILTER=lambda d: d['FILTER'].mask(
                d['CL_FILTER'] == self.__id, d['FILTER'] + ';' + self.__id
            ).str.replace(r'PASS;', '')
        )[vcf_cols]
        self.__logger.info(
            'VariantGMMFilter filtered out variants: {0} / {1}'.format(
                (
                    df_cl['CL_FILTER'].value_counts().to_dict().get(self.__id)
                    or 0
                ),
                df_xvcf.shape[0]
            )
        )
        return vcfdf

    def _draw_fig(self, df, out_fig_path):
        self.__logger.info('Draw a fig: {}'.format(out_fig_path))
        rcParams['figure.figsize'] = (14, 10)
        sns.set(style='ticks', color_codes=True)
        sns.set_context('paper')
        cl_labs = ['CL_AF', 'CL_DP', 'CL_INSLEN', 'CL_DELLEN']
        fig_lab_names = {
            'AF': 'Allele frequency (AF)', 'DP': 'Total read depth (DP)',
            'CL': 'Cluster [{}]'.format(', '.join(cl_labs).replace('CL_', '')),
            'VT': 'Variant Type'
        }
        sns.set_palette(
            palette='GnBu_d', n_colors=df[cl_labs].drop_duplicates().shape[0]
        )
        df_fig = df.sort_values(['INSLEN', 'DELLEN']).sort_values(
            'CL_AF', ascending=False
        ).assign(
            CL=lambda d: d[cl_labs].apply(
                lambda r: '[{0:.3f}, {1:.0f}, {2:.3f}. {3:.3f}]'.format(*r),
                axis=1
            ),
            VT=lambda d: np.where(
                d['INSLEN'] > d['DELLEN'], 'Insertion',
                np.where(d['DELLEN'] > d['INSLEN'], 'Deletion', 'Substitution')
            )
        ).rename(columns=fig_lab_names)[fig_lab_names.values()]
        self.__logger.debug('df_fig:{0}{1}'.format(os.linesep, df_fig))
        _ = sns.scatterplot(
            x=fig_lab_names['DP'], y=fig_lab_names['AF'],
            style=fig_lab_names['VT'], hue=fig_lab_names['CL'], data=df_fig,
            style_order=['Substitution', 'Deletion', 'Insertion'],
            alpha=0.8, edgecolor='none', legend='full'
        )
        plt.title('Variant GMM Clusters')
        plt.savefig(out_fig_path)


class ReversibleNormalizer(object):
    def __init__(self, df, columns=None):
        self.df = df
        self.columns = columns or self.df.columns.tolist()
        self.mean_dict = self.df[self.columns].mean(axis=0).to_dict()
        self.std_dict = self.df[self.columns].std(axis=0).to_dict()
        self.normalized_df = self.normalize(df=self.df)

    def normalize(self, df):
        np.seterr(divide='ignore')
        return df.pipe(
            lambda d: d[[c for c in d.columns if c not in self.columns]]
        ).join(
            pd.DataFrame([
                {
                    'index': id,
                    **{
                        k: np.divide((v - self.mean_dict[k]), self.std_dict[k])
                        for k, v in row.items()
                    }
                } for id, row in df[self.columns].iterrows()
            ]).set_index('index'),
            how='left'
        )[df.columns]

    def denormalize(self, df):
        return df.pipe(
            lambda d: d[[c for c in d.columns if c not in self.columns]]
        ).join(
            pd.DataFrame([
                {
                    'index': id,
                    **{
                        k: ((v * self.std_dict[k]) + self.mean_dict[k])
                        for k, v in row.items()
                    }
                } for id, row in df[self.columns].iterrows()
            ]).set_index('index'),
            how='left'
        )[df.columns]
