#!/usr/bin/env python
"""GMM-based Infrequent Variant Filter for VCF data
https://github.com/dceoy/vgmmfilter
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pylab import rcParams
from sklearn.mixture import GaussianMixture


class VariantGMMFilter(object):
    def __init__(self, af_cutoff=0.02, altdp_cutoff=10, alpha_for_mvalue=1e-2,
                 target_filtered_variants=None, filter_label='VGMM'):
        self.__logger = logging.getLogger(__name__)
        self.__cos = {'AF': af_cutoff, 'ALTDP': altdp_cutoff}
        self.__logger.debug('self.__cos: {}'.format(self.__cos))
        self.__a4m = alpha_for_mvalue
        if not target_filtered_variants:
            self.__tfv = None
        elif isinstance(target_filtered_variants, str):
            self.__tfv = {target_filtered_variants}
        else:
            assert isinstance(target_filtered_variants, (list, tuple, set))
            self.__tfv = set(target_filtered_variants)
        self.__fl = filter_label

    def run(self, vcfdf, out_fig_pdf_path=None, covariance_type='full',
            peakout_iter=5):
        self._validate_df_vcf(df=vcfdf.df)
        df_vcf = (
            vcfdf.df.pipe(lambda d: d[d['FILTER'].isin(self.__tfv)])
            if self.__tfv else vcfdf.df
        )
        if df_vcf.size:
            df_cl = self._cluster_variants(
                df_xvcf=vcfdf.expanded_df(
                    df=df_vcf, by_info=True, by_samples=False, drop=False
                ),
                covariance_type=covariance_type, peakout_iter=peakout_iter
            )
            vcf_cols = vcfdf.df.columns.tolist()
            vcfdf.df = vcfdf.df.merge(
                df_cl[[*vcf_cols, 'CL_FILTER']], on=vcf_cols, how='left'
            ).assign(
                FILTER=lambda d:
                np.where(d['CL_FILTER'].isna(), d['FILTER'], d['CL_FILTER'])
            )[vcf_cols]
            self.__logger.info(
                'VariantGMMFilter filtered out variants: {0} / {1}'.format(
                    (
                        df_cl['CL_FILTER'].value_counts().to_dict().get(
                            self.__fl
                        ) or 0
                    ),
                    df_vcf.shape[0]
                )
            )
            if out_fig_pdf_path:
                self._draw_fig(df=df_cl, out_fig_path=out_fig_pdf_path)
            else:
                pass
        else:
            self.__logger.info('No variant targeted for {}.'.format(self.__fl))
        return vcfdf

    @staticmethod
    def _validate_df_vcf(df):
        ra = df[['REF', 'ALT']].apply(lambda r: ''.join(r), axis=1)
        if ra[ra.str.contains(',')].size:
            raise ValueError('multiple allele pattern is not supported.')
        elif ra[ra.str.contains(r'[^a-zA-Z]')].size:
            raise ValueError('invalid allele pattern')

    def _cluster_variants(self, df_xvcf, covariance_type='full', tol=1e-4,
                          max_iter=1000, peakout_iter=5):
        axes = ['M_AF', 'LOG2_DP', 'LOG2_INSLEN', 'LOG2_DELLEN']
        df_x = df_xvcf.assign(
            AF=lambda d: d['INFO_AF'].astype(float),
            DP=lambda d: d['INFO_DP'].astype(int),
            INDELLEN=lambda d: (d['ALT'].apply(len) - d['REF'].apply(len))
        ).assign(
            ALTDP=lambda d: (d['AF'] * d['DP']),
            INSLEN=lambda d: d['INDELLEN'].clip(lower=0),
            DELLEN=lambda d: (-d['INDELLEN']).clip(lower=0)
        ).assign(
            M_AF=lambda d: self._af2mvalue(
                af=d['AF'], dp=d['DP'], alpha=self.__a4m
            ),
            LOG2_DP=lambda d: np.log2(d['DP'] + 1),
            LOG2_INSLEN=lambda d: np.log2(d['INSLEN'] + 1),
            LOG2_DELLEN=lambda d: np.log2(d['DELLEN'] + 1)
        )
        self.__logger.debug('df_x:{0}{1}'.format(os.linesep, df_x))
        rvn = ReversibleNormalizer(
            df=df_x,
            columns=df_x[axes].pipe(
                lambda d: d.columns[d.nunique() > 1].tolist()
            )
        )
        x_train = rvn.normalized_df[rvn.columns]
        self.__logger.debug('x_train:{0}{1}'.format(os.linesep, x_train))
        best_gmm_dict = dict()
        for k in range(1, x_train.shape[0]):
            gmm = GaussianMixture(
                n_components=k, covariance_type=covariance_type, tol=tol,
                max_iter=max_iter
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
            **{c: df_x[c].iloc[0] for c in axes if c not in x_train.columns}
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
            CL_DP=lambda d: (np.exp2(d['CL_LOG2_DP']) - 1),
            CL_INSLEN=lambda d: (np.exp2(d['CL_LOG2_INSLEN']) - 1),
            CL_DELLEN=lambda d: (np.exp2(d['CL_LOG2_DELLEN']) - 1)
        ).assign(
            CL_AF=lambda d: self._mvalue2af(
                mvalue=d['CL_M_AF'], dp=d['CL_DP'], alpha=self.__a4m
            )
        ).assign(
            CL_ALTDP=lambda d: (d['CL_AF'] * d['CL_DP']),
        ).assign(
            CL_FILTER=lambda d: d['FILTER'].mask(
                (
                    (d['CL_AF'] < self.__cos['AF'])
                    | (d['CL_ALTDP'] < self.__cos['ALTDP'])
                ),
                (d['FILTER'] + ';' + self.__fl).str.replace(r'^PASS;', '')
            )
        )
        return df_cl

    @staticmethod
    def _af2mvalue(af, dp, alpha=0):
        return np.log2(np.divide((af * dp + alpha), ((1 - af) * dp + alpha)))

    @staticmethod
    def _mvalue2af(mvalue, dp, alpha=0):
        return (
            lambda x: np.divide(((x * dp) - ((x - 1) * alpha)), ((x + 1) * dp))
        )(x=np.exp2(mvalue))

    def _draw_fig(self, df, out_fig_path):
        self.__logger.info('Draw a fig: {}'.format(out_fig_path))
        rcParams['figure.figsize'] = (11.88, 8.40)  # A4 aspect: (297x210)
        sns.set(style='ticks', color_codes=True)
        sns.set_context('paper')
        df_fig = df.sort_values(
            'CL_AF', ascending=False
        ).assign(
            CL=lambda d: d[[
                'CL_ALTDP', 'CL_DP', 'CL_AF', 'CL_INSLEN', 'CL_DELLEN'
            ]].apply(
                lambda r:
                '[{0:.1f}/{1:.1f}, {2:.4f}, {3:.2f}, {4:.2f}]'.format(*r),
                axis=1
            ),
            VT=lambda d: np.where(
                d['INDELLEN'] > 0, 'Insertion',
                np.where(d['INDELLEN'] < 0, 'Deletion', 'Substitution')
            )
        )
        cl_labels = {
            k: '{0}\t(x{1})'.format(k, v)
            for k, v in df_fig['CL'].value_counts().to_dict().items()
        }
        vt_labels = {
            k: '{0}\t(x{1})'.format(k, v)
            for k, v in df_fig['VT'].value_counts().to_dict().items()
        }
        fig_lab_names = {
            'AF': 'ALT allele frequency (AF)', 'DP': 'Total read depth (DP)',
            'CL': 'Estimated cluster [ALT/DP, AF, INS, DEL]',
            'VT': 'Variant Type'
        }
        sns.set_palette(palette='GnBu_d', n_colors=df_fig['CL'].nunique())
        self.__logger.debug('df_fig:{0}{1}'.format(os.linesep, df_fig))
        ax = sns.scatterplot(
            x=fig_lab_names['DP'], y=fig_lab_names['AF'],
            style=fig_lab_names['VT'], hue=fig_lab_names['CL'],
            data=df_fig.assign(
                CL=lambda d: d['CL'].apply(lambda k: cl_labels[k]),
                VT=lambda d: d['VT'].apply(lambda k: vt_labels[k])
            ).rename(columns=fig_lab_names)[fig_lab_names.values()],
            markers={
                vt_labels[k]: v for k, v in {
                    'Substitution': '.', 'Deletion': '>', 'Insertion': '<'
                }.items() if k in vt_labels
            },
            alpha=0.8, edgecolor='none', legend='full'
        )
        ax.set_xscale('log')
        ax.set_title('Variant GMM Clusters')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
        axp = ax.get_position()
        ax.set_position([axp.x0, axp.y0, axp.width * 0.75, axp.height])
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
