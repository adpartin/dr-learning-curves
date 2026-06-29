from post_process import *
fpath = Path(__file__).resolve().parent
print('Current path:', fpath)

# Settings
drop_bad_r2fit = False
# drop_bad_r2fit = True
dpi = 300

# Plot params
legend_fontsize = 12
met = 'mean_absolute_error'
t_set = 'te'
xtick_scale = 'log2'
ytick_scale = 'log2'

# Plot labels
dgb_label = 'dGBDT'
hgb_label = 'hGBDT'
snn_label = 'sNN'
mnn_label = 'mNN'

# Outpath
outpath = fpath/'../lc.fits/cmpr_models'
os.makedirs(outpath, exist_ok=True)

# LC_raw
dpath = fpath/'../lc.raw'  # LC_raw pathfile
fname = 'all_scores.csv'   # main dir to LC_raw

# Datasets
source = ['GDSC1', 'GDSC2', 'CTRP', 'NCI-60']

datasets = {
    'GDSC1': 
    {
        'snn_path': Path(dpath, 'lc.gdsc1.sNN', fname),
        'mnn_path': Path(dpath, 'lc.gdsc1.mNN', fname),
        'hgb_path': Path(dpath, 'lc.gdsc1.hGBDT', fname),
        'dgb_path': Path(dpath, 'lc.gdsc1.dGBDT', fname)
    },
    'GDSC2':
    {
        'snn_path': Path(dpath, 'lc.gdsc2.sNN', fname),
        'mnn_path': Path(dpath, 'lc.gdsc2.mNN', fname),
        'hgb_path': Path(dpath, 'lc.gdsc2.hGBDT', fname),
        'dgb_path': Path(dpath, 'lc.gdsc2.dGBDT', fname)
    },
    'CTRP':
    {
        'snn_path': Path(dpath, 'lc.ctrp.sNN', fname),
        'mnn_path': Path(dpath, 'lc.ctrp.mNN', fname),
        'hgb_path': Path(dpath, 'lc.ctrp.hGBDT', fname),
        'dgb_path': Path(dpath, 'lc.ctrp.dGBDT', fname)
    },
    'NCI60':
    {
        'snn_path': Path(dpath, 'lc.nci60.sNN', fname),
        'mnn_path': Path(dpath, 'lc.nci60.mNN', fname),
        'hgb_path': Path(dpath, 'lc.nci60.hGBDT', fname),
        'dgb_path': Path(dpath, 'lc.nci60.dGBDT', fname)
    }
}


if __name__ == '__main__':

    for src in source:
        print(f'Processing dataset: {src}')

        dgb_path = Path(dpath, datasets[src]['dgb_path'])
        hgb_path = Path(dpath, datasets[src]['hgb_path'])
        snn_path = Path(dpath, datasets[src]['snn_path'])
        mnn_path = Path(dpath, datasets[src]['mnn_path'])

        # --------------------------------
        # Load data
        # --------------------------------
        dgb = load_data(dgb_path, tr_set='te');
        hgb = load_data(hgb_path, tr_set='te');
        snn = load_data(snn_path, tr_set='te');
        mnn = load_data(mnn_path, tr_set='te');

        # Remove scores with negative R2
        if drop_bad_r2fit:
            snn = drop_bad_r2(snn)
            mnn = drop_bad_r2(mnn)
            hgb = drop_bad_r2(hgb)
            dgb = drop_bad_r2(dgb)
            
        # NCI60: remove bad samples
        if src == 'NCI-60':
            # np.unique(sorted(snn.tr_size))
            snn = snn[ ~snn['tr_size'].isin([369806, 499618, 580947, 610743, 625000]) ].reset_index(drop=True)
            mnn = mnn[ ~mnn['tr_size'].isin([369806, 499618, 580947, 610743, 625000]) ].reset_index(drop=True)


        # --------------------------------
        # Plot LC_raw each source
        # --------------------------------
        kwargs = {'metric_name': met,
                  'tr_set': t_set,
                  'xtick_scale': xtick_scale,
                  'ytick_scale': ytick_scale,
                  'plot_median': True}

#         def plot_lc_model(df, src_name, label):
#             if df is None:
#                 return None
#             kwargs.update({'title': f'{src_name}; {label}'})
#             ax = lc_plots.plot_lc_single_metric(df, **kwargs);
#             ax.legend(frameon=True, fontsize=legend_fontsize, loc='best')
#             ax.grid(False)

        plot_lc_model(df=dgb, src_name=src, label=dgb_label)
        plot_lc_model(df=hgb, src_name=src, label=hgb_label)
        plot_lc_model(df=snn, src_name=src, label=snn_label)
        plot_lc_model(df=mnn, src_name=src, label=mnn_label)


        # --------------------------------
        # Plot LC_raw combined
        # --------------------------------    
        ax = None
        kwargs = {'metric_name': met,
                  'xtick_scale': xtick_scale,
                  'ytick_scale': ytick_scale}

        # if dgb is not None:
        #     lgb = dgb
        #     lgb = lgb[ lgb.metric==met ].reset_index(drop=True)
        #     ax = lc_plots.plot_lc(x=lgb['tr_size'].values, y=lgb['score'].values,
        #                           color='m', label=f'{dgb_label}', **kwargs, ax=ax);

        if hgb is not None:
            lgb = hgb
            lgb = lgb[ lgb.metric==met ].reset_index(drop=True)
            ax = lc_plots.plot_lc(x=lgb['tr_size'].values,
                                  y=lgb['score'].values,
                                  color='g', label=f'{hgb_label}', **kwargs, ax=ax);

        if snn is not None:
            snn = snn[ snn.metric==met ].reset_index(drop=True)
            ax = lc_plots.plot_lc(x=snn['tr_size'].values,
                                  y=snn['score'].values,
                                  color='b', label=f'{snn_label}', **kwargs, ax=ax);

        if mnn is not None:
            mnn = mnn[ mnn.metric==met ].reset_index(drop=True)
            ax = lc_plots.plot_lc(x=mnn['tr_size'].values,
                                  y=mnn['score'].values,
                                  color='r', label=f'{mnn_label}', **kwargs, ax=ax);

        ax.legend(frameon=True, fontsize=legend_fontsize, loc='best');
        ax.grid(False)


        # --------------------------------
        # Determine LC_f
        # -------------------------------- 
        if src == 'GDSC1':
            x_fit_mn = 10000;
            # x_fit_mn = 20000;
            x_fit_mx = None
            # startParams = {'a': 1.2, 'b': -0.3, 'c': 0.04}

        elif src == 'GDSC2':
            x_fit_mn = 10000;
            # x_fit_mn = 20000;
            x_fit_mx = None
            # startParams = {'a': 1.2, 'b': -0.3, 'c': 0.04}
            
        elif src == 'CTRP':
            x_fit_mn = 10000;
            # x_fit_mn = 50000;
            x_fit_mx = None
            # startParams = {'a': 1.2, 'b': -0.3, 'c': 0.04}    
            
        elif src == 'NCI-60':    
            x_fit_mn = 100000;
            x_fit_mx = None
            # startParams = {'a': 1.2, 'b': -0.3, 'c': 0.04}

        # Dataframes to fit    
        dfit_lgb_dft = fit_data(dgb, x_fit_mn=x_fit_mn, x_fit_mx=x_fit_mx)
        dfit_lgb_hpo = fit_data(hgb, x_fit_mn=x_fit_mn, x_fit_mx=x_fit_mx)
        dfit_snn     = fit_data(snn, x_fit_mn=x_fit_mn, x_fit_mx=x_fit_mx)
        dfit_mnn     = fit_data(mnn, x_fit_mn=x_fit_mn, x_fit_mx=x_fit_mx)


        # --------------------------------------------
        # Percent improve compared to baseline (dGBDT)
        # --------------------------------------------
        v_dGBDT = dfit_lgb_dft.loc[:,'y'].values[-1]
        v_hGBDT = dfit_lgb_hpo.loc[:,'y'].values[-1]
        v_sNN   = dfit_snn.loc[:,'y'].values[-1]
        v_mNN   = dfit_mnn.loc[:,'y'].values[-1]

        # dGBDT
        df_lgb_dft = dfit_lgb_dft.iloc[-1:, :]
        df_lgb_dft['model'] = 'dGBDT'
        df_lgb_dft['Ds'] = np.nan

        # hGBDT
        df_lgb_hpo = dfit_lgb_hpo.iloc[-1:, :]
        df_lgb_hpo['model'] = 'hGBDT'
        df_lgb_hpo['Ds'] = np.around((v_dGBDT - v_hGBDT)/v_dGBDT * 100, 2)

        # sNN
        df_snn = dfit_snn.iloc[-1:, :]
        df_snn['model'] = 'sNN'
        df_snn['Ds'] = np.around((v_dGBDT - v_sNN)/v_dGBDT * 100, 2)

        # mNN
        df_mnn = dfit_mnn.iloc[-1:, :]
        df_mnn['model'] = 'mNN'
        df_mnn['Ds'] = np.around((v_dGBDT - v_mNN)/v_dGBDT * 100, 2)

        # Concat
        df = pd.concat([df_lgb_dft, df_lgb_hpo, df_snn, df_mnn], axis=0).reset_index(drop=True)
        df = df.drop(columns=['w'])
        df.insert(loc=0, column='dataset', value=src, allow_duplicates=True)
        df = df.rename(columns={'y': 'sK'})
        df = df[['dataset', 'tr_size', 'model', 'sK', 'Ds']]

        df_improve = df
        del df
        df_improve.to_csv(outpath/f'percent_imporve_{src}.csv', index=False)


        # -------------------------------- 
        # Fit LC_f
        # -------------------------------- 
        pnts_args = {'metric_name': met,
                     'xtick_scale': xtick_scale,
                     'ytick_scale': ytick_scale,
                     'alpha': 0.8,
                     'ls': '',
                     'marker': '.'}

        fit_args = {'metric_name': met,
                    'xtick_scale': xtick_scale,
                    'ytick_scale': ytick_scale,
                    'alpha': 0.8,
                    'ls': '--',
                    'marker': ''}

        ax = None
        fit_method = 'old'
        # fit_method = 'new'

        # dGBDT (LGBM default HPs)
        if dgb is not None:
            aa = dfit_lgb_dft
            name = dgb_label
            color = 'm'
            pnts_args.update({'marker': '>'})

            xf = aa['tr_size'].values
            yf = aa['y'].values
            
            cc_lgb_dft = None
            prms_lgb_dft = None
            
            if fit_method == 'new':
                cc = FitPwrLaw(xf=xf, yf=yf, w=aa['w'].values, **startParams)
                xf_plot, yf_plot = cc.calc_fit( x1=xf[0], x2=xf[-1] )
                cc_lgb_dft = cc
            else:
                prms_lgb_dft = fit_params(x=xf, y=yf)
                yf_plot = biased_powerlaw(xf, **prms_lgb_dft)
                xf_plot = xf
            
            gof_lgb_dft = calc_gof(yf, yf_plot)
            gof_lgb_dft['model'] = 'dGBDT'
            # Don't plot!
            # ax = lc_plots.plot_lc(x=xf, y=yf, ax=ax, **pnts_args, color=color, label=f'{name} data')
            # ax = lc_plots.plot_lc(x=xf_plot, y=yf_plot, ax=ax, **fit_args, color=color, label=f'{name} fit')

        # hGBDT (LGBM HPO)
        if hgb is not None:
            aa = dfit_lgb_hpo    
            name = hgb_label
            color = 'g'
            pnts_args.update({'marker': 'o'})

            xf = aa['tr_size'].values
            yf = aa['y'].values

            cc_lgb_hpo = None
            prms_lgb_hpo = None
            
            if fit_method == 'new':
                cc = FitPwrLaw(xf=xf, yf=yf, w=aa['w'].values, **startParams)
                # xf_plot, yf_plot = cc.calc_fit( x1=xf[0], x2=xf[-1] )
                xf_plot, yf_plot = cc.calc_fit( x=xf )
                cc_lgb_hpo = cc
            else:    
                prms_lgb_hpo = fit_params(x=xf, y=yf)
                yf_plot = biased_powerlaw(xf, **prms_lgb_hpo)
                xf_plot = xf
                
            gof_lgb_hpo = calc_gof(yf, yf_plot)
            gof_lgb_hpo['model'] = 'hGBDT'
            ax = lc_plots.plot_lc(x=xf, y=yf, ax=ax, **pnts_args, color=color, label=f'{name} ' + '$LC_{f}$ data')
            ax = lc_plots.plot_lc(x=xf_plot, y=yf_plot, ax=ax, **fit_args, color=color, label=f'{name} power-law fit') 

        # sNN (NN0)
        if snn is not None:
            aa = dfit_snn    
            name = snn_label
            color = 'b'
            pnts_args.update({'marker': 's'})
            
            xf = aa['tr_size'].values
            yf = aa['y'].values
            
            cc_snn = None
            prms_snn = None
            
            if fit_method == 'new':
                cc = FitPwrLaw(xf=xf, yf=yf, w=aa['w'].values, **startParams)  # new!
                # xf_plot, yf_plot = cc_snn.calc_fit( x1=xf[0], x2=xf[-1] )
                xf_plot, yf_plot = cc.calc_fit( x=xf )
                cc_snn = cc
            else:
                prms_snn = fit_params(x=xf, y=yf)
                yf_plot = biased_powerlaw(xf, **prms_snn)
                xf_plot = xf
                
            gof_snn = calc_gof(yf, yf_plot)
            gof_snn['model'] = 'sNN'
            ax = lc_plots.plot_lc(x=xf, y=yf, ax=ax, **pnts_args, color=color, label=f'{name} ' + '$LC_{f}$ data')
            ax = lc_plots.plot_lc(x=xf_plot, y=yf_plot, ax=ax, **fit_args, color=color, label=f'{name} power-law fit')

        # mNN (NN1)
        if mnn is not None:
            aa = dfit_mnn    
            name = mnn_label
            color = 'r'
            pnts_args.update({'marker': 'v'})
            
            xf = aa['tr_size'].values
            yf = aa['y'].values
            
            cc_mnn = None
            prms_mnn = None
            
            if fit_method == 'new':
                cc = FitPwrLaw(xf=xf, yf=yf, w=aa['w'].values, **startParams)  # new!
                # xf_plot, yf_plot = cc_mnn.calc_fit( x1=xf[0], x2=xf[-1] )
                xf_plot, yf_plot = cc.calc_fit( x=xf )
                cc_mnn = cc        
            else:
                prms_mnn = fit_params(x=xf, y=yf)
                yf_plot = biased_powerlaw(xf, **prms_mnn)
                xf_plot = xf
            
            gof_mnn = calc_gof(yf, yf_plot)
            gof_mnn['model'] = 'mNN'
            ax = lc_plots.plot_lc(x=xf, y=yf, ax=ax, **pnts_args, color=color, label=f'{name} ' + '$LC_{f}$ data')
            ax = lc_plots.plot_lc(x=xf_plot, y=yf_plot, ax=ax, **fit_args, color=color, label=f'{name} power-law fit')

        # Save figure
        ax.set_title(f'{src}')
        ax.legend(frameon=True, fontsize=12, loc='best')
        ax.grid(False)
        plt.savefig(outpath/f'{src}_fits.png', dpi=dpi)


        # --------------------------------
        # Calc gof measures
        # --------------------------------
        gof_df = pd.DataFrame([gof_lgb_dft, gof_lgb_hpo, gof_snn, gof_mnn])
        gof_df = gof_df[['model', 'rmse', 'mae', 'r2']]
        gof_df.insert(loc=0, column='dataset', value=src, allow_duplicates=True)
        gof_df.to_csv(outpath/f'gof_{src}.csv', index=False)


        # --------------------------------
        # Estimated power-law params
        # --------------------------------
        df_list = []

        if prms_lgb_dft is not None:
            prms_lgb_dft_df = pd.DataFrame([prms_lgb_dft])
            prms_lgb_dft_df['model'] = 'dGDBT'
            df_list.append(prms_lgb_dft_df)

        if prms_lgb_hpo is not None:
            prms_lgb_hpo_df = pd.DataFrame([prms_lgb_hpo])
            prms_lgb_hpo_df['model'] = 'hGDBT'
            df_list.append(prms_lgb_hpo_df)

        if prms_snn is not None:
            prms_snn_df = pd.DataFrame([prms_snn])
            prms_snn_df['model'] = 'sNN'
            df_list.append(prms_snn_df)

        if prms_mnn is not None:
            prms_mnn_df = pd.DataFrame([prms_mnn])
            prms_mnn_df['model'] = 'mNN'
            df_list.append(prms_mnn_df)

        df_prms = pd.concat(df_list, axis=0).reset_index(drop=True)
        df_prms = df_prms[['model', 'alpha', 'beta', 'gamma']]
        df_prms.insert(loc=0, column='dataset', value=src, allow_duplicates=True)
        df_prms.to_csv(outpath/f'prms_{src}.csv', index=False)


        # --------------------------------
        # Extrapolate
        # --------------------------------
        # def inv_powerlaw(y, prms):
        #     vv = ((y - prms['gamma']) / prms['alpha'] ) ** (1/prms['beta'])
        #     if np.isnan(vv) == False:
        #         vv = int(vv)
        #     return vv

        # def get_score_at_2mK(dfit, prms):
        #     x_a = 2 * dfit['tr_size'].values[-1]
        #     return biased_powerlaw(x_a, **prms)


        red_percent = 0.9  # percent reduction

        dGBDT = {}
        dGBDT['model'] = 'dGBDT'
        dGBDT['m_func_s'] = inv_powerlaw(red_percent * dfit_lgb_dft['y'].values[-1], prms_lgb_dft)
        dGBDT['s2T'] = get_score_at_2mK(dfit_lgb_dft, prms_lgb_dft)

        hGBDT = {}
        hGBDT['model'] = 'hGBDT'
        hGBDT['m_func_s'] = inv_powerlaw(red_percent * dfit_lgb_hpo['y'].values[-1], prms_lgb_hpo)
        hGBDT['s2T'] = get_score_at_2mK(dfit_lgb_hpo, prms_lgb_hpo)

        snn = {}
        snn['model'] = 'sNN'
        snn['m_func_s'] = inv_powerlaw(red_percent * dfit_snn['y'].values[-1], prms_snn)
        snn['s2T'] = get_score_at_2mK(dfit_snn, prms_snn)

        mnn = {}
        mnn['model'] = 'mNN'
        mnn['m_func_s'] = inv_powerlaw(red_percent * dfit_mnn['y'].values[-1], prms_mnn)
        mnn['s2T'] = get_score_at_2mK(dfit_mnn, prms_mnn)

        df_ext = pd.DataFrame([dGBDT, hGBDT, snn, mnn])


        dff = df_ext.merge(df_improve, on='model')
        dff = dff[['dataset', 'model', 'tr_size', 'sK', 'Ds', 's2T', 'm_func_s']]


        dff['s2T_impv'] = None
        dff['m_factor'] = None

        for m in dff['model'].values:
            # Percent improvement as compared to sK
            s_k = dff.loc[ dff['model']==m, 'sK' ]
            s_2k = dff.loc[ dff['model']==m, 's2T' ]
            vv = np.around((s_k - s_2k)/s_k * 100, 7)
            dff.loc[dff['model']==m, 's2T_impv'] = vv
            
            # The required data size as a factor of mK
            if np.isnan( dff.loc[ dff['model']==m, 'm_func_s' ] ).values[0] == False:
                dff.loc[dff['model']==m, 'm_factor'] = dff.loc[ dff['model']==m, 'm_func_s' ] / dff.loc[ dff['model']==m, 'tr_size' ]
                
        dff = dff[['dataset', 'model', 'tr_size', 'sK', 'Ds', 's2T', 's2T_impv', 'm_func_s', 'm_factor']]

        dff.to_csv(outpath/f'Table2_{src}.csv', index=False)
