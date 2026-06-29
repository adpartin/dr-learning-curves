from post_process import *
fpath = Path(__file__).resolve().parent
print('Current path:', fpath)

# Settings
dpi = 300

# Plot params
legend_fontsize = 12
met = 'mean_absolute_error'
t_set = 'te'
xtick_scale = 'log2'
ytick_scale = 'log2'

# Outpath
outpath = fpath/'../lc.fits/glimpse'
os.makedirs(outpath, exist_ok=True)

# LC_raw
dpath = fpath/'../lc.raw'  # LC_raw pathfile
fname = 'all_scores.csv'   # main dir to LC_raw

# Datasets
source = ['GDSC1', 'GDSC2', 'CTRP', 'NCI60']


if __name__ == '__main__':

    for src in source:
        print(f'Processing dataset: {src}')

        # dgb_path = Path(dpath, datasets[src]['dgb_path'])
        # hgb_path = Path(dpath, datasets[src]['hgb_path'])
        # snn_path = Path(dpath, datasets[src]['snn_path'])
        # mnn_path = Path(dpath, datasets[src]['mnn_path'])

        model = 'dGBDT'
        dgb_path = Path(dpath, f'lc.{src.lower()}.dGBDT', fname)

        # --------------------------------
        # Load data
        # --------------------------------
        # dgb = load_data(dgb_path, tr_set='te');
        # hgb = load_data(hgb_path, tr_set='te');
        # snn = load_data(snn_path, tr_set='te');
        # mnn = load_data(mnn_path, tr_set='te');

        data = load_data(dgb_path, tr_set='te');

        # --------------------------------
        # Plot LC_raw each source
        # --------------------------------
        kwargs = {'metric_name': met,
                  'tr_set': t_set,
                  'xtick_scale': xtick_scale,
                  'ytick_scale': ytick_scale,
                  'plot_median': True,
                  'plot_shadow': False}

        # Plot dGBDT (GDSC1)
        # -------------------------------
        kwargs.update({'title': f'{src.upper()}'})
        ax = lc_plots.plot_lc_single_metric(data, **kwargs);
        ax.legend(frameon=True, fontsize=legend_fontsize, loc='best')
        ax.grid(False)

        plt.savefig(outpath/f'lc_raw_{model}_{src}.png', dpi=dpi)


        # Plot median on linear scale
        ax = None
        dfit_all = fit_data(data, x_fit_mn=0, x_fit_mx=None, method='binomial')
        x_all = dfit_all['tr_size'].values
        y_all = dfit_all['y'].values
        color = 'b'
        pnts_args_all = {'metric_name': met,
                         'xtick_scale': 'log2',
                         'ytick_scale': 'log2',
                         'alpha': 0.9,
                         'ls': '', 'marker': '.'}

        ax = lc_plots.plot_lc(x=x_all, y=y_all, ax=ax, **pnts_args_all, color=color)

        ax.set_title(f'{src.upper()}')
        ax.grid(False)
        plt.savefig(outpath/f'median_{model}_{src}_loglog.png', dpi=dpi)


        # Plot median on log scale
        ax = None
        dfit_all = fit_data(data, x_fit_mn=0, x_fit_mx=None, method='binomial')
        x_all = dfit_all['tr_size'].values
        y_all = dfit_all['y'].values
        color = 'b'
        pnts_args_all = {'metric_name': met,
                         'xtick_scale': 'linear',
                         'ytick_scale': 'linear',
                         'alpha': 0.9,
                         'ls': '', 'marker': '.'}

        ax = lc_plots.plot_lc(x=x_all, y=y_all, ax=ax, **pnts_args_all, color=color)

        ax.set_title(f'{src.upper()}')
        ax.grid(False)
        plt.savefig(outpath/f'median_{model}_{src}_linlin.png', dpi=dpi)
