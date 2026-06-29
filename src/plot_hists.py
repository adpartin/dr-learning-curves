from post_process import *
fpath = Path(__file__).resolve().parent
print('Current path:', fpath)

# Settings
dpi = 300

# outpath = fpath/f'histograms'
outpath = fpath/'../lc.fits/histograms'
os.makedirs(outpath, exist_ok=True)


maindir = fpath/'../data/ml.dfs/July2020'

# Datasets
source = ['GDSC1', 'GDSC2', 'CTRP', 'NCI60']


if __name__ == '__main__':

    for src in source:
        print(f'Processing dataset: {src}')

        if src.lower() != 'nci60':
            dpath = maindir/f'data.{src.lower()}.dd.ge/data.{src.lower()}.dd.ge.parquet'
        else:
            dpath = maindir/f'data.{src.lower()}.dd.ge.random/data.{src.lower()}.dd.ge.parquet'
                    
        df = pd.read_parquet(dpath)

        score_name = 'AUC'
        fig, ax = plt.subplots()
        ax.hist(df[score_name], bins=150, facecolor='b', alpha=0.7);
        plt.grid(False)
        # plt.legend(frameon=False, shadow=False, loc='best', framealpha=0.5)
        plt.xlabel('Dose-independent Drug Response (AUC) Value')
        plt.ylabel('Count')
        plt.title(f'{src}');
        plt.tight_layout()
        plt.savefig(outpath/f'hist_{src}.png', dpi=dpi)
