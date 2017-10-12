class TopK:
    def __init__(self, df, k):
        self.top_proba = {}
        for name in df['seriesuid'].unique():
            self.top_proba[name] = sorted(df.loc[df['seriesuid']==name,'probability'])[-k:]
        
    def __call__(self,row):
        return row['probability'] in self.top_proba[row['seriesuid']]
        
df[df.apply(TopK(df,3),axis=1)]
# axis : {0 or 'index', 1 or 'columns'}, default 0
#    * 0 or 'index': apply function to each column
#    * 1 or 'columns': apply function to each row
