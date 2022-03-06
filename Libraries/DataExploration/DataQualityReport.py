############################################################################
#   Contributor: John Smutny
#   Original Author: Dr Creed Jones
#
#   statsdf     data frame summarizing the Data Quality Report from a dataset.
############################################################################
import pandas

class DataQualityReport:
    # Contructor. Define what values are being calculated for the stats report.
    def __init__(self):
        self.statsdf = pandas.DataFrame()
        self.statsdf['stat'] = ['cardinality',
                                'mean',
                                'median',
                                'n_at_median',
                                'mode',
                                'n_at_mode',
                                'stddev',
                                'min',
                                'max',
                                'n_zero',
                                'n_missing']
        pass

    # Add a 'feature' to be included on the report. Append columns by 1.
    def addCol(self, label, data):
        cardinalityV = len(data.unique())
        meanV = data.mean()
        medianV = data.median()
        modeV = data.mode()[0]
        stdDev = data.std()
        minV = data.min()
        maxV = data.max()
        n_missing = data.isnull().sum()

        # Various qualities throw errors when trying to calculate them.
        #   1) 'value_counts' cannot handle situations where there are multiple
        #   median or mode values.
        try:
            n_medianV = data.value_counts()[medianV]
        except(TypeError, ValueError, KeyError):
            n_medianV = "N/A"

        try:
            n_modeV = data.value_counts()[modeV]
        except(TypeError, ValueError, KeyError):
            n_modeV = "N/A"

        #   2) 'value_counts()[0]' to find zero fields cannot handle blank
        #   values.
        try:
            n_zeros = data.value_counts()[0]
        except(TypeError, ValueError, KeyError):
            n_zeros = "N/A"

        self.statsdf[label] = [cardinalityV,
                               meanV,
                               medianV,
                               n_medianV,
                               modeV,
                               n_modeV,
                               stdDev,
                               minV,
                               maxV,
                               n_zeros,
                               n_missing]

    def to_string(self):
        return self.statsdf.to_string()
