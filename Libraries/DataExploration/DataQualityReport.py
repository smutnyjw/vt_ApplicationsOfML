############################################################################
#   Author: John Smutny
#   Contributor: Dr Creed Jones
#   Date Updated: 03/06/2022
#   Date Created: 02/15/2022
#
#   Description:
#       Class dedicated to producing an informative Data Quality Report to
#       the console. This report should be used to better understand a dataset.
#
#       User must use the pandas.DataFrame object to insert columns of
#       features for processing.
#
#   Future Improvements:
#       1) Add ability to process columns of 'char' data values. Test Case: if
#       user attempts to '.addCol()' of 'chars' (A, B, C, etc); 1) the code
#       could convert those chars to ints using the ascii table, 2) do data
#       process as if it was a numeric for all non-float metrics like 'mean',
#       and 3) convert the resulting ints back to chars using the ascii table.
#
#       2) Add ability to publish DataQualityReport as a csv.
#
#   Parameters:
#       statsdf data frame summarizing the Data Quality Report from a dataset.
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

    ##################################################################
    # @input label    str column header to be added to the DQR.
    # @input data     DataFrame column of data to be processed. SINGLE COLUMN.
    # Des: Add a 'feature' to be included on the report. Append columns by 1.
    def addCol(self, label, data):
        cardinalityV = len(data.unique())
        n_missing = data.isnull().sum()

        # Only process columns of data that are 1) Numeric & 2) All have values.
        if type(data[0]) == str:
            # Data is a string type. Do not perform mathematical operations.
            print("WARN: DataQualityReport.addCol: Entered data is not a "
                  "numeric type. Setting relevant dataQuality values to zero")
            REPLACEMENT_VALUE = "NA"
            meanV = REPLACEMENT_VALUE
            medianV = REPLACEMENT_VALUE
            modeV = REPLACEMENT_VALUE
            stdDev = REPLACEMENT_VALUE
            minV = REPLACEMENT_VALUE
            maxV = REPLACEMENT_VALUE
            n_medianV = REPLACEMENT_VALUE
            n_modeV = REPLACEMENT_VALUE
        else:
            meanV = data.mean()
            medianV = data.median()
            modeV = data.mode()[0]
            stdDev = data.std()
            minV = data.min()
            maxV = data.max()

            # Various qualities throw errors when trying to calculate them.
            # 1) 'value_counts' cannot handle situations where there are
            #       multiple median or mode values.
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

    ##################################################################
    # @output str    String of the dataFrame.
    # Des: Publish DataQualityReport to the console
    def to_string(self):
        return self.statsdf.to_string()
