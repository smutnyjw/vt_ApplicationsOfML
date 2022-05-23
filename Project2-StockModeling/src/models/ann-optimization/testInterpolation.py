

# Start of work
#######################

import pandas as pd
import datetime

def interpolateData(df: pd.DataFrame, FEATURES) -> pd.DataFrame:

    # perform hf.reformatDates
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.date
    df = df.sort_values(by='Date', ascending=True, ignore_index=True)

    sDate = df.loc[0, 'Date']
    eDate = df.loc[df['Date'].size-1, 'Date']
    print("StartDate {}\nEndDate {}".format(sDate, eDate))

    #df_dates = pd.DataFrame({'Date':pd.date_range(
        #sDate, eDate-datetime.timedelta(days=1), freq='d').tolist()})
    #df_dates = pd.date_range(
        #sDate, eDate-datetime.timedelta(days=1), freq='d')

    listDays = pd.date_range(sDate,
                             eDate-datetime.timedelta(days=1), freq='d')
    print("All Dates: {}".format(listDays))
    print("All Dates: {}".format(listDays.dt.date))


    #INPUT_COMPANY1 = 'QCOM-SimFin-CompanyCashFlow.csv'
    #df_cashFlow = pd.read_csv(INPUT_COMPANY1)

    #df_dates = pd.DataFrame({'Date': df['Date'], 'EX1': ""})

    return df





# Main
####################################

INPUT_QUALCOMM = '../../../data/QCOM_HistoricalData_5yr.csv'
df_raw = pd.read_csv(INPUT_QUALCOMM)

df = interpolateData(df_raw, 'Net Income/Starting Line')

print()


