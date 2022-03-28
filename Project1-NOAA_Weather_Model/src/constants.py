'''
@package    aml.project1.const
@info       Constants and settings related to the data preparation and modeling
            of NOAA historical daily weather data
@author     bencjohn@vt.edu
'''

# Python standard libraries

# External libraries
from enum import Enum
import numpy as np
import re

# Lookup tables for various data fields
DEFAULT_DATA_FILE = 'data/USW00013880.csv'
DEFAULT_STATIONS_FILE = 'data/stations.txt'

NOAA_DAILY_HEADER = [
    'station_id',
    'date',
    'element',
    'value',
    'measurement_flag',
    'quality_flag',
    'source_flag',
    'obs_time'
]

OUTPUT_ID_COLUMNS = ['date', 'event_count', 'year', 'month', 'day']

CORE5_FEATURES = ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN']

class DATA_TYPES(Enum):
    NUMERICAL   = 0
    INTERVAL    = 1
    ORDINAL     = 2
    CATEGORICAL = 3
    BINARY      = 4
    TEXTUAL     = 5
    DISCRETE    = 6
    CONTINUOUS  = 7

def ElementDescription(element: str):
    ''' Get element description with lookup of enumerations
    :param element: String element name
    :return str: Element description
    '''
    if re.match(r"S[N,X][0-9][0-9]+", element):
        return ELEMENT_LABELS[element[:2]+'*#'].replace(
                    '%soil_type', GROUND_COVER[element[2]]
                ).replace(
                    '%soil_depth', str(DEPTH_CODES_CM[element[3]])
                )
    if re.match(r"W[V,T][0-9][0-9]+", element):
        return ELEMENT_LABELS[element[:2]+'**'].replace(
                    '%weather_type', WEATHER_TYPES[element[2:4]]
                )
    else:
        return ELEMENT_LABELS.get(element, None)

def ElementType(element: str):
    ''' Get element data type
    :param element: String element name
    :return DATA_TYPE: Enumeration value of data type for element
    '''
    if any(prefix in element for prefix in ['SN', 'SX']):
        return ELEMENT_TYPES[element[:2] + '*#']
    elif any(prefix in element for prefix in ['WV', 'WT']):
        return ELEMENT_TYPES[element[:2] + '**']
    else:
        return ELEMENT_TYPES.get(element, None)


ELEMENT_LABELS = {
    'PRCP': 'Precipitation (tenths of mm)',
    'SNOW': 'Snowfall (mm)',
    'SNWD': 'Snow Depth (mm)',
    'TMAX': 'Maximum Temperature (tenths of degrees C)',
    'TMIN': 'Minimum Temperature (tenths of degrees C)',
    'ACMC': 'Average Cloudiness - 24-hour 30s Auto (%)',
    'ACMH': 'Average Cloudiness - 24-hour Manual (%)',
    'ACSH': 'Average Cloudiness - Sunrise-Sunset Manual (%)',
    'ACSC': 'Average Cloudiness - Sunrise-Sunset 30s Auto (%)',
    'AWDR': 'Average daily wind direction (degrees)',
    'AWND': 'Average daily wind speed (tenths of m/s)',
    'DAEV': 'Number of days in multiday evaporation total',
    'DAPR': 'Number of days in multiday precipitation total',
    'DASF': 'Number of days in multiday snowfall total',
    'DATN': 'Number of days in multiday min temperature',
    'DATX': 'Number of days in multiday max temperature',
    'DAWM': 'Number of days in multiday wind movement',
    'DWPR': 'Number of days without precipitation in multiday precipitation',
    'EVAP': 'Evaporation of water from pan (tenths of mm)',
    'FMTM': 'Time of fastest windspeed (HHMM)',
    'FRGB': 'Base of frozen ground layer (cm)',
    'FRGT': 'Top of frozen ground layer (cm)',
    'FRTH': 'Thickness of frozen ground layer (cm)',
    'GAHT': 'Difference river and gauge height (cm)',
    'MDEV': 'Multiday evaporation total (tenths of mm)',
    'MDPR': 'Multiday precipitation total (tenths of mm)',
    'MDSF': 'Multiday snowfall total (??? - Assume mm)',
    'MDTN': 'Multiday min temperature (tenths of degrees C)',
    'MDWM': 'Multiday wind movement (km)',
    'MNPN': 'Daily min water temperature in pan (tenths of degrees C)',
    'MXPN': 'Daily max water temperature in pan (tenths of degrees C)',
    'PGTM': 'Peak gust time (HHMM)',
    'PSUN': 'Daily percent sunshine (%)',
    'SN*#': 'Minimum soil temperature %soil_type - %soil_depth cm (tenths of degrees C)',
    'SX*#': 'Maximum soil temperature %soil_type - %soil_depth cm (tenth of degrees C)',
    'TAVG': 'Average temperature (tenths of degrees C)',
    'THIC': 'Thickness of ice on water (tenths of mm)',
    'TOBS': 'Observation time temperature (tenths of degrees C)',
    'TSUN': 'Daily total sunshine (minutes)',
    'WDF1': 'Direction fastest 1-minute wind speed (degrees)',
    'WDF2': 'Direction fastest 2-minute wind speed (degrees)',
    'WDF5': 'Direction fastest 5-minute wind speed (degrees)',
    'WDFG': 'Direction peak wind gust (degrees)',
    'WDFI': 'Direction fastest instantaneous wind speed (degrees)',
    'WDFM': 'Fastest mile wind direction (degrees)',
    'WDMV': '24-hour wind movement (km)',
    'WESD': 'Water equivalent of snow on the ground (tenths of mm)',
    'WESF': 'Water equivalent of snowfall (tenths of mm)',
    'WSF1': 'Fastest 1-minute wind speed (tenths of m/s)',
    'WSF2': 'Fastest 2-minute wind speed (tenths of m/s)',
    'WSF5': 'Fastest 5-minute wind speed (tenths of m/s)',
    'WSFG': 'Peak gust wind speed (tenths of m/s)',
	'WSFI': 'Fastest instantaneous wind speed (tenths of m/s)',
	'WSFM': 'Fastest mile wind speed (tenths of m/s)',
    'WT**': '%weather_type present',
    'WV**': '%weather_type present in vicinity'
}

ELEMENT_TYPES = {
    'PRCP': DATA_TYPES.CONTINUOUS,
    'SNOW': DATA_TYPES.CONTINUOUS,
    'SNWD': DATA_TYPES.CONTINUOUS,
    'TMAX': DATA_TYPES.CONTINUOUS,
    'TMIN': DATA_TYPES.CONTINUOUS,
    'ACMC': DATA_TYPES.CONTINUOUS,
    'ACMH': DATA_TYPES.CONTINUOUS,
    'ACSH': DATA_TYPES.CONTINUOUS,
    'ACSC': DATA_TYPES.CONTINUOUS,
    'AWDR': DATA_TYPES.CONTINUOUS,
    'AWND': DATA_TYPES.CONTINUOUS,
    'DAEV': DATA_TYPES.DISCRETE,
    'DAPR': DATA_TYPES.DISCRETE,
    'DASF': DATA_TYPES.DISCRETE,
    'DATN': DATA_TYPES.DISCRETE,
    'DATX': DATA_TYPES.DISCRETE,
    'DAWM': DATA_TYPES.DISCRETE,
    'DWPR': DATA_TYPES.DISCRETE,
    'EVAP': DATA_TYPES.CONTINUOUS,
    'FMTM': DATA_TYPES.DISCRETE,
    'FRGB': DATA_TYPES.CONTINUOUS,
    'FRGT': DATA_TYPES.CONTINUOUS,
    'FRTH': DATA_TYPES.CONTINUOUS,
    'GAHT': DATA_TYPES.CONTINUOUS,
    'MDEV': DATA_TYPES.CONTINUOUS,
    'MDPR': DATA_TYPES.CONTINUOUS,
    'MDSF': DATA_TYPES.CONTINUOUS,
    'MDTN': DATA_TYPES.CONTINUOUS,
    'MDWM': DATA_TYPES.CONTINUOUS,
    'MNPN': DATA_TYPES.CONTINUOUS,
    'MXPN': DATA_TYPES.CONTINUOUS,
    'PGTM': DATA_TYPES.DISCRETE,
    'PSUN': DATA_TYPES.CONTINUOUS,
    'SN*#': DATA_TYPES.CATEGORICAL,
    'SX*#': DATA_TYPES.CATEGORICAL,
    'TAVG': DATA_TYPES.CONTINUOUS,
    'THIC': DATA_TYPES.CONTINUOUS,
    'TOBS': DATA_TYPES.CONTINUOUS,
    'TSUN': DATA_TYPES.DISCRETE,
    'WDF1': DATA_TYPES.CONTINUOUS,
    'WDF2': DATA_TYPES.CONTINUOUS,
    'WDF5': DATA_TYPES.CONTINUOUS,
    'WDFG': DATA_TYPES.CONTINUOUS,
    'WDFI': DATA_TYPES.CONTINUOUS,
    'WDFM': DATA_TYPES.CONTINUOUS,
    'WDMV': DATA_TYPES.CONTINUOUS,
    'WESD': DATA_TYPES.CONTINUOUS,
    'WESF': DATA_TYPES.CONTINUOUS,
    'WSF1': DATA_TYPES.CONTINUOUS,
    'WSF2': DATA_TYPES.CONTINUOUS,
    'WSF5': DATA_TYPES.CONTINUOUS,
    'WSFG': DATA_TYPES.CONTINUOUS,
	'WSFI': DATA_TYPES.CONTINUOUS,
	'WSFM': DATA_TYPES.CONTINUOUS,
    'WT**': DATA_TYPES.CATEGORICAL,
    'WV**': DATA_TYPES.CATEGORICAL
}

GROUND_COVER = {
    '0': 'Unknown',
    '1': 'Grass',
    '2': 'Fallow',
    '3': 'Bare ground',
    '4': 'Brome grass',
    '5': 'Sod',
    '6': 'Straw mulch',
    '7': 'Grass muck',
    '8': 'Bare muck',
}

DEPTH_CODES_CM = {
    '1': 5,
    '2': 10,
    '3': 20,
    '4': 50,
    '5': 100,
    '6': 150,
    '7': 180
}

WEATHER_TYPES = {
    '01': 'Fog (all types)',
    '02': 'Heavy fog (not always distinguished)',
    '03': 'Thunder',
    '04': 'Ice pellets/sleet',
    '05': 'Hail',
    '06': 'Glaze',
    '07': 'Dust, sand, ash storm',
    '08': 'Smoke/haze',
    '09': 'Drifting snow',
    '10': 'Tornado, waterspout, funnel cloud',
    '11': 'Dangerous winds',
    '12': 'Blowing spray',
    '13': 'Mist',
    '14': 'Drizzle',
    '15': 'Freezing drizzle',
    '16': 'Rain',
    '17': 'Freezing rain',
    '18': 'Snow',
    '19': 'Unknown precipitation',
    '20': 'Rain or snow shower',
    '21': 'Ground fog',
    '22': 'Ice fog',
}

QUALITY = {
    '': 'OK',
    np.nan: 'OK',
    'nan': 'OK',
    'NaN': 'OK',
    None: 'OK',
    'None': 'OK',
    'D': 'Duplicate failure',
    'G': 'Gap failure',
    'I': 'Internal consistency failure',
    'K': 'Frequent value/streak failure',
    'L': 'Multi-day length failure',
    'M': 'Megaconsistency failure (contradicts sibling values i.e. min > max)',
    'N': 'Naught check failure',
    'O': 'Climatological check failure',
    'R': 'Lagged range check failure',
    'S': 'Spatial consistency check failure',
    'T': 'Temporal consistency check failure',
    'W': 'Temperature too warm for snow',
    'X': 'Bound error',
    'Z': 'Result of Datzilla investigation'
}

MEASUREMENT = {
    '': 'Not present',
    np.nan: 'Not present',
    np.nan: 'Not present',
    'nan': 'Not present',
    'NaN': 'Not present',
    None: 'Not present',
    'None': 'Not present',
    'B': '2x12-hour precipitation total',
    'D': '4x6-hour precipitation total',
    'H': 'TMAX, TMIN, or TAVG',
    'K': 'Converted from knots',
    'L': 'Temperature lag from observation',
    'O': 'Converted from oktas',
    'P': 'Identified as missing or presumed zero',
    'T': 'Trace precipitation',
    'W': 'Converted from 16-point WBAN wind direction code'
}

SOURCE = {
    '': '',
    np.nan: '',
    'nan': '',
    'NaN': 'OK',
    None: '',
    'None': '',
    '0': 'US Coop (NCDC DSI-3200)',
    '7': 'US Coop -- Transmitted',
    'B': 'US ASOS 10/2000-12/2005 (DSI-3211)',
    'D': 'Short delay USNWS CF6 daily summary',
    'K': 'US Coop daily summary 2011-Present',
    'S': 'Global daily summary (NCDC DSI-9618)',
    'W': 'WBAN/ASOS daily summary (NCDC ISD)',
    'X': 'US First-Order daily summary (DSI-3210)',
    'Z': 'Datzilla official edit'
}

SOURCE_PRIORITY = [
    'Z', 'R', 'D', '0', '6', 'C', 'X', 'W', 'K',
    '7', 'F', 'B', 'M', 'm', 'r', 'E', 'z', 'u',
    'b', 's', 'a', 'G', 'Q', 'I', 'A', 'N', 'T',
    'U', 'H', 'S'
]

STATION_INFO = {
    'station_id': 'USW00013880',
    'latitude': 32.8986,
    'longitude': -80.0403,
    'altitude': 12.2,
    'state': 'SC',
    'name': 'CHARLESTON INTL AP',
    'gsn_flag': 'GSN',
    'hcn_flag': None,
    'wmo_id': 72208
}

NOAA_STATION_HEADER = [
    'ID'
    'LATITUDE',
    'LONGITUDE',
    'ELEVATION',
    'STATE',
    'NAME',
    'GSN FLAG',
    'HCN/CRN FLAG',
    'WMO ID'
]