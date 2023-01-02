from urllib.request import urlopen
from xmltodict import parse
import pandas as pd
from tqdm import tqdm

def QuerySingleSeries(series):
    flowREF = series.split(".")[0].upper()
    key = series.replace(series.split(".")[0] + '.', '')
    try:
        raw1 = urlopen("https://sdw-wsrest.ecb.europa.eu/service/data/"+ flowREF +"/" + key)
        raw2 = raw1.read().decode('utf8')
        raw3 = parse(raw2)
        raw4 = raw3['message:GenericData']['message:DataSet']['generic:Series']['generic:Obs']
        table = pd.DataFrame([[series.upper(),x['generic:ObsDimension']['@value'],x['generic:ObsValue']['@value']] for x in raw4],
                              columns=['SERIES_KEY', 'OBS_DATE', 'OBS_VALUE'])
        ErrorAlert = 0
        return {'table':table,'ErrorAlert': ErrorAlert}
    except:
        table = pd.DataFrame()
        ErrorAlert = 1
        return {'table':table,'ErrorAlert': ErrorAlert}
    

def getECBData(ser):
    FinalTable = pd.DataFrame(columns=['SERIES_KEY', 'OBS_DATE', 'OBS_VALUE'])
    NotRetrievedSeries = []
    for series in tqdm(ser):
        ResultQuery = QuerySingleSeries(series)
        if ResultQuery['ErrorAlert'] :
            NotRetrievedSeries.append(series)
        else:
            FinalTable = pd.concat([FinalTable,ResultQuery['table']])
    # here we convert the columns to the appropriate data type for working with
    FinalTable.OBS_VALUE = FinalTable.OBS_VALUE.astype(float)
    FinalTable['OBS_DATE'] = pd.to_datetime(FinalTable['OBS_DATE'])
    return {'Data': FinalTable, 'Missing' : NotRetrievedSeries}
