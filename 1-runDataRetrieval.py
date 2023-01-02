import pandas as pd
import requests, zipfile, io
import ECBData.DataRetrieval as ecb

# On importe la documentation de la base de données ICP
url_doc = "https://sdw.ecb.europa.eu/datasetCatalog.do?dsiId=122"
UrlResponse = requests.get(url_doc)
z = zipfile.ZipFile(io.BytesIO(UrlResponse.content))
docu = pd.read_csv(z.open(zipfile.ZipFile.namelist(z)[0]), encoding= 'unicode_escape', dtype=str)

# On définit plusieurs filtres afin de récupérer les données qui nous intéressent
Filter1 = docu['series key'].str.contains('ICP.M.')
Filter2 = docu['icp_suffix'] == "ANR"
Filter3 = docu['dom_ser_ids'].notnull()  
Filter4 = docu['first date (dd-mm-yyyy)'] != docu['last date (dd-mm-yyyy)']
Filter5 = docu['first date (dd-mm-yyyy)'] >= "01-01-1995"

docu = docu[Filter1 & Filter2 & Filter3 & Filter4 & Filter5]
ListSeries = docu['series key'].tolist()

# On récupère les données via l'API de SDW.
Results = ecb.getECBData(ListSeries)

# On reshape nos données 
Data = Results['Data']
Data['COUNTRY'] = Data.apply(lambda x: x['SERIES_KEY'].split(".")[2], axis=1)
Data['ITEM_CODE'] = Data.apply(lambda x: x['SERIES_KEY'].split(".")[4], axis=1)

Dataset = Data.pivot(index=["COUNTRY","OBS_DATE"], columns="ITEM_CODE", values="OBS_VALUE")
Dataset.reset_index(inplace=True)

Dataset['OBS_DATE'] = pd.to_datetime(Dataset['OBS_DATE'])
Dataset['MONTH'] = Dataset['OBS_DATE'].dt.month
Dataset['YEAR'] = Dataset['OBS_DATE'].dt.year
Dataset['COUNTRY_IDX'] = Dataset['COUNTRY']

# On définit nos index pour notre panel et on sauvegarde nos données au format csv
Dataset.set_index(['COUNTRY_IDX', 'OBS_DATE'], inplace=True)
Dataset.to_csv('data/DatasetFromSDW.csv')
