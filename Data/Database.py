import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import argparse

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



parser = argparse.ArgumentParser()
parser.add_argument("-q", help="for search term")


args = parser.parse_args()

# All URLs
urlRD = f"https://www.teamrankings.com/college-football/stat/opponent-rushing-yards-per-game"
urlPD = f"https://www.teamrankings.com/college-football/stat/opponent-passing-yards-per-game"
urlRO = f"https://www.teamrankings.com/college-football/stat/rushing-yards-per-game"
urlPO = f"https://www.teamrankings.com/college-football/stat/passing-yards-per-game"
urlSD = f"https://www.teamrankings.com/college-football/stat/opponent-points-per-game"
urlSO = f"https://www.teamrankings.com/college-football/stat/points-per-game"
urlTR = f"https://www.teamrankings.com/ncf/trends/win_trends/"
urlTD = f"https://www.teamrankings.com/college-football/stat/takeaways-per-game"
urlTO = f"https://www.teamrankings.com/college-football/stat/giveaways-per-game"
urlPR = f"https://www.teamrankings.com/college-football/ranking/predictive-by-other"
urlHome = f"https://www.teamrankings.com/college-football/ranking/home-by-other"
urlAway = f"https://www.teamrankings.com/college-football/ranking/away-by-other"
urlSOS = f"https://www.teamrankings.com/college-football/ranking/schedule-strength-by-other"
urlWP = f"https://www.teamrankings.com/ncf/trends/win_trends/?date=%7B%7D"


# Getting requests of each URl
RD = requests.get(urlRD)

PD = requests.get(urlPD)

RO = requests.get(urlRO)

PO = requests.get(urlPO)

SD = requests.get(urlSD)

SO = requests.get(urlSO)

TD = requests.get(urlTD)

TO = requests.get(urlTO)

TR = requests.get(urlTR)

PR = requests.get(urlPR)

Home = requests.get(urlHome)

Away = requests.get(urlAway)

SOS = requests.get(urlSOS)

WP = requests.get(urlWP)


# Setting up text parser
soupRD = bs(RD.text, "html.parser")

soupPD = bs(PD.text, "html.parser")

soupRO = bs(RO.text, "html.parser")

soupPO = bs(PO.text, "html.parser")

soupSD = bs(SD.text, "html.parser")

soupSO = bs(SO.text, "html.parser")

soupTD = bs(TD.text, "html.parser")

soupTO = bs(TO.text, "html.parser")

soupTR = bs(TR.text, "html.parser")

soupPR = bs(PR.text, "html.parser")

soupHome = bs(Home.text, "html.parser")

soupAway = bs(Away.text, "html.parser")

soupSOS = bs(SOS.text, "html.parser")

soupWP = bs(WP.text, "html.parser")

# Scraping Files into CSVs
RD_scrape = soupRD.find("table", {'class': 'tr-table'})
RDR =RD_scrape.find_all('tr')
RDdata =[]
for row in RDR:
    cells = row.find_all('td')  # Use 'th' for header cells
    RDdata.append([cell.text.strip() for cell in cells])
RDT=pd.DataFrame(RDdata)
RD_Table = RDT.iloc[1:, [1, 2]]
RD_Table.to_csv('CSVs/RD_Table.csv', index=False)



PD_scrape = soupPD.find("table", {'class': 'tr-table'})
PDR =PD_scrape.find_all('tr')
PDdata = []
for row in PDR:
    cells = row.find_all('td')  # Use 'th' for header cells
    PDdata.append([cell.text.strip() for cell in cells])
PDT=pd.DataFrame(PDdata)
PD_Table = PDT.iloc[1:, [1, 2]]
PD_Table.to_csv('CSVs/PD_Table.csv', index=False)


RO_scrape = soupRO.find("table", {'class': 'tr-table'})
ROR =RO_scrape.find_all('tr')
ROdata = []
for row in ROR:
    cells = row.find_all('td')  # Use 'th' for header cells
    ROdata.append([cell.text.strip() for cell in cells])
ROT=pd.DataFrame(ROdata)
RO_Table = ROT.iloc[1:, [1, 2]]
RO_Table.to_csv('CSVs/RO_Table.csv', index=False)



PO_scrape = soupPO.find("table", {'class': 'tr-table'})
POR =PO_scrape.find_all('tr')
POdata = []
for row in POR:
    cells = row.find_all('td')  # Use 'th' for header cells
    POdata.append([cell.text.strip() for cell in cells])
POT=pd.DataFrame(POdata)
PO_Table = POT.iloc[1:, [1, 2]]
PO_Table.to_csv('CSVs/PO_Table.csv', index=False)



SD_scrape = soupSD.find("table", {'class': 'tr-table'})
SDR =SD_scrape.find_all('tr')
SDdata=[]
for row in SDR:
    cells = row.find_all('td')  # Use 'th' for header cells
    SDdata.append([cell.text.strip() for cell in cells])
SDT=pd.DataFrame(SDdata)
SD_Table = SDT.iloc[1:, [1, 2]]
SD_Table.to_csv('CSVs/SD_Table.csv', index=False)


SO_scrape = soupSO.find("table", {'class': 'tr-table'})
SOR =SO_scrape.find_all('tr')
SOdata=[]
for row in SOR:
    cells = row.find_all('td')  # Use 'th' for header cells
    SOdata.append([cell.text.strip() for cell in cells])
SOT=pd.DataFrame(SOdata)
SO_Table = SOT.iloc[1:, [1, 2]]
SO_Table.to_csv('CSVs/SO_Table.csv', index=False)



TD_scrape = soupTD.find("table", {'class':'tr-table'})
TDR =TD_scrape.find_all('tr')
TDdata=[]
for row in TDR:
    cells = row.find_all('td')  # Use 'th' for header cells
    TDdata.append([cell.text.strip() for cell in cells])
TDT=pd.DataFrame(TDdata)
TD_Table = TDT.iloc[1:, [1, 2]]
TD_Table.to_csv('CSVs/TD_Table.csv', index=False)



TO_scrape = soupTO.find("table", {'class':'tr-table' })
TOR = TO_scrape.find_all('tr')
TOdata=[]
for row in TOR:
    cells = row.find_all('td')  # Use 'th' for header cells
    TOdata.append([cell.text.strip() for cell in cells])
TOT=pd.DataFrame(TOdata)
TO_Table = TOT.iloc[1:, [1, 2]]
TO_Table.to_csv('CSVs/TO_Table.csv', index=False)



TR_scrape = soupTR.find("table", {'class':'tr-table' })
TRR = TR_scrape.find_all('tr')
TRdata=[]
for row in TRR:
    cells = row.find_all('td')  # Use 'th' for header cells
    TRdata.append([cell.text.strip() for cell in cells])
TRT=pd.DataFrame(TRdata)
TR_Table = TRT.iloc[1:, [0, 2]]
TR_Table[2] = TR_Table[2].str.rstrip('%').astype('float')
TR_Table.to_csv('CSVs/TR_Table.csv', index=False)

PR_scrape = soupPR.find("table", {'class':'tr-table' })
PRR = PR_scrape.find_all('tr')
PRdata = []
for row in PRR:
    cells = row.find_all('td')  
    if len(cells) > 2:  
        team_name = cells[1].find('a').text.strip()  
        rating = cells[2].text.strip()
        PRdata.append([team_name, rating])  
PRT = pd.DataFrame(PRdata, columns=['Team', 'Rating'])
PRT.to_csv('CSVs/PR_Table.csv', index=False)


Home_scrape = soupHome.find("table", {'class':'tr-table' })
HomeR = Home_scrape.find_all('tr')
Home_data = []
for row in HomeR:
    cells = row.find_all('td')
    if len(cells) > 2:
        team_name = cells[1].find('a').text.strip()
        rating = cells[2].text.strip()
        Home_data.append([team_name, rating])
HomeT = pd.DataFrame(Home_data, columns=['Team', 'Rating'])
HomeT.to_csv('CSVs/Home_Table.csv', index=False)

Away_scrape = soupAway.find("table", {'class':'tr-table' })
AwayR = Away_scrape.find_all('tr')
Away_data = []
for row in AwayR:
    cells = row.find_all('td')
    if len(cells) > 2:
        team_name = cells[1].find('a').text.strip()
        rating = cells[2].text.strip()
        Away_data.append([team_name, rating])
AwayT = pd.DataFrame(Away_data, columns=['Team', 'Rating'])
AwayT.to_csv('CSVs/Away_Table.csv', index=False)

SOS_scrape = soupSOS.find("table", {'class':'tr-table' })
SOSR = SOS_scrape.find_all('tr')
SOS_data = []
for row in SOSR:
    cells = row.find_all('td')
    if len(cells) > 2:
        team_name = cells[1].find('a').text.strip()
        rating = cells[2].text.strip()
        SOS_data.append([team_name, rating])
SOST = pd.DataFrame(SOS_data, columns=['Team', 'Rating'])
SOST.to_csv('CSVs/SOS_Table.csv', index=False)

WP_scrape = soupWP.find("table", {'class':'tr-table' })
WPR = WP_scrape.find_all('tr')
WP_data = []
for row in WPR:
    cells = row.find_all('td')
    if len(cells) > 2:
        team_name = cells[0].find('a').text.strip()
        WinPct = cells[2].text.strip().replace("%", "")
        WP_data.append([team_name, WinPct])
WPT = pd.DataFrame(WP_data, columns=['Team', 'WinPct'])
WPT.to_csv('CSVs/WP_Table.csv', index=False)



