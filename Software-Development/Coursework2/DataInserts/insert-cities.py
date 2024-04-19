from requests import get
from bs4 import BeautifulSoup
import pymongo


def clean(entry):
    if entry.find('...')!=-1:
        return entry.replace('...', '')
    elif entry.find('The ')!=-1:
        return entry.replace('The ', '')
    elif entry.find('University College London'):
        return entry.replace('University College London', 'UCL')
    else:
        return entry


def nullify(col):

    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["findu-db"]
    col_uni_all = mydb[col]

    #add null
    c = {}
    c['city'] = 'no data available'
    col_uni_all.update_many({}, {"$set": c})

def insert_cities(url, col):

    response = get(url)
    html_soup = BeautifulSoup(response.text, 'html.parser')

    uni_containers = html_soup.find('table')
    uni_data = uni_containers.findAll('tr')
    print(len(uni_containers))
    print(len(uni_data))

    #mongodb
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["findu-db"]
    col_uni_all = mydb[col]



    list_of_rows = []
    for row in uni_containers.findAll('tr'):
        list_of_cells = []
        for cell in row.findAll(["td"]):
            text = cell.text
            text = clean(text)
            list_of_cells.append(text)
        list_of_rows.append(list_of_cells)
    list_of_rows.pop(0)
    list_of_rows.pop(0)
    list_of_rows.pop()
    list_of_rows.pop()
    city_data = []
    print(city_data)


    for item in list_of_rows:
        print(item)
        d = {}
        c = {}
       #print(item[1])
        d['name'] = str(item[1])
        c['city'] = str(item[2].strip())
        #print(item[2])
        val = col_uni_all.find_one(d)
        print(val)
        result = col_uni_all.update_many(d, {"$set": c}, upsert=False)

url = 'https://www.4icu.org/gb/'
col = 'results_uni'
nullify(col)
insert_cities(url, col)
url = 'https://www.4icu.org/us/'
insert_cities(url, col)

url = 'https://www.4icu.org/gb/'
col = 'results_uni_cs'
nullify(col)
insert_cities(url, col)
url = 'https://www.4icu.org/us/'
insert_cities(url, col)


url = 'https://www.4icu.org/gb/'
col = 'results_uni_eng'
nullify(col)
insert_cities(url, col)
url = 'https://www.4icu.org/us/'
insert_cities(url, col)
