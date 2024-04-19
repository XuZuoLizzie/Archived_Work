from requests import get
from bs4 import BeautifulSoup
import pymongo
import csv


def insert_city_data(collection):

    #mongodb
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["findu-db"]
    col_uni_all = mydb[col]



    item = []
    with open('cities2.csv', 'rt') as f:
        reader = csv.reader(f)
        your_list = list(reader)

    dlist = []
    your_list.pop(0)
    i=0


    c = {}
    c['quality_of_life_index'] = 0
    c['safety_index'] = 0
    c['cost_of_living_index'] = 0
    c['climate_index'] = 0
    col_uni_all.update_many({}, {"$set": c})

    for item in your_list:
        d = {}
        #item = item.split(',')
        d['quality_of_life_index'] = float(item[2])
        d['safety_index'] = float(item[3])
        d['cost_of_living_index'] = float(item[4])
        d['climate_index'] = float(item[5])
        c = {}
        #print(item[1])
        c['city'] = str(item[0])
        print(item[1])
        print(item[2])
        c['location'] = str(item[1])
        val = col_uni_all.find_one(c)
        print(val)
        result = col_uni_all.update_many(c, {"$set": d}, upsert=False)


col_list = ['results_uni', 'results_uni_cs', 'results_uni_eng']
for col in col_list:
    insert_city_data(col)
