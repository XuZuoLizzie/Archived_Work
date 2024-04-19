import pymongo
import json
from requests import get
from bs4 import BeautifulSoup

def float_cast(item):
    if item.find('.')!=-1:
        return float(item)
    elif item.find('.')==-1:
        return int(item)

def num_cast(item):
    try:
        return float_cast(item)
    except ValueError:
        return item

def num_cast2(item):
    if item.find('=')!=-1:
        return int(item.replace('=', ''))
    elif item.find('-')!=-1:
        return int(item.split('-')[0])
    elif item.find('—')!=-1:
        return int(item.split('—')[0])
    elif item.find('—')!=-1:
        return int(item.split('—')[0])
    elif item.find('–')!=-1:
        if item.find('.')!=-1:
            return float(item.split('–')[0])
        else:
            return int(item.split('–')[0])
    elif item.find('.')!=-1:
        return float(item)
    elif item.find('+')!=-1:
        return int(item.replace('+', ''))
    try:
        return int(item)
    except ValueError:
        print(item)
        return 0

def insert_unis(url, collection, mode):

    url_shanghai = 'http://www.shanghairanking.com/ARWU2018.html'
    # html request
    response = get(url_shanghai)
    html_soup = BeautifulSoup(response.text, 'html.parser')
    uni_containers = html_soup.find('table')
    uni_data = uni_containers.findAll('tr')

 
    list_of_rows = []
    for row in uni_containers.findAll('tr'):
        list_of_cells = []
        for cell in row.findAll(["div","td"]):
            text = cell.text
            list_of_cells.append(text)
        list_of_rows.append(list_of_cells)
    list_of_rows.pop(0)


    #THE
    response = get(url)
    THE_data = json.loads(response.text)
    THE_data == response.json()
    mylist = []
    stringlist = ["rank", "scores_citations", "scores_industry_income", "scores_international_outlook", "scores_overall", "scores_research", "scores_teaching", "stats_student_staff_ratio"]


    mylist2 = []
    for p in THE_data['data']:
        mylist2.append( p['name'])
        mylist2.append( p['location'])
        for val in stringlist:
            mylist2.append(num_cast2(p[val]))
        mylist2.append( num_cast2(str(p['stats_pc_intl_students']).replace('%', '')) )
        mylist2.append( num_cast2(str(p['stats_number_students']).replace(',', '') ))
        mylist2.append( num_cast2(str(p['stats_female_male_ratio']).split()[0]  ))
        mylist2.append(0)
        mylist2.append(0)
        for item in list_of_rows:
            if item[1].lower() == p['name'].lower():
                del mylist2[-1]
                del mylist2[-1]
                mylist2.append( num_cast2(item[0]) )
                mylist2.append( num_cast2(item[3]) )

    newlist = []
    size = len(stringlist)
    size +=5

    stringlist.insert(0, "name")
    stringlist.insert(1, "location")
    stringlist.append("stats_pc_intl_students")
    stringlist.append("stats_number_students")
    stringlist.append("stats_female_male_ratio")
    stringlist.append("s_rank")
    stringlist.append("s_rank_country")


    d = {}
    dlist = []
    j=0
    for val in range(0, len(mylist2)-len(stringlist), len(stringlist)):
        d = {}
        for i in range(0,len(stringlist)):
            d['id'] = j
            d[stringlist[i]] = mylist2[i+val]
            d['new_rank'] = 0
        dlist.append(d)
        j+=1


    if mode=='I':
        for item in dlist:
            if (item['location'] == 'United Kingdom') or (item['location'] == 'United States'):
                collection.insert(item)
    elif mode=='D':
        collection.delete_many({})


# urls
url_THE = 'https://www.timeshighereducation.com/sites/default/files/the_data_rankings/world_university_rankings_2019_limit0_7216a250f6ae72c71cd09563798a9f18.json'
url_THE_cs ='https://www.timeshighereducation.com/sites/default/files/the_data_rankings/computer_science_rankings_2019_limit0_cd2ac383e6941d14a6f03236a3d78ada.json'
url_THE_eng = 'https://www.timeshighereducation.com/sites/default/files/the_data_rankings/engineering_technology_rankings_2019_limit0_f8ee736742bd112c8fb640463353ae04.json'
url_list = [url_THE, url_THE_cs, url_THE_eng]

#mongodb
col_list = []
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["findu-db"]
col_uni_all = mydb["results_uni"]
col_list.append(col_uni_all)
col_uni_cs = mydb["results_uni_cs"]
col_list.append(col_uni_cs)
col_uni_eng = mydb["results_uni_eng"]
col_list.append(col_uni_eng)

for i in range(0, len(col_list)):
    insert_unis(url_list[i], col_list[i], 'I')


