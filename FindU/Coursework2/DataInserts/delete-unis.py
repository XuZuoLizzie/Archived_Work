import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["findu-db"]
collection = mydb["results_uni"]

collection.delete_many({})

