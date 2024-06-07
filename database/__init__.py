from mongoengine import connect
try:
    username = 'dattongduy10'
    password = 'TongDat1.'
    hostname = 'cluster0.pvie75s.mongodb.net'
    database_name = 'ABE'
    print("connect db")
    #mongodb+srv://dattongduy10:TongDat1.@aicamera.iuo8xkc.mongodb.net/
    connect(
        host="mongodb+srv://dattongduy10:TongDat1.@aicamera.iuo8xkc.mongodb.net/aicamera"
    )
    print("Connect Success with  MongoDB!")
except:
    print("connect db error")