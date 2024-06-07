from mongoengine import Document, StringField, IntField
from database.handerror import encrypt
from datetime import datetime
class User(Document):
    username = StringField(required=True, max_length=50)
    password = StringField(required=True)
    email = StringField(required=True)
    fullname = StringField(required=True)
    meta = {"collection": "USERS"}
    
    def login(username, passwd):
        user =  User.objects(username = username).first()
        if user and (user.password == encrypt(username+passwd)):
            return True
        return False
    
    def register(username, password, email, fullname):
        user = User.objects(username = username).first()
        if user is not None:
            return None, False
        else:
            try:
                user = User(
                username=username,
                password= encrypt(username+password),
                email= email,
                fullname= fullname,
                )
                user.save()
                return str(user.id), True
            except:
                return None, False
            
    def get_user_by_user_name(user_name):
        user = User.objects(username = user_name).first()
        return str(user.id)
    
    def delete_user(_id):
        user = User.objects(id = _id).first()
        if user:
            user.delete()
            return True
        return False
    
    @staticmethod
    def map_values(value, value_mapping):
        return value_mapping.get(value, None)

    