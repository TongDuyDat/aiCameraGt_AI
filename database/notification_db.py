from mongoengine import Document, BooleanField, StringField, DateTimeField, ReferenceField
from datetime import datetime
class Notification(Document):
    username = StringField(required=True)
    id_record = StringField()
    content = StringField()
    token_expo = StringField()
    id_camera = StringField()
    created = DateTimeField(default=datetime.utcnow)
    meta = {"collection": "NOTIFICATION"}
    def get_all_notifications():
        notifications = Notification.objects()
        if notifications:
            return notifications
        return False  
              
    def get_notification_by_id(notification_id):
        notification = Notification.objects(id=notification_id).first()
        if notification:
            return notification
        else:
            return False
        
    def get_notification_by_expo_token(token):
        notification = Notification.objects(token_expo=token).order_by('-created').limit(20)
        if notification:
            if len(notification) > 10:
                notification =  notification[:10]
            return notification
        else:
            return False

    def create_notification(token_expo, username, id_record, content, id_camera):
        
        try:
            new_notification = Notification(
            token_expo = token_expo,
            username=username,
            id_record=str(id_record),
            content=content,
            id_camera = str(id_camera)
        )
            new_notification.save()
            return True
        except Exception as e:
            print("Lỗi  gì đó", e)
    @staticmethod
    def to_json(cls):
        return {
            "id":str(cls.id), 
            'username': cls.username,
            'id_record': cls.id_record,
            'content': cls.content,
            'token_expo': cls.token_expo,
            'id_camera': cls.id_camera,
            "created": str(cls.created)
        }
    