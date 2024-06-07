from mongoengine import Document, StringField, DateTimeField, ReferenceField
from datetime import datetime
from database.notification_db import Notification

class Record(Document):
    file_path = StringField(required=True)
    created_at = DateTimeField(default=datetime.now)
    username = StringField(required=True)
    id_camera = StringField()
    content = StringField()
    meta = {"collection": "RECORD"}
    def create_record(file_path, username, id_camera, content):
        try:
            new_record = Record(
            file_path=file_path,
            username=username,
            id_camera = id_camera,
            content = content
            )
            new_record.save()
            return str(new_record.id)
        except Exception as e:
            print(e)
            return False
    def update_record(record_id, new_file_path):
        record = Record.objects(id=record_id).first()
        if record:
            record.file_path = new_file_path
            record.save()
            return True
        else:
            return False
    def delete_record(record_id):
        record = Record.objects(id=record_id).first()
        if record:
            record.delete()
            return True
        else:
            return False
