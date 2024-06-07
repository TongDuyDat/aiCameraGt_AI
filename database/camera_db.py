from datetime import datetime
from mongoengine import Document, StringField, BooleanField, IntField, DateTimeField

class Camera(Document):
    name = StringField(required=True)
    camera_type_id = StringField()
    username = StringField()
    password = StringField()
    rtsp_url = StringField()
    status = BooleanField(default=True)
    position_id = StringField()
    camera_width = IntField()
    camera_height = IntField()
    port = StringField()
    ip_address = StringField()
    created_by = StringField()
    token_expo = StringField()
    created_at = DateTimeField(default=datetime.utcnow) 
    meta = {"collection": "CAMERA"}
    @staticmethod
    def create_camera(name, username, password, rtsp_url,created_by, token_expo, camera_type_id ='', position_id='', camera_width=0, camera_height=0, port='', ip_address=''):
        try:
            camera = Camera(
                name=name,
                camera_type_id=camera_type_id,
                username=username,
                password=password,
                rtsp_url=rtsp_url,
                position_id=position_id,
                camera_width=camera_width,
                camera_height=camera_height,
                port=port,
                ip_address=ip_address,
                created_by = created_by,
                token_expo = token_expo
            )
            camera.save()
            return camera
        except Exception as e:
            return False, str(e)

    @staticmethod
    def update_camera(camera_id, **kwargs):
        try:
            Camera.objects(id=camera_id).update_one(**kwargs)
            return True
        except Exception as e:
            return False, str(e)

    @staticmethod
    def delete_camera(camera_id):
        try:
            Camera.objects(id=camera_id).delete()
            return True
        except Exception as e:
            print(e)
            return False

    @staticmethod
    def get_all_cameras(created_by):
        return Camera.objects(created_by = created_by).order_by('-created_at')

    @staticmethod
    def get_camera_by_id(camera_id):
        return Camera.objects(id=camera_id).first()
    @staticmethod
    def is_camera_exists(id):
        return Camera.objects(id=id).first() is not None
    @staticmethod
    def to_json(cls):
        return {
            "id": str(cls.id),
            "name":cls.name,
            "camera_type_id":cls.camera_type_id,
            "username":cls.username,
            "password":cls.password,
            "rtsp_url":cls.rtsp_url,
            "position_id":cls.position_id,
            "camera_width":cls.camera_width,
            "camera_height":cls.camera_height,
            "port":cls.port,
            "ip_address":cls.ip_address,
            "created_by" : cls.created_by
        }