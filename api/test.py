from flask import request
from database.camera_db import Camera
import json
from athu.auth_midde import token_required
from flask import Blueprint, request
test = Blueprint('test', __name__, url_prefix="api")
@test.route('test', methods=['GET'])
@token_required
def get_cameras(user):
    return "hello bạn nhỏ" + user