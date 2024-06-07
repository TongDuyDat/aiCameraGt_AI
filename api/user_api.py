from flask import Blueprint, request
from database.user_db import User
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from athu.auth_midde import token_required
from datetime import datetime, timedelta
from config import settings
import jwt
# api 
user_api = Blueprint("user_api", __name__, url_prefix="api")
@user_api.route("/login", methods = ["POST", "GET"])
def login():
    if request.method == "POST":
        data = request.get_json()
        user = data["username"]
        passwd = data["password"]
        if User.login(user, passwd):
            token = jwt.encode({
            'username': user,
            'exp' : datetime.utcnow() + timedelta(minutes = 30000000)
        }, settings.SECRET_KEY, algorithm= settings.JWT_ALGORITHM)
            return {"status": True, "access_token": token}
    return {"status": False}

@user_api.route("/register", methods =["GET", "POST"])
def register():
    if request.method == "POST":
        try:
            data = request.get_json()
            if data is None:
                return {
                    "message": "Please provide user details",
                    "data": None,
                    "error": "Bad request"
                }, 400
            else:
                if data["username"] is None or data["password"] is None or data["email"] is None:
                    return {"message": False}
                user, status = User.register(**data)
                print(user, status)
                print(user)
                if status:
                    return {"message": "Successfully created new user",
                            "data": user}, 201
                else:
                    return {"message": "User already exists",
                            "data": None,
                            "error": "Bad request"}
        except Exception as e:
            return {
            "message": "Something went wrong",
            "error": str(e),
            "data": None
        }, 500
@user_api.route("/<username>/account", methods = ["PUT"])
@token_required
def account(user_id, username):
    data = request.get_json()
    if User.update_user(user_id, **data):
        return {"message":"Account updated successfully"}, 200
    else:
        return {"message":"Error updating the account"}, 400
    
@user_api.route("/get_users", methods=["GET"])
@token_required
def get_users(user_id):
    if str(user_id) == "653e6ea6cc9dc6413c4fad06":
        users, status = User.get_all_user()
        if status:
            return {
                "message": "All users fetched successfully",
                "data": users
            }
        return {
            "message": "Failed to fetch users",
            "data": None
        }, 500
    return {
        "message": "User is not an admin",
        "data": None,
        "error": "Unauthorized"
    }, 401

