from flask import jsonify, request
from database.camera_db import Camera
import json
from athu.auth_midde import token_required
from flask import Blueprint, request
camera_api = Blueprint('camera', __name__, url_prefix="api")
@camera_api.route('<username>/getAllcamera', methods=['GET'])
def get_cameras(username):
    """
        Get all the cameras in the system.
        
        Returns:
            A JSON object containing a list of camera objects if found, else an error message.  
    """
    
    try:
        # Query the database for all camera records and return as a response  
        # in JSON format.    
        data = Camera.get_all_cameras(username) 
        data = [Camera.to_json(cap) for cap in data] 
        return json.dumps(data)  
    except Exception as e: 
        print("Error occurred while fetching cameras")
        return {"error": str(e)}
@camera_api.route('<username>/getcamera', methods=['GET'])
def get_camera(username):
    """
        Get all the cameras in the system.
        
        Returns:
            A JSON object containing a list of camera objects if found, else an error message.  
    """ 
    # try:
        # Query the database for all camera records and return as a response  
        # in JSON format.
    id = request.args.get("id")
    data = Camera.get_camera_by_id(str(id))  
    return Camera.to_json(data)  
    # except Exception as e: 
    #     print("Error occurred while fetching cameras")
    #     return {"error": str(e)}
@camera_api.route('<username>/addcamera', methods=['POST'])
def add_new_camera(username):
    """
        Adds a new camera to the system. The details of the camera are sent in the body of the HTTP request.
        
        Returns:
            201 - If the camera is added successfully.
            400 - If there's any issue with the payload (missing fields).
            500 - For any other exception that occurs during processing.    
    """
    # Check if we have received a POST request.
    if request.method == 'POST':
        try:
            # Extract the camera information from the request body.
            camera_info = request.json
            # Validate whether required fields are present or not.
            req_fields = ['name', "username", "password", "rtsp_url", "token_expo"]
            for field in req_fields:
                if not field in camera_info.keys():
                    return {'error':'Field %s is missing from the request.'%field},  400
            
            # Create a new camera record using the provided info and save it into the database.
            camera_info["created_by"] = username
            cap = Camera.create_camera(**camera_info)
            # Return a successful status code along with the newly created camera id.
            return jsonify({"message":"Camera added successfully.", "data":Camera.to_json(cap)}),  200
        except ValueError as ve:    
            return {'error':str(ve)}, 400            
        except Exception as e:                  
            return {'error':str(e)}, 500                    
    else:
        return {'error':'This endpoint only accepts POST requests'}, 405            

@camera_api.route('<username>/editcamera', methods=['POST'])
@token_required
def edit_camera(username):
    """
        Edits an existing camera in the system. The details of the camera to be edited 
        are sent in the body of the HTTP request.
        
        Returns:
            200 - If the camera is edited successfully.
            400 - If there's any issue with the payload (missing fields) or the camera does not exist.
            500 - For any other exception that occurs during processing.    
    """
    # Check if we have received a POST request.
    if request.method == 'POST':
        try:
            # Extract the camera information from the request body.
            camera_info = request.json
            id_camera = request.args.get("id")
            # Validate whether required fields are present or not.
            req_fields = ['name', 'ip', "username", "password", "rtsp_url"]
            for field in req_fields:
                if not field in camera_info.keys():
                    return {'error':'Field %s is missing from the request.'%field},  400
            
            # Check if the camera exists.
            # Assuming 'id' field is present in the request body.
            if not id_camera:
                return {'error':'Camera ID is missing from the request.'},  400
                
            if not Camera.is_camera_exists(camera_info['name']):
                return {'error':'Camera does not exist.'},  400

            # Update the existing camera record using the provided info.
            Camera.update_camera(id_camera, **camera_info)
            
            # Return a successful status code.
            return {'message':"Camera has been edited."},  200
        except ValueError as ve:    
            return {'error':str(ve)}, 400            
        except Exception as e:                  
            return {'error':str(e)}, 500                    
    else:
        return {'error':'This endpoint only accepts POST requests'}, 405
@camera_api.route('<username>/deletecamera/', methods=['DELETE'])
@token_required
def delete_camera(username):
    """
        Deletes an existing camera from the system.
        
        Returns:
            200 - If the camera is deleted successfully.
            400 - If there's any issue with the request.
            404 - If the camera with the given ID does not exist.
            500 - For any other exception that occurs during processing.    
    """
    try:
        camera_id = request.args.get("id")
        # Check if the camera exists.
        camera = Camera.get_camera_by_id(camera_id)
        if not camera:
            return {'error': 'Camera not found.'}, 404
        
        # Check if the camera belongs to the user.
        if camera.created_by != username:
            return {'error': 'You do not have permission to delete this camera.'}, 403
        
        # Delete the camera.
        Camera.delete_camera(camera_id)
        
        # Return a successful status code.
        return {'message': 'Camera has been deleted.'}, 200
    except Exception as e:
        return {'error': str(e)}, 500