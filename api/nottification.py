import json
from flask import Blueprint, Flask, request, jsonify
from flask_cors import CORS
from mongoengine import connect
from database.notification_db import Notification 

notifications = Blueprint('notifications', __name__, url_prefix="/api")
# CORS(notifications)

@notifications.route('/notifications', methods=['GET'])
def get_all_notifications():
    notifications = Notification.get_all_notifications()
    if notifications:
        return jsonify([notification.to_json() for notification in notifications]), 200
    else:
        return jsonify({'error': 'No notifications found'}), 404

@notifications.route('/notifications/<notification_id>', methods=['GET'])
def get_notification_by_id(notification_id):
    notification = Notification.get_notification_by_id(notification_id)
    if notification:
        return jsonify(notification.to_json()), 200
    else:
        return jsonify({'error': 'Notification not found'}), 404
    
@notifications.route('/notifications/expo', methods=['GET'])
def get_notification_by_expo():
    try:
        token = request.args.get('token')
        if not token:
            return jsonify({'error': 'Token is missing'}), 200

        notifications = Notification.get_notification_by_expo_token(token)
        if notifications:
            notifications_json = [Notification.to_json(notification) for notification in notifications]
            return json.dumps(notifications_json), 200
        else:
            return jsonify({'error': 'Notification not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500

