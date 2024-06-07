import requests
import json

def send_push_notification(token, title, body):
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }
    data = {
        'to': token,
        'title': title,
        'body': body,
    }
    response = requests.post('https://exp.host/--/api/v2/push/send', headers=headers, data=json.dumps(data))
    print("send notfication ..................................................................")
    return response.json()