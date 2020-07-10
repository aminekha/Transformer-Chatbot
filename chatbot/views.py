from django.shortcuts import render, render_to_response
from django.http import HttpResponse
from .test import Chatbot
import json
from django.views.decorators.csrf import csrf_exempt

# @csrf_exempt
# def get_response(request):
# 	response = {'status': None}

# 	if request.method == 'POST':
# 		data = json.loads(request.body.decode('utf-8'))
# 		message = data['message']

# 		chat_response = Chatbot.predict(message)
# 		response['message'] = {'text': chat_response, 'user': False, 'chat_bot': True}
# 		response['status'] = 'ok'

# 	else:
# 		response['error'] = 'no post data found'

# 	return HttpResponse(json.dumps(response), content_type="application/json")

@csrf_exempt
def index(request):
	response = {'status': None}
	template_name = "index.html"
	if request.method == 'POST':
		data = json.loads(request.body.decode('utf-8'))
		message = data['message']
		chat_response = Chatbot.predict(message)
		response['message'] = {'text': chat_response, 'user': False, 'chat_bot': True}
		response['status'] = 'ok'
	else:
		response['error'] = 'no post data found'
	message = json.dumps(response)

	print("Message is ---> ", message)

	context = {
		"message": message,
	}
	return render(request, template_name, context)