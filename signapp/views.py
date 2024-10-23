from django.shortcuts import render
from keras.models import load_model
from signapp.models import SignLanguages
import cv2
import mediapipe as mp
import numpy as np
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect

# Now you can use the User model directly


# Create your views here.
@login_required
def index(request):
    return render(request,"index.html")

@login_required
def home(request):
    return render(request,"home.html")

@login_required
def TextToSign(request):
    if request.method=='POST':
        name=request.POST['text']
        value=SignLanguages.objects.filter(comment=name)
        if value:
            print("success")
            det=SignLanguages.objects.get(comment=name)
            return render(request,"texttosign.html",{'data':det})
    return render(request,"texttosign.html")

@login_required
def tutorials(request):
    return render(request,"tutorials.html")

@login_required
def alphabets(request):
    return render(request,"alphabets.html")

@login_required
def numbers(request):
    return render(request,"numbers.html")
        
@login_required
def upload_image(request):
    return render(request,"test.html") 
from django.http import JsonResponse

from rest_framework.decorators import api_view
from rest_framework.response import Response


from django.http import JsonResponse, HttpResponse
import base64
from PIL import Image
from io import BytesIO


model = load_model('static\hdf\cnn_model_final1.h5')
@api_view(['POST'])
def detect(request):
    if request.method == 'POST':
        try:
            image_data = request.data['image_data']
            # Decode the Base64 image data
            decoded_image_data = base64.b64decode(image_data.split(",")[1])
            # Convert the image data into a PIL image
            pil_image = Image.open(BytesIO(decoded_image_data))
            # Display the image in the browser (for debugging purposes)
            open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)



            
            def image_processed(hand_img):

                # Image processing
                # 1. Convert BGR to RGB
                img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

                # 2. Flip the img in Y-axis
                img_flip = cv2.flip(img_rgb, 1)

                # accessing MediaPipe solutions
                mp_hands = mp.solutions.hands

                # Initialize Hands
                hands = mp_hands.Hands(static_image_mode=True,
                max_num_hands=1, min_detection_confidence=0.7)

                # Results
                output = hands.process(img_flip)

                hands.close()
                if output.multi_hand_landmarks:
                    try:
                        data = output.multi_hand_landmarks[0]
                        #print(data)
                        data = str(data)

                        data = data.strip().split('\n')

                        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

                        without_garbage = []

                        for i in data:
                            if i not in garbage:
                                without_garbage.append(i)
                                        
                        clean = []

                        for i in without_garbage:
                            i = i.strip()
                            clean.append(i[2:])

                        for i in range(0, len(clean)):
                            clean[i] = float(clean[i])
                        return np.array(clean)
                    except:
                        return np.zeros(63)
                else:
                    return None

            def getLetter(result):
                classLabels = { 0: 'A',
                                1: 'B',
                                2: 'C',
                                3: 'D',
                                4: 'E',
                                5: 'F',
                                6: 'G',
                                7: 'H',
                                8: 'I',
                                9: 'I love You',
                            }
                try:
                    res = int(result)
                    return classLabels[res]
                except:
                    return "Error"

   
                
            data = image_processed(open_cv_image)
                
                # print(data.shape)
            data = np.array(data)
            y_pred = str(np.argmax(model.predict(data.reshape(-1,63))))
            print(getLetter(y_pred))

            
            result = {'detected_gesture': getLetter(y_pred)}
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    else:
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)



def loginr(request):
    if request.method == "GET":
        return render(request, "login.html")
    elif request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("index")  # Redirect to the index page upon successful login
        else:
            return HttpResponse("Invalid credentials")

def registerr(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get('email')
        password = request.POST.get('password')

        if not (username and email and password):  # Check if any field is empty
            return HttpResponse("All fields are required")

        # Check if the username or email already exists
        if User.objects.filter(username=username).exists() or User.objects.filter(email=email).exists():
            return HttpResponse("Username or email already exists")

        # Create a new user
        user = User.objects.create_user(username=username, email=email, password=password)
        if user:
            return render(request, "login.html")
        else:
            return HttpResponse("Error creating user")
    else:
        return HttpResponse("Method not allowed")
        

            
