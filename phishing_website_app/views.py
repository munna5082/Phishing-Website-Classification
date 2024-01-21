from urllib import response
from django.shortcuts import render
import pickle
import numpy as np

with open('websiteclassification.pkl', 'rb')as file:
    model = pickle.load(file)

# Create your views here.
def home(request):
    res = 0
    try:
        if request.method == 'POST':
            name = request.POST['Name']
            numdots = request.POST['NumDots']
            pathlevel = request.POST['PathLevel']
            numdash = request.POST['NumDash']
            numsensitivewords = request.POST['NumSensitiveWords']
            hyperlinks = request.POST['PctExtHyperlinks']
            resourceurls = request.POST['PctExtResourceUrls']
            insecureforms = request.POST['InsecureForms']
            selfredirecthyperlinks = request.POST['PctNullSelfRedirectHyperlinks']
            freqdomainnamemismatch = request.POST['FrequentDomainNameMismatch']
            submitinfotoemail = request.POST['SubmitInfoToEmail']
            frame = request.POST['IframeOrFrame']

            data = np.array([float(numdots), float(pathlevel), float(numdash), float(numsensitivewords), float(hyperlinks), float(resourceurls), float(insecureforms),
                            float(selfredirecthyperlinks), float(freqdomainnamemismatch), float(submitinfotoemail), float(frame)])
            data = data.reshape(1, -1)

            op = model.predict(data)

            if op[0] == 1:
                res = "It is a phising site. Beaware of it!!"
            else:
                res = "It is a Genuine site. Go for it!!"

            return render(request, 'show.html', {'response': res})
    except:
        return render(request, 'home.html', {'error': 'Error Occured: Please fill all required value'})
    
    return render(request, 'home.html')