import re
import traceback
from CoeuSearch.main import *
from django.contrib import messages
from django.shortcuts import render

def home(request):
    return render(request,'CoeuSearch/home.html')

def search(request):
    if request.method == 'POST':
        err_flag = 0
        start_time = time.time()
        path = request.POST.get('path')
        query = request.POST.get('query')
        query = re.sub(" +"," ", query)

        if not os.path.exists(path):
            results, err_flag = None, 1
            messages.warning(request, "Search directory not found.")
        
        if not query or query == " ":
            results, err_flag = None, 1
            messages.warning(request, "Invalid search query.")
        
        if not err_flag:
            try:                    
                print(color.DARKCYAN + "--"*100 + color.END)
                print(color.GREEN + "[INPUT RECEIVED]" + color.END)
                print(color.CYAN + f"\t[PATH]: {path}" + color.END)
                print(color.CYAN + f"\t[QUERY]: {query}" + color.END)
                
                probs = configs.probs
                results, end_time = getFiles(path, query, probs)
                time_taken = np.round(end_time - start_time, 2)

                print(color.DARKCYAN + "--"*100 + color.END)
                print(color.GREEN + f'[TIME TAKEN]: {time_taken}s' + color.END)
                print(color.GREEN + "[RESULTS]:" + color.END)
                print(color.CYAN + f"\t{results}" + color.END)
                print(color.DARKCYAN + "--"*100 + color.END)
                
                if 'error' in results.keys():
                    messages.warning(request, str(results['error']))
                else:
                    if len(results['files']) <= 0:
                        results = None
                        messages.warning(request, "No matching files found.")
                    messages.success(request, f'Time Taken: {time_taken}s')

            except Exception as E:
                print(color.RED + f"Exception Occurred: {E}" + color.END)
                results = None
                messages.warning(request, 'Oops! There was some error from our side')
    return render(request,'CoeuSearch/home.html', {"results":results, "path":path, "query":query })
