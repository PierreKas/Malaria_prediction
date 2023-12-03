from django.shortcuts import render
import joblib

def Homepage(user_input):
    return render(user_input,"Homepage.html")

def results(request):
    load_model=joblib.load('malaria_model.joblib')
    lis=[]

    lis.append(request.GET['wbc_count'])
    lis.append(request.GET['rbc_count'])
    lis.append(request.GET['hb_level'])
    lis.append(request.GET['hematocrit'])
    lis.append(request.GET['mean_cell_volume'])
    lis.append(request.GET['mean_corp_hb'])
    lis.append(request.GET['mean_cell_hb_conc'])
    lis.append(request.GET['platelet_count'])
    lis.append(request.GET['platelet_distr_width'])
    lis.append(request.GET['mean_platelet_vl'])
    lis.append(request.GET['neutrophils_percent'])
    lis.append(request.GET['lymphocytes_percent'])
    lis.append(request.GET['mixed_cells_percent'])
    lis.append(request.GET['neutrophils_count'])
    lis.append(request.GET['lymphocytes_count'])
    lis.append(request.GET['mixed_cells_count'])
    lis.append(request.GET['RBC_dist_width_Percent'])
    lis.append(request.GET['temperature'])

    answer=load_model.predict([lis])

    return render(request,"results.html",{'Reponse':answer})


