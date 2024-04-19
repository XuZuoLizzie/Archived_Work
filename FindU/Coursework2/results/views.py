import numpy
from django.shortcuts import render
from django.db.models import Max
# Create your views here.
from django.db.models import Case, When

from results.models import Uni
from results.models import Uni_cs
from results.models import Uni_eng

from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

def gen_rank(fields, scores, Subject):
    score_val=[]
    rank_val=[]
    max_val=[]
    scores = [x for x in scores if x != '0']
    q = Subject.objects.all().values_list(*fields)
    for i in range(0, len(fields)):
        max_val.append(get_max(q, fields[i]))
    for e in Subject.objects.values_list(*fields):
        val=0
        for i in range(0, len(fields)):
            norm = make_pc(float(e[i]),float(max_val[i]))
            val += float(scores[i]) * float(norm)
        score_val.append(val)
    count=0
    rank_val = get_rank_val(score_val)
    for e in Subject.objects.all():
        e.new_rank = rank_val[count]
        e.save(update_fields=['new_rank'])
        count+=1
    return

def get_rank_val(score_val):
    rank_val = len(score_val) - numpy.argsort(numpy.argsort(numpy.array(score_val)))
    return rank_val.tolist()

def get_max(query, field):
    return list(query.aggregate(Max(field)).values())[0]
def make_pc(value, max_val):
    return float(value*100/max_val)

def index(request, template_name='index.html'):
    """View function for home page of site."""

    context_dict = {}
    model = Uni
    column_headers = ['rank', 'name', 'location', 'city', 'scores_overall']
    check_fields = ['scores_citations', 'scores_industry_income', 'scores_international_outlook', 'scores_research', 'scores_teaching', 'stats_student_staff_ratio', 'stats_pc_intl_students', 'stats_number_students', 'quality_of_life_index', 'safety_index', 'cost_of_living_index', 'climate_index' ]
            
    uni_main = Uni.objects.values(*column_headers)
    
    # subject filter
    if request.GET.get('sub_drop'):
        subject_filter = request.GET.get('sub_drop')
        if subject_filter == 'All':
            uni_main = Uni.objects.values(*column_headers)

        elif subject_filter == 'Computer Science':
            model = Uni_cs
            uni_main = Uni_cs.objects.values(*column_headers)
        elif subject_filter == 'Engineering':
            model = Uni_eng
            uni_main = Uni_eng.objects.values(*column_headers)
    else:
        uni_main = Uni.objects.values(*column_headers)


    # Locations filter
    if request.GET.get('loc_drop'):
        location_filter = request.GET.get('loc_drop')
        if location_filter == 'All':
            listings = uni_main

        else:
            listings = uni_main.filter(location=location_filter)
    else:
        listings = uni_main

    # advanced search
    qs = listings
    if request.GET.getlist('v_scores[]'):
        scorevar = request.GET.getlist('v_scores[]')
        if scorevar != ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']:
            for i in range (0, len(scorevar)):
                if scorevar[i]=='0':
                    check_fields[i]='0' 
            check_fields = [x for x in check_fields if x != '0']
            checkvar = check_fields
            gen_rank(checkvar,scorevar,model)
            column_headers[0] = 'new_rank'
            values = column_headers + checkvar
            qs = listings.values(*values).order_by('new_rank')
    listings = qs
    uni_name = listings

    paginator = Paginator(listings, 10)
    page = request.GET.get('page')
    try:
        uni_list = paginator.page(page) 
    except PageNotAnInteger: 
        uni_list = paginator.page(1) 
    except EmptyPage: 
        uni_list = paginator.page(paginator.num_pages)

    context_dict = {'uni_list': uni_list, 'uni_name': uni_name, 'loc_list' : Uni.objects.order_by('location').values_list('location', flat=True).distinct()}

    # Render the HTML template index.html with the data in the context variable
#    return render(request, 'index.html', context=context)
    return render(request, template_name, context_dict)

def about(request):
    return render(request, 'about.html', {})

def compare(request):
    return render(request, 'compare.html', {})
