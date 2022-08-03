from collections import defaultdict
import streamlit as st
import hydralit_components as hc
import pickle, gzip, json, requests, numpy, math, os
from datetime import timedelta
import pandas as pd
from bokeh.models import DataTable, TableColumn, HTMLTemplateFormatter, ColumnDataSource
from requests.adapters import HTTPAdapter, Retry
from stqdm import stqdm
from gensim.models.doc2vec import Doc2Vec


def download_file():
    with st.spinner('Skill Space Model not found! Downloading (This is only run once) ...'):
        with requests.get('https://www.dropbox.com/s/9pfnhr71nlbpi3s/doc2vec.U.PtAlAPI_U.ep1.trained.pickle.gz?dl=1', stream=True) as r:
            total = int(r.headers.get('content-length', 0))
            with open('tempfile', 'wb') as f, stqdm(total=total,unit='iB',unit_scale=True,unit_divisor=1024) as bar:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

        return 'tempfile'

@st.cache(persist=True, show_spinner=False, allow_output_mutation=True, suppress_st_warning=True)
def load_skill_space_model(filename):
    try:
        # if file exists
        with gzip.open(filename, 'rb') as f:
            mod = pickle.load(f)
    except:
        name = download_file()
        with gzip.open(name, 'rb') as f:
            mod = pickle.load(f)  
        os.remove(name)
    return mod

# Define the Cosine Similarity Function
def cos_sim (av, bv):
    return (sum(av*bv)/math.sqrt(sum(av*av)*sum(bv*bv)))

def show_table(obj, colnames):
    
    df = pd.DataFrame(obj, columns=colnames)
    df['Rank'] =  range(1,len(obj)+1)
    # print(df)
    cds = ColumnDataSource(df)
    columns = [
    TableColumn(field=colnames[x], title=colnames[x], 
    formatter=HTMLTemplateFormatter(template='<a href="<%= value %>"target="_blank"><%= value %>')) if x ==0 else \
        TableColumn(field=colnames[x], title=colnames[x]) for x in range(len(colnames))
    ]+[TableColumn(field="Rank", title="Rank")]

    p = DataTable(source=cds, columns=columns, css_classes=["card"], index_position=None,
    autosize_mode="fit_columns", syncable=False, width_policy='max', height=270, height_policy="auto")

    return(p)

def show_tz():
    with gzip.open('./data/tz_project_gender.json.gz', 'rt') as f:
        data = json.load(f)
        tz_list = [f"UTC-{str(timedelta(hours=-float(x)))[:-3]}" if float(x)<0 else f"UTC+{str(timedelta(hours=float(x)))[:-3]}" \
            for x in sorted(list(map(float, data.keys()))) ]
        tz_select = st.selectbox("Please Select Your Nearest TimeZone from this list", ['SELECT A TIMEZONE']+tz_list)
        
        if tz_select != 'SELECT A TIMEZONE':
            temptz = (tz_select.replace('UTC',''))[1:].split(':')
            tzoffset = str((float(temptz[0])+float(temptz[1])/60)*float(f"{(tz_select.replace('UTC',''))[0]}1") )
            if tzoffset == '0.0' :
                tzoffset = '0'
            st.success(f"Your Selected TimeZone is: {tz_select}")
        else:
            tzoffset = tz_select

    return (data,tzoffset)

def check_project_url(project):
    project = project.replace('__','_')
    if 'gitlab.com' not in project and 'bitbucket.org' not in project and 'gitbox.com' not in project:
        url = 'https://github.com/' + project.replace('_', '/', 1)
    else:
        url = 'https://' + project.replace('_', '/', 2)
    # check if url exists
    s = requests.Session()
    try:
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[ 429, 502, 504 ])
        s.mount('https://', HTTPAdapter(max_retries=retries))
        response = s.get(url)
    except:
        return None
    if response.status_code == 200:
        return url
    else:
        return None

def recommend_project(apis, languages, langdict, mod):
    poslist = numpy.zeros((200,))
    # get WoC language names
    if type(languages) == str:
        poslist += mod.dv[langdict[languages]]
    elif type(languages) == list:
        for x in languages:
             poslist += mod.dv[langdict[x]]

    for api in apis.split(';'):
        api = api.strip()
        try:
            poslist += mod.wv.get_vector(api)
        except:
            return (ValueError('API '+api+' Not Found in our data'))
    
    # get similar tags 
    similar_tags = mod.dv.most_similar(positive=[poslist], topn = 1000)
    
    return poslist, similar_tags

def transfer_project(source_lang, dest_lang, apis, mod, langdict, no_api=0):
    poslist = mod.dv[langdict[dest_lang]] - mod.dv[langdict[source_lang]]

    for api in apis.split(';'):
        api = api.strip()
        try:
            poslist += mod.wv.get_vector(api)
        except:
            return (ValueError('API '+api+' Not Found in our data'))
    
    # get similar tags 
    similar_tags = mod.dv.most_similar(positive=[poslist], topn = 1000)

    if no_api > 0:
        similar_apis = mod.wv.most_similar(positive=[poslist], topn = no_api)
        return poslist, similar_tags, similar_apis
    else:
        return poslist, similar_tags

def show_mentors(dev_vect, cores, mod):
    # calculate similarity of dev & cores
    dev_core_proj_sim = []
    for core in cores.keys():
        if 'noreply' in core or '.' not in core:
            continue
        try:
            cvec = mod.dv[core]
            projs = ','.join(cores[core])
            sim = cos_sim(dev_vect,cvec)
            dev_core_proj_sim.append([core, projs, sim])
        except:
            continue
    # sort
    dev_core_proj_sim.sort(key=lambda x: x[2], reverse=True)
    # top 10
    top_ment = dev_core_proj_sim[:10]
    colnames = ['Potential Mentor', 'Core Developer in Project', 'Similarity']
    
    st.header(f'Mentor Recommendation Table - Sorted by similarity (scrollable)')
    df = pd.DataFrame(top_ment, columns=colnames)
    st.table(df)


def show_project_recommendation_table(similar_tags, no_project, proj_info, is_diversity, exclude, lang, gender_pct=0):
    if type(similar_tags) == ValueError:
        st.write(similar_tags)
    else:
    # filter for projects & check if exists
        with st.spinner('Model Loaded. Getting your Project Recommendations ...'):
            i = 0
            colnames = ['Project URL', 'Similarity', 'No. Stars', 'No. Forks', 'Total No. Contributors','Female Developer Percentage' ]
            rows = []
            cores = defaultdict(list)
            # filtered_proj = {k:v for k,v in proj_info.items() if langselect in v['FileInfo'].keys()}
            for element,similarity in similar_tags:
                if i >= no_project: 
                    break
                if '_' in element and '<' not in element: 
                    if element in exclude:
                        continue
                    try:
                        female_pct = proj_info[element]['female_pct']
                        if type(lang) == str:
                            if lang not in proj_info[element]['FileInfo'].keys():
                                continue
                        elif type(lang) == list:
                            if not any(lx in proj_info[element]['FileInfo'].keys() for lx in lang):
                                continue
                    except:
                        continue
                    if is_diversity:
                        if female_pct >= gender_pct:
                            # check if exist
                            url = check_project_url(element)
                            if url:
                                rows.append([url, "{:.2f}".format(similarity),  proj_info[element]['NumStars'], proj_info[element]['NumForks'],
                                proj_info[element]['NumAuthors'],f'{female_pct:.2f}%' ])
                                for k in proj_info[element]['Core'].keys():
                                    cores[k].append(url)
                                i += 1
                    else:
                        # check if exist
                        url = check_project_url(element)
                        if url:
                            rows.append([url, "{:.2f}".format(similarity),  proj_info[element]['NumStars'], proj_info[element]['NumForks'],
                            proj_info[element]['NumAuthors'],f'{female_pct:.2f}%' ])
                            for k in proj_info[element]['Core'].keys():
                                cores[k].append(url)
                            i += 1
            p = show_table(rows, colnames)
            st.header(f'Project Recommendation Table - Sorted by similarity (scrollable)')
            with st.expander("INFORMATION & DISCLAIMER"):
                st.info("""The results shown here are based on World of Code (WoC) dataset version U, collected on Nov. 2021.
                WoC uses a community-resolution based approach to calculate various statistics, so the values may differ from what you see 
                on GitHub. Since we did not use any fork-resolution, the projects might be a fork of another project, in which case, 
                the user is recommeded to look into the source project. Any obvious inconsistencies should be reported to WoC maintainers.""")
                st.info('The Tables are Interactive and can be SORTED by any of the columns - just click on the column header!')
                st.warning(
                    """We fully support people who do not identify with either of the two binary genders, 
                    the reason for showing only two genders here is because WoC only identifies people as male or female 
                    (or 'Unknown' for developers who do not provide a common name)."""
                )
            st.bokeh_chart(p)

            return cores
                    

def show_page():
    ###################################################
    #define custom font css for tables
    ###################################################
    st.markdown("""
    <style>
    .card {
    box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    transition: 0.3s;
    width: 100%;
    overflow: hidden;
    border-style: solid;
    border-color: #F0F0F0;
    border-width: thin;
    border-radius: 10px; /* 5px rounded corners */
    background-color: white;
    margin-bottom: 30px;
    font-size:12px !important;
    }
    .card-text {
        word-wrap: break-word;
        margin-left: 3rem;
    }
    .card:hover {
        box-shadow: 0 16px 32px 0 rgba(0,0,255,0.2);
        z-index: 2;
    -webkit-transition: all 200ms ease-in;
    -webkit-transform: scale(1.1);
    -ms-transition: all 200ms ease-in;
    -ms-transform: scale(1.1);
    -moz-transition: all 200ms ease-in;
    -moz-transform: scale(1.1);
    transition: all 200ms ease-in;
    transform: scale(1.1);
    }
    </style>
    """, unsafe_allow_html=True)

    ###################################################
    # Define Navbar
    ###################################################
    nav_items = [
    {'label':'Expertise based Recommendation', 'id':'exp', 'icon':"fas fa-user-cog"}, 
    {'label':'Skill-Transfer based Recommendation', 'id':'trans', 'icon':"fas fa-exchange-alt"},
    {'label':'Popularity based Recommendation', 'id':'pop', 'icon':"fas fa-users"}
    ]

    nav_id = hc.nav_bar(menu_definition=nav_items, hide_streamlit_markers=True)

    ###################################################
    # Define supporting variables
    ###################################################
    langdict = {'C/C++':'C', 'C#':'Cs', 'Go':'Go', 'Perl':'pl', 'Ruby':'rb', 'JavaScript':'JS',\
        'Python':'PY', 'R':'R', 'Rust':'Rust', 'Scala':'Scala', 'TypeScript':'Typescript',  'Java':'java'}
    
    with gzip.open('./data/Proj_info.pickle.gz', 'rb') as fproj:
        proj_info = pickle.load(fproj)

    proj_langs = proj_info.pop('langs')
    proj_langs.add('ALL')
    gender_pct = 0

    exclude = ['frioux_dotfiles', 'auto-program_vendor', 'Reese-D_my_emacs', 'bloomberg_chromium.bb', '996icu_996.ICU', \
        'Jackeagle_kernel_msm-3.18', 'AdrianDC_aosp_development_sony8960_q', 'docker-library_commit-warehouse'] # Exclude potential bugs in WoC

    ###################################################
    # Inputs
    ###################################################
    # Choice-specific options
    # Skill-based
    if nav_id == 'exp':
        st.header("Recommending* Projects based on Developer's Expertise")
        st.markdown("_*We use the [Skill Space](https://doi.org/10.1109/ICSE43902.2021.00094) model to recommend similar projects\
              based on a developer's knowledge of Languages and APIs._")
        
        # basic & advanced options - basic = 1 language, advanced = multiple language
        dispopt = st.radio('Expertise Input Options', options=['Show me the Standard Options', 'Show me Advanced Options'], horizontal=True, 
                             help='Advanced options let you choose multiple languages, however, the recommendation quality may suffer')

        # Arrange Input Widgets
        col1, col2, col3 = st.columns(3)
        with col1: 
            if dispopt == 'Show me the Standard Options':
                langselect = st.selectbox('Select a Programming Language for Project Recommendation:', langdict.keys(), \
                    help='Select the language you would consider for your Open Source contribution')
            else:
                langselect = st.multiselect('Select ALL the Languages you are familiar with for Project Recommendation:', langdict.keys(), \
                                help='Select ALL the languages you would consider for your Open Source contribution')
        
        with col2:
            apiselect = st.text_input('Enter ALL the Libraries/Packages/APIs you are familiar with in the chosen language(s) (separated by semicolon ";"):')

        with col3:
            no_project = st.slider('Select How Many Similar Projects you wish to see:', min_value=1, max_value=20, value=10)

    # skill transfer based
    elif nav_id == 'trans':
        st.header("Recommending* Projects based on Developer's Expertise Transfer")
        st.markdown("""*Developers might have expertise in one programming language and a number of APIs/libraries/packages in that language,
        and might want to learn/contribute to projects in the _same domain but in a different language_. Given a developer's knowledge in one language
        and set of APIs/libraries/packages in that language, we use the [Skill Space](https://doi.org/10.1109/ICSE43902.2021.00094) model to 
        identify a set of similar APIs/libraries/packages in a **second language** and also recommend a set of projects for 
        them to consider contributing to.""")

        # Arrange Input Widgets
        col1, col2, col3 = st.columns(3)
        with col1: 
            source_lang = st.selectbox('Select a Programming Language you are familiar with:', langdict.keys(),index=1)
        
        with col2:
            api1_trans = st.text_input('Enter ALL the Libraries/Packages/APIs you are familiar with in the chosen language (separated by semicolon ";"):')

        with col3:
            dest_lang = st.selectbox('Select a Programming Language you want to transfer you skills to:', langdict.keys())

        if source_lang == dest_lang: 
            st.error('ERROR! The selected languages must be different!')
        else:
            is_api = st.checkbox('Do you want to see API/library/package recommendations for the second language?')
            if is_api:
                col4, col5  = st.columns(2)
                with col4:
                    no_project = st.slider('Select How Many Similar Projects you wish to see:', min_value=1, max_value=20, value=10)
                with col5:
                    no_api = st.slider('Select How Many Similar APIs/libraries/packages you wish to see:', min_value=1, max_value=20, value=10)
            else:
                no_project = st.slider('Select How Many Similar Projects you wish to see:', min_value=1, max_value=20, value=10)
                

    # popularity based
    elif nav_id == 'pop':
        st.header('Recommending Projects based on their popularity')
        # Arrange Input Widgets
        col1, col2, col3  = st.columns(3)
        with col1:
            pop_metric  = st.radio('Select Popularity Metric to use for recommendation', ['No. of Stars', 'No. of Contributors', 
            'No. of Forks', 'Location (TimeZone)'], help='You can choose to see either projects with most no. of stars/contributors/forks, or the\
            projects popular among active developers (>100 commits) in a chosen timezone (select timezone below!)', horizontal=True)
        with col2:
            langselect = st.selectbox('Select a Programming Language for Project Recommendation:', sorted(proj_langs))
        with col3:    
            no_project = st.slider('Select How Many Projects you wish to see:', min_value=1, max_value=20, value=10)

    ###################################################
    # Adding Diversity (Gender) Filters - company email, ethnicity, elephant facor, non-binary pronoun
    ###################################################
    is_location = False
    if nav_id == 'pop':
        if pop_metric == 'Location (TimeZone)':
            is_location = True
        is_diversity = st.checkbox("Check if you want to filter results by diversity")
    else:
        col_d, col_m = st.columns(2)
        with col_d:
            is_diversity = st.checkbox("Check if you want to filter results by diversity")
        with col_m:
            is_mentor =  st.checkbox("Check if you want recommendation for potential Mentors", 
            help="Potential Mentors are chosen from core developers in the recommended projects, limited to max. 10")


    if is_diversity and is_location:
        col_f3, col_f4 = st.columns(2)
        with col_f3:
            gender_pct = st.slider('Minimum percentage of female developers in the project', min_value=0, max_value=100, value=5,
            help='Only developer names that could be identified as male/female are used for this calculation')
        with col_f4:
            data, tzoffset = show_tz()
    elif is_diversity:
        gender_pct = st.slider('Minimum percentage of female developers in the project', min_value=0, max_value=100, value=5,
            help='Only developer names that could be identified as male/female are used for this calculation')
    elif is_location:
        data, tzoffset = show_tz()
    
###################################################
# Outputs
###################################################
    go = st.button('Get Project Recommendation')
    if go:
        ###################################################
        # Output for skill-space based recommendation
        ###################################################
        if nav_id == 'exp':
            with st.spinner('Loading the Skill Space Model, this might take a few minutes ...'):
                mod = load_skill_space_model('./data/doc2vec.U.PtAlAPI_U.ep1.trained.pickle.gz') 

            dev_vect, similar_tags = recommend_project(apiselect, langselect, langdict, mod)
            coredevs = show_project_recommendation_table(similar_tags, no_project, proj_info, is_diversity, exclude, langselect, gender_pct)
            if is_mentor:
                show_mentors(dev_vect, coredevs, mod)
        ###################################################
        # Output for skill transfer based recommendation
        ###################################################
        elif nav_id == 'trans':
            with st.spinner('Loading the Skill Space Model, this might take a few minutes ...'):
                mod = load_skill_space_model('./data/doc2vec.U.PtAlAPI_U.ep1.trained.pickle.gz')
            # API & Project recommendations
            if is_api:
                dev_vect, similar_tags, similar_apis = transfer_project(source_lang, dest_lang, api1_trans, mod, langdict, no_api)
                with st.spinner('Model Loaded! Getting API Recommendations ...'):
                    col_api = ['API', 'Similarity Score']
                    row_api = []
                    for api, similarity in similar_apis:
                        row_api.append([api, "{:.2f}".format(similarity)])
                    p_api = show_table(row_api, col_api)
                    st.header(f'API Recommendation Table - Sorted by similarity (scrollable)')
                    st.bokeh_chart(p_api)
            else:
                dev_vect, similar_tags = transfer_project(source_lang, dest_lang, api1_trans, mod, langdict)

            coredevs = show_project_recommendation_table(similar_tags, no_project, proj_info, is_diversity, exclude, dest_lang, gender_pct)
            if is_mentor:
                show_mentors(dev_vect, coredevs, mod)
        ###################################################
        # Output for popularity-based recommendation
        ###################################################
        elif nav_id == 'pop':
            with st.spinner('Getting your Recommendations ...'):
                if langselect != 'ALL':
                    filtered_proj = {k:v for k,v in proj_info.items() if langselect in v['FileInfo'].keys()}
                else:
                    filtered_proj = dict(proj_info)

                if pop_metric == 'Location (TimeZone)':
                    if tzoffset not in data.keys():
                        st.error('You Need to Select a Valid TimeZone!')
                    else:
                        colnames = ['Project URL', 'Active Dev. Count at selected TZ', 'No. Stars', 'No. Forks', 'Total No. Contributors','Female Developer Percentage' ]
                        rec_table = []
                        for item in data[tzoffset]:
                            project = item[0]
                            if 'github.com' in project:
                                prj = '_'.join(project.split('/')[-2:])
                            else:
                                prj = '_'.join(project.split('/')[-3:])
                            if prj not in filtered_proj.keys() or prj in exclude:
                                continue
                            female_pct = filtered_proj[prj]['female_pct']
                            if is_diversity:
                                if female_pct >= gender_pct:
                                    rec_table.append([project, item[1]['all'], filtered_proj[prj]['NumStars'], filtered_proj[prj]['NumForks'],
                                    filtered_proj[prj]['NumAuthors'],f'{female_pct:.2f}%' ])
                            else:
                                rec_table.append([project, item[1]['all'], filtered_proj[prj]['NumStars'], filtered_proj[prj]['NumForks'],
                                    filtered_proj[prj]['NumAuthors'],f'{female_pct:.2f}%' ])
                        if len(rec_table) > no_project:
                            rec_table = rec_table[:no_project]
                        p = show_table(rec_table, colnames)
                        st.header('Project Recommendation Table - Sorted by No. of Active Developers in Selected Time Zone (scrollable)')
                        with st.expander("INFORMATION & DISCLAIMER"):
                            st.info("""The results shown here are based on World of Code (WoC) dataset version U, collected on Nov. 2021.
                        WoC uses a community-resolution based approach to calculate various statistics, so the values may differ from what you see 
                        on GitHub. Since we did not use any fork-resolution, the projects might be a fork of another project, in which case, 
                        the user is recommeded to look into the source project. Any obvious inconsistencies should be reported to WoC maintainers.""")
                            st.info('The Tables are Interactive and can be SORTED by any of the columns - just click on the column header!')
                            st.warning(
                            """We fully support people who do not identify with either of the two binary genders, 
                            the reason for showing only two genders here is because WoC only identifies people as male or female 
                            (or 'Unknown' for developers who do not provide a common name)."""
                        )
                        st.bokeh_chart(p)
                else:
                    # sort by metric
                    metric_dict = {'No. of Stars':"NumStars", 'No. of Contributors':"NumAuthors", 'No. of Forks':"NumForks"}
                    metric = metric_dict[pop_metric]
                    sorted_proj = [k for k,_ in sorted(filtered_proj.items(), key=lambda item:int(item[1][metric]), reverse=True )]
                    colnames = ['Project URL', 'No. Stars', 'No. Forks', 'Total No. Contributors','Female Developer Percentage' ]
                    rec_table = []
                    for prj in sorted_proj:
                        if prj in exclude:
                            continue
                        with st.spinner('Checking if Project URL exists ...'):
                            project = check_project_url(prj)
                            if not project:
                                continue
                        female_pct = filtered_proj[prj]['female_pct']
                        if is_diversity:
                            if female_pct >= gender_pct:
                                rec_table.append([project, filtered_proj[prj]['NumStars'], filtered_proj[prj]['NumForks'],
                                filtered_proj[prj]['NumAuthors'],f'{female_pct:.2f}%' ])
                        else:
                            rec_table.append([project, filtered_proj[prj]['NumStars'], filtered_proj[prj]['NumForks'],
                                filtered_proj[prj]['NumAuthors'],f'{female_pct:.2f}%' ])
                        if len(rec_table) >= no_project:
                            break
                    p = show_table(rec_table, colnames)
                    st.header(f'Project Recommendation Table - Sorted by {pop_metric} (scrollable)')
                    with st.expander("INFORMATION & DISCLAIMER"):
                        st.info("""The results shown here are based on World of Code (WoC) dataset version U, collected on Nov. 2021.
                        WoC uses a community-resolution based approach to calculate various statistics, so the values may differ from what you see 
                        on GitHub. Since we did not use any fork-resolution, the projects might be a fork of another project, in which case, 
                        the user is recommeded to look into the source project. Any obvious inconsistencies should be reported to WoC maintainers.""")
                        st.info('The Tables are Interactive and can be SORTED by any of the columns - just click on the column header!')
                        st.warning(
                            """We fully support people who do not identify with either of the two binary genders, 
                            the reason for showing only two genders here is because WoC only identifies people as male or female 
                            (or 'Unknown' for developers who do not provide a common name)."""
                        )
                    st.bokeh_chart(p)

########################
# Main function call
########################

if __name__ == '__main__':
    st.set_page_config(page_title='OSS Project Recommendation for Newcomers', layout="wide", initial_sidebar_state='collapsed')
    st.title('OSS Project Recommendation for Newcomers')
    show_page()