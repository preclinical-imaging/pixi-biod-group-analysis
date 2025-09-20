import os
import streamlit as st
import xnat
import json
import pandas as pd
import base64
from collections import defaultdict
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go

css='''
<style>
    section.main > div {max-width: 80%;}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

@st.cache_data
def load_data(url, user, password):
    df = pd.read_csv(url, storage_options={"Authorization": b"Basic " + base64.b64encode(f"{user}:{password}".encode())})
    return df

class App:

    def __init__(self, host=None, user=None, password=None, project_id=None):
        self._host = host or os.environ.get('XNAT_HOST')
        self._user = user or os.environ.get('XNAT_USER')
        self._password = password or os.environ.get('XNAT_PASS')
        self._project_id = project_id or (os.environ.get('XNAT_ITEM_ID') if os.environ.get('XNAT_XSI_TYPE') == 'xnat:projectData' else None)
        self._connection = xnat.connect(self._host, user=self._user, password=self._password)

        if self._project_id:
            try: 
                self._project = self._connection.projects[self._project_id]
            except Exception as e:
                raise Exception(f'Error connecting to project {self._project_id}', e)
        else:
            raise Exception('Must be started from an XNAT project.')
        
        self._init_session_state()
        self._load_bioDExperiments()
        self._init_ui()
    
    def _init_session_state(self):
        # Initialize streamlit session state
        # Values will be populated later
        if 'project' not in st.session_state:
            st.session_state.project = self._project

        if 'project_id' not in st.session_state:
            st.session_state.project_id = self._project_id

        if 'subjects' not in st.session_state:
            st.session_state.subjects = []

        if 'subject_groups' not in st.session_state:
            st.session_state.subject_groups = []

        if 'organs' not in st.session_state:
            st.session_state.organs = []

        
    def _load_bioDExperiments(self):
        allBiodSamplesInProject = []
        project_id = self._project_id

        for subject in self._project.subjects.values():
            if subject.label == 'Hotel':
                continue
            subjectUrl = '/data/projects/' + project_id + '/subjects/' + subject.label
            bioDExperimentsRequest = self._connection.get(subjectUrl +  '/experiments?format=json&xsiType=pixi:bioDistributionData')
            if not bioDExperimentsRequest.ok:
                print(f'Failed to get BioD experiments')
            bioDExpObj = bioDExperimentsRequest.json() 
            bioDExpID = bioDExpObj['ResultSet']['Result'][0]['ID']
            bioDExperimentRequest = self._connection.get(subjectUrl + '/experiment/'+ bioDExpID + '?format=json')
            if not bioDExperimentRequest.ok:
                print(f'Failed to get BioD experiment ' + bioDExpID)
            bioDExp = bioDExperimentRequest.json() 
            bioDSampleUptakes = self._extract_sample_data(bioDExp)
            for bioDSampleUptake in bioDSampleUptakes:
                allBiodSamplesInProject.append(bioDSampleUptake)
        df = pd.DataFrame(allBiodSamplesInProject)        
        self._biods_experiments = df


        # Update selectors
        st.session_state.subjects.clear()
        st.session_state.subjects.extend(df['subject_id'].unique().tolist())
        st.session_state.subjects.sort()

        st.session_state.subject_groups.clear()
        st.session_state.subject_groups.extend(df['subject_group'].unique().tolist())
        st.session_state.subject_groups.sort

        st.session_state.organs.clear()
        st.session_state.organs.extend(df['sample_type'].unique().tolist())
        st.session_state.organs.sort


    def _extract_sample_data(self, json_data):
        """
        Extract sample uptake data from the nested JSON structure.
        Returns a list of dictionaries with flattened data.
        """
        samples = []
        # Navigate through the nested structure
        for item in json_data.get('items', []):
            # Get subject group from the top level
            subject_group = item.get('data_fields', {}).get('group', 'Unknown')
            subject_id = item.get('data_fields', {}).get('label', 'Unknown')
            
            # Navigate to experiments
            for child in item.get('children', []):
                if child.get('field') == 'experiments/experiment':
                    for experiment in child.get('items', []):
                        # Navigate to sample_uptake_data
                        for exp_child in experiment.get('children', []):
                            if exp_child.get('field') == 'sample_uptake_data':
                                for sample in exp_child.get('items', []):
                                    sample_data = sample.get('data_fields', {})
                                    if sample_data:  # Only process if data exists
                                        sample_record = {
                                            'subject_group': subject_group,
                                            'subject_id': subject_id,
                                            'sample_type': sample_data.get('sample_type'),
                                            'percent_injected_dose_per_organ': sample_data.get('percent_injected_dose_per_organ'),
                                            'percent_injected_dose_per_gram': sample_data.get('percent_injected_dose_per_gram'),
                                            'sample_weight': sample_data.get('sample_weight'),
                                            'sample_weight_unit': sample_data.get('sample_weight_unit')
                                        }
                                        samples.append(sample_record)
        return samples

    
    def _init_ui(self):
        # Hide streamlit deploy button
        st.markdown("""
            <style>
                .reportview-container {
                    margin-top: -2em;
                }
                #MainMenu {visibility: hidden;}
                .stDeployButton {display:none;}
                footer {visibility: hidden;}
                #stDecoration {display:none;}
            </style>
        """, unsafe_allow_html=True)

        # Initialize UI
        self._init_sidebar()
        self._init_main()

    def _init_sidebar(self):
        # Streamlit setup
        with st.sidebar:
            st.title("BioD Summary Statistics Visualizer")
            st.markdown("*View statistical plots for BioD experiments within a project.*")
            
            with st.expander("Options", expanded=True):
                # Excluded subjects
                self._excluded_subjects = st.multiselect("Excluded Subjects", st.session_state.subjects, default=[], key='excluded_subjects', on_change=self._update_plot)

                # Excluded groups
                self._excluded_groups = st.multiselect("Excluded Groups", st.session_state.subject_groups, default=[], key='excluded_groups', on_change=self._update_plot)

                # Excluded organ
                self._excluded_organs = st.multiselect("Excluded Organs", st.session_state.organs, default=[], key='excluded_organs', on_change=self._update_plot)
                

    def _init_main(self):
        self._main = st.container()

        with self._main:

            tab1, tab2 = st.tabs([
                'All Measurements Plot',
                'Data Table',
            ])

            with tab1:
                self._tab1 = st.empty()

            with tab2:
                self._tab2 = st.empty()

        self._update_plot()
        
        
    def _update_plot(self):
        self._tab1.empty()
        with self._tab1:
            self._plot_all_measurements()


        self._tab2.empty()
        with self._tab2:
            df = self._get_filtered_data()
            st.dataframe(df, height=600)

    def _get_filtered_data(self):
        df = self._biods_experiments

        if len(self._excluded_subjects) > 0:
            df = df[~df['subject_id'].isin(self._excluded_subjects)]


        if len(self._excluded_groups) > 0:
            df = df[~df['subject_group'].isin(self._excluded_groups)]

        if len(self._excluded_organs) > 0:
            df = df[~df['sample_type'].isin(self._excluded_organs)]

        return df
    

    def _create_error_bar_chart(self, field, title):
         
         # Convert to DataFrame
         df = pd.DataFrame(self._get_filtered_data())
         
         # Filter out records with missing percent_injected_dose_per_organ
         df = df.dropna(subset=[field, 'sample_type'])
         
         if df.empty:
             st.error("No valid data found for " + field)
             return
         
         # Calculate statistics by sample_type and subject_group
         stats = df.groupby(['sample_type', 'subject_group'])[field].agg([
             'mean', 
             'std', 
             'sem',  # standard error of mean
             'count'
         ]).reset_index()
         
         # Create the error bar chart
         fig = go.Figure()
         
         # Get unique subject groups for different colors
         subject_groups = stats['subject_group'].unique()
         colors = px.colors.qualitative.Set1[:len(subject_groups)]
         
         for i, group in enumerate(subject_groups):
             group_data = stats[stats['subject_group'] == group]
             
             fig.add_trace(go.Bar(
                 name=group,
                 x=group_data['sample_type'],
                 y=group_data['mean'],
                 error_y=dict(
                     type='data',
                     array=group_data['sem'],  # Using standard error
                     visible=True
                 ),
                 marker_color=colors[i % len(colors)],
                 text=[f'n={n}' for n in group_data['count']],
                 textposition='outside'
             ))
         
         # Update layout
         fig.update_layout(
             title= title,
             xaxis_title='Sample Type',
             yaxis_title=title,
             barmode='group',
             template='plotly_white',
             showlegend=True,
             legend=dict(title='Subject Group'),
             height=600
         )
         
         return fig
     
    def _plot_all_measurements(self):
        tab1, tab2, tab3 = st.tabs(["Dose per Organ", "Dose per Gram", "Sample Weight"])
        with tab1:
           st.subheader("Percent Injected Dose per Organ")
           fig1 = self._create_error_bar_chart('percent_injected_dose_per_organ', 'Percent Injected Dose per Organ by Organ')
           if fig1:
               st.plotly_chart(fig1, use_container_width=True)
        with tab2:
           st.subheader("Percent Injected Dose per Gram")
           fig2 = self._create_error_bar_chart('percent_injected_dose_per_gram', 'Percent Injected Dose per Gram by Organ')
           if fig2:
               st.plotly_chart(fig2, use_container_width=True)
        with tab3:
           st.subheader('Sample Weight')
           fig3 = self._create_error_bar_chart('sample_weight', 'Sample Weight')
           if fig3:
               st.plotly_chart(fig3, use_container_width=True)
               
          

app = App()
