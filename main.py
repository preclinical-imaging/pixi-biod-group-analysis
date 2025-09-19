import os
import streamlit as st
import xnat
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

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
        self.__load_bioDExperiments()
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
                self._excluded_organs = st.multiselect("Excluded Organs", st.session_state.organs, default=[], key='excluded_groups', on_change=self._update_plot)
                

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
    
    def _plot_all_measurements(self):
        df = self._get_filtered_data()

        fig = go.Figure()

        # Line for each subject colored by group
        colors = px.colors.qualitative.Plotly
        for i, group in enumerate(df['group'].unique()):
            group_df = df[df['group'] == group]
            for subject in group_df['subject'].unique():
                subject_df = group_df[group_df['subject'] == subject]
                fig.add_trace(go.Scatter(x=subject_df['days'], y=subject_df['volume'], mode='lines+markers', name=subject, line=dict(color=colors[i]),
                                         legendgroup=group, legendgrouptitle_text=group.capitalize()))

        fig.update_layout(
            height=600,
            title='Tumor Volume vs Time for All Subjects',
            xaxis_title='Time (days)',
            yaxis_title='Tumor Volume (mm^3)',
            legend_title='Subjects',
        )

        st.plotly_chart(fig, use_container_width=True)

    def plot_error_bars_by_sample_type_and_group(self, dpi=300):
        """
        Create error bar plots for each metric by sample_type and study_group
        
        Parameters:
        -----------
        df : DataFrame
            Input data
        save_path : str, optional
            Full path to save the plot (e.g., 'plots/error_bars.png')
            If None, plot is displayed but not saved
        save_format : str, default 'png'
            File format for saving ('png', 'pdf', 'svg', 'jpg')
        dpi : int, default 300
            Resolution for saved image
        """
        
        df = self._get_filtered_data()
        
        # Metrics to plot
        metrics = [
            ('percent_injected_dose_per_organ', '%ID/organ'),
            ('percent_injected_dose_per_gram', '%ID/gram'), 
            ('sample_weight', 'Sample Wt (g)')
        ]
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Biodistribution Data by Organ and Study Group', fontsize=16, fontweight='bold')
        
        for idx, (metric_col, metric_label) in enumerate(metrics):
            ax = axes[idx]
            
            # Calculate statistics grouped by sample_type and subject_group
            stats = df.groupby(['sample_type', 'subject_group'])[metric_col].agg(['mean', 'std']).reset_index()
            
            # Get unique sample types and study groups
            sample_types = sorted(df['sample_type'].unique())
            study_groups = sorted(df['subject_group'].unique())
            
            # Set up bar positions
            x_pos = np.arange(len(sample_types))
            width = 0.35
            
            # Plot bars for each study group
            for i, group in enumerate(study_groups):
                group_stats = stats[stats['subject_group'] == group]
                
                means = []
                stds = []
                
                for sample_type in sample_types:
                    sample_stats = group_stats[group_stats['sample_type'] == sample_type]
                    if len(sample_stats) > 0:
                        means.append(sample_stats['mean'].iloc[0])
                        stds.append(sample_stats['std'].iloc[0] if not np.isnan(sample_stats['std'].iloc[0]) else 0)
                    else:
                        means.append(0)
                        stds.append(0)
                
                # Plot bars with error bars
                bars = ax.bar(x_pos + i * width, means, width, 
                             yerr=stds, capsize=5, 
                             label=group, alpha=0.8)
                
                # Add value labels on bars
                for j, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
                    if mean_val > 0:  # Only label non-zero bars
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + std_val,
                               f'{mean_val:.2f}', ha='center', va='bottom', fontsize=8)
            
            # Customize the plot
            ax.set_xlabel('Organ')
            ax.set_ylabel(metric_label)
            ax.set_title(f'{metric_label} by Organ')
            ax.set_xticks(x_pos + width/2)
            ax.set_xticklabels(sample_types, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot if path is provided
        if save_path:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Save with specified format and DPI
            plt.savefig(save_path, format=save_format, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        return fig        
        
app = App()
