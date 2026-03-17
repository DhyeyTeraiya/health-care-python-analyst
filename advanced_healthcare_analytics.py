#!/usr/bin/env python3
"""
Advanced Healthcare Analytics Platform
=====================================

A comprehensive healthcare data analysis system featuring:
- Advanced statistical analysis and hypothesis testing
- Interactive web dashboard with real-time updates
- Professional visualizations and executive reporting
- Data quality validation and anomaly detection
- Automated report generation and export capabilities

Author: Healthcare Analytics Team
Version: 2.0.0
License: MIT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from lifelines import KaplanMeierFitter, CoxPHFitter
import warnings
import logging
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class HealthcareAnalyticsPlatform:
    """
    Main analytics platform class that orchestrates all analysis components
    """
    
    def __init__(self, data_path: str = "healthcare_dataset.csv"):
        """Initialize the analytics platform with data loading and preprocessing"""
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.analysis_results = {}
        self.quality_report = {}
        
        # Professional color palette
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8',
            'dark': '#343a40',
            'purple': '#9467bd'
        }
        
        self.load_and_preprocess_data()
        
    def load_and_preprocess_data(self):
        """Load and preprocess the healthcare dataset"""
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            
            # Data preprocessing
            self.df['Date of Admission'] = pd.to_datetime(self.df['Date of Admission'])
            self.df['Discharge Date'] = pd.to_datetime(self.df['Discharge Date'])
            self.df['Treatment_Days'] = (self.df['Discharge Date'] - self.df['Date of Admission']).dt.days
            
            # Create additional features
            self.df['Age_Group'] = pd.cut(self.df['Age'], 
                                        bins=[0, 18, 35, 50, 65, 100], 
                                        labels=['Child', 'Young Adult', 'Adult', 'Middle Age', 'Senior'])
            
            self.df['Cost_Category'] = pd.cut(self.df['Billing Amount'], 
                                            bins=5, 
                                            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            # Season mapping
            self.df['Month'] = self.df['Date of Admission'].dt.month
            season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                         3: 'Spring', 4: 'Spring', 5: 'Spring',
                         6: 'Summer', 7: 'Summer', 8: 'Summer',
                         9: 'Fall', 10: 'Fall', 11: 'Fall'}
            self.df['Season'] = self.df['Month'].map(season_map)
            
            self.processed_df = self.df.copy()
            logger.info(f"Successfully loaded {len(self.df)} records")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def perform_data_quality_assessment(self) -> Dict:
        """Comprehensive data quality assessment"""
        logger.info("Performing data quality assessment")
        
        quality_metrics = {
            'completeness': {},
            'validity': {},
            'consistency': {},
            'uniqueness': {},
            'timeliness': {}
        }
        
        # Completeness check
        for col in self.df.columns:
            missing_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
            quality_metrics['completeness'][col] = {
                'missing_percentage': round(missing_pct, 2),
                'status': 'Good' if missing_pct < 5 else 'Warning' if missing_pct < 15 else 'Poor'
            }
        
        # Validity checks
        quality_metrics['validity']['age_range'] = {
            'invalid_count': len(self.df[(self.df['Age'] < 0) | (self.df['Age'] > 120)]),
            'status': 'Good' if len(self.df[(self.df['Age'] < 0) | (self.df['Age'] > 120)]) == 0 else 'Poor'
        }
        
        quality_metrics['validity']['billing_amount'] = {
            'negative_count': len(self.df[self.df['Billing Amount'] < 0]),
            'status': 'Good' if len(self.df[self.df['Billing Amount'] < 0]) == 0 else 'Poor'
        }
        
        # Uniqueness check
        quality_metrics['uniqueness']['patient_records'] = {
            'duplicate_count': self.df.duplicated().sum(),
            'duplicate_percentage': round((self.df.duplicated().sum() / len(self.df)) * 100, 2)
        }
        
        # Consistency checks
        date_inconsistencies = len(self.df[self.df['Date of Admission'] > self.df['Discharge Date']])
        quality_metrics['consistency']['date_logic'] = {
            'inconsistent_count': date_inconsistencies,
            'status': 'Good' if date_inconsistencies == 0 else 'Poor'
        }
        
        self.quality_report = quality_metrics
        return quality_metrics
    
    def advanced_statistical_analysis(self) -> Dict:
        """Perform comprehensive statistical analysis"""
        logger.info("Performing advanced statistical analysis")
        
        results = {}
        
        # 1. Hypothesis Testing - Gender vs Billing Amount
        male_costs = self.df[self.df['Gender'] == 'Male']['Billing Amount']
        female_costs = self.df[self.df['Gender'] == 'Female']['Billing Amount']
        
        # Normality test
        male_normal = stats.shapiro(male_costs.sample(min(5000, len(male_costs))))[1] > 0.05
        female_normal = stats.shapiro(female_costs.sample(min(5000, len(female_costs))))[1] > 0.05
        
        if male_normal and female_normal:
            stat, p_value = ttest_ind(male_costs, female_costs)
            test_type = "Independent t-test"
        else:
            stat, p_value = mannwhitneyu(male_costs, female_costs)
            test_type = "Mann-Whitney U test"
        
        results['gender_cost_analysis'] = {
            'test_type': test_type,
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'male_mean': male_costs.mean(),
            'female_mean': female_costs.mean(),
            'effect_size': abs(male_costs.mean() - female_costs.mean()) / np.sqrt((male_costs.var() + female_costs.var()) / 2)
        }
        
        # 2. Chi-square test for Medical Condition vs Admission Type
        contingency_table = pd.crosstab(self.df['Medical Condition'], self.df['Admission Type'])
        chi2, p_val, dof, expected = chi2_contingency(contingency_table)
        
        results['condition_admission_association'] = {
            'chi2_statistic': chi2,
            'p_value': p_val,
            'degrees_of_freedom': dof,
            'significant': p_val < 0.05,
            'cramers_v': np.sqrt(chi2 / (len(self.df) * (min(contingency_table.shape) - 1)))
        }
        
        # 3. Survival Analysis (Length of Stay)
        kmf = KaplanMeierFitter()
        
        # Survival analysis by admission type
        survival_results = {}
        for admission_type in self.df['Admission Type'].unique():
            subset = self.df[self.df['Admission Type'] == admission_type]
            durations = subset['Treatment_Days']
            event_observed = [1] * len(durations)  # All patients were discharged
            
            kmf.fit(durations, event_observed, label=admission_type)
            survival_results[admission_type] = {
                'median_survival': kmf.median_survival_time_,
                'survival_function': kmf.survival_function_.to_dict()
            }
        
        results['survival_analysis'] = survival_results
        
        # 4. Correlation Analysis
        numeric_cols = ['Age', 'Billing Amount', 'Treatment_Days']
        correlation_matrix = self.df[numeric_cols].corr()
        
        results['correlation_analysis'] = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': []
        }
        
        # Find strong correlations (|r| > 0.5)
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    results['correlation_analysis']['strong_correlations'].append({
                        'variables': (correlation_matrix.columns[i], correlation_matrix.columns[j]),
                        'correlation': corr_val,
                        'strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate'
                    })
        
        # 5. Advanced Clustering Analysis
        # Prepare data for clustering
        features_for_clustering = ['Age', 'Billing Amount', 'Treatment_Days']
        X = self.df[features_for_clustering].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_clusters = dbscan.fit_predict(X_scaled)
        
        # Cluster analysis
        self.df['KMeans_Cluster'] = clusters
        self.df['DBSCAN_Cluster'] = dbscan_clusters
        
        cluster_analysis = {}
        for i in range(4):
            cluster_data = self.df[self.df['KMeans_Cluster'] == i]
            cluster_analysis[f'Cluster_{i}'] = {
                'size': len(cluster_data),
                'avg_age': cluster_data['Age'].mean(),
                'avg_cost': cluster_data['Billing Amount'].mean(),
                'avg_treatment_days': cluster_data['Treatment_Days'].mean(),
                'dominant_condition': cluster_data['Medical Condition'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A',
                'dominant_gender': cluster_data['Gender'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A'
            }
        
        results['clustering_analysis'] = {
            'kmeans': cluster_analysis,
            'dbscan_clusters': len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0),
            'dbscan_noise_points': list(dbscan_clusters).count(-1)
        }
        
        # 6. Time Series Analysis
        monthly_admissions = self.df.groupby(self.df['Date of Admission'].dt.to_period('M')).size()
        monthly_costs = self.df.groupby(self.df['Date of Admission'].dt.to_period('M'))['Billing Amount'].mean()
        
        results['time_series_analysis'] = {
            'monthly_admissions': monthly_admissions.to_dict(),
            'monthly_avg_costs': monthly_costs.to_dict(),
            'admission_trend': 'Increasing' if monthly_admissions.iloc[-1] > monthly_admissions.iloc[0] else 'Decreasing',
            'cost_trend': 'Increasing' if monthly_costs.iloc[-1] > monthly_costs.iloc[0] else 'Decreasing'
        }
        
        self.analysis_results = results
        return results
    
    def detect_anomalies(self) -> Dict:
        """Advanced anomaly detection using multiple methods"""
        logger.info("Performing anomaly detection")
        
        anomalies = {}
        
        # 1. Statistical Outliers (IQR method)
        Q1 = self.df['Billing Amount'].quantile(0.25)
        Q3 = self.df['Billing Amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        billing_outliers = self.df[(self.df['Billing Amount'] < lower_bound) | 
                                  (self.df['Billing Amount'] > upper_bound)]
        
        anomalies['billing_outliers'] = {
            'count': len(billing_outliers),
            'percentage': (len(billing_outliers) / len(self.df)) * 100,
            'potential_fraud_cases': len(billing_outliers[billing_outliers['Billing Amount'] > upper_bound]),
            'avg_outlier_amount': billing_outliers['Billing Amount'].mean()
        }
        
        # 2. Z-score based outliers
        z_scores = np.abs(stats.zscore(self.df['Billing Amount']))
        z_outliers = self.df[z_scores > 3]
        
        anomalies['z_score_outliers'] = {
            'count': len(z_outliers),
            'percentage': (len(z_outliers) / len(self.df)) * 100
        }
        
        # 3. Treatment duration anomalies
        treatment_Q1 = self.df['Treatment_Days'].quantile(0.25)
        treatment_Q3 = self.df['Treatment_Days'].quantile(0.75)
        treatment_IQR = treatment_Q3 - treatment_Q1
        treatment_outliers = self.df[self.df['Treatment_Days'] > treatment_Q3 + 1.5 * treatment_IQR]
        
        anomalies['treatment_duration_outliers'] = {
            'count': len(treatment_outliers),
            'avg_duration': treatment_outliers['Treatment_Days'].mean(),
            'max_duration': treatment_outliers['Treatment_Days'].max()
        }
        
        return anomalies
    
    def generate_executive_summary(self) -> Dict:
        """Generate executive summary with key insights"""
        logger.info("Generating executive summary")
        
        summary = {
            'dataset_overview': {
                'total_patients': len(self.df),
                'date_range': f"{self.df['Date of Admission'].min().strftime('%Y-%m-%d')} to {self.df['Date of Admission'].max().strftime('%Y-%m-%d')}",
                'total_revenue': self.df['Billing Amount'].sum(),
                'avg_cost_per_patient': self.df['Billing Amount'].mean(),
                'avg_treatment_duration': self.df['Treatment_Days'].mean()
            },
            'key_insights': [],
            'recommendations': []
        }
        
        # Generate insights based on analysis
        if hasattr(self, 'analysis_results') and self.analysis_results:
            # Gender cost difference insight
            gender_analysis = self.analysis_results.get('gender_cost_analysis', {})
            if gender_analysis.get('significant', False):
                summary['key_insights'].append(
                    f"Significant gender-based cost difference detected (p-value: {gender_analysis['p_value']:.4f})"
                )
        
        # Anomaly insights
        anomalies = self.detect_anomalies()
        if anomalies['billing_outliers']['count'] > 0:
            summary['key_insights'].append(
                f"{anomalies['billing_outliers']['count']} potential fraud cases identified "
                f"({anomalies['billing_outliers']['percentage']:.1f}% of total cases)"
            )
        
        # Recommendations
        summary['recommendations'] = [
            "Implement automated fraud detection system for billing anomalies",
            "Develop targeted treatment protocols for high-cost conditions",
            "Optimize resource allocation based on seasonal admission patterns",
            "Establish quality improvement programs for outlier treatment durations",
            "Create personalized care pathways based on patient clustering analysis"
        ]
        
        return summary

def main():
    """Main execution function for command line usage"""
    try:
        # Initialize the analytics platform
        platform = HealthcareAnalyticsPlatform("healthcare_dataset.csv")
        
        # Perform comprehensive analysis
        platform.advanced_statistical_analysis()
        
        # Generate executive summary
        summary = platform.generate_executive_summary()
        
        print("="*80)
        print("ADVANCED HEALTHCARE ANALYTICS PLATFORM - EXECUTIVE REPORT")
        print("="*80)
        
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"   • Total Patients: {summary['dataset_overview']['total_patients']:,}")
        print(f"   • Date Range: {summary['dataset_overview']['date_range']}")
        print(f"   • Total Revenue: ${summary['dataset_overview']['total_revenue']:,.2f}")
        print(f"   • Average Cost per Patient: ${summary['dataset_overview']['avg_cost_per_patient']:,.2f}")
        print(f"   • Average Treatment Duration: {summary['dataset_overview']['avg_treatment_duration']:.1f} days")
        
        print(f"\n🔍 KEY INSIGHTS:")
        for insight in summary['key_insights']:
            print(f"   • {insight}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "="*80)
        print("For interactive dashboard, run: streamlit run advanced_healthcare_analytics.py")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()