#!/usr/bin/env python3
"""
Enhanced Healthcare Analytics - GitHub Ready Version
===================================================

Professional healthcare data analysis with advanced statistical insights,
interactive visualizations, and comprehensive reporting capabilities.

Features:
- Advanced statistical analysis and hypothesis testing
- Professional visualizations with corporate styling
- Data quality assessment and anomaly detection
- Patient segmentation and clustering analysis
- Executive summary and actionable insights

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
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import warnings
import logging
from datetime import datetime
import json

# Configure logging and warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Professional color palette
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8',
    'dark': '#343a40',
    'purple': '#9467bd'
}

def load_and_preprocess_data(file_path="healthcare_dataset.csv"):
    """Load and preprocess healthcare data with comprehensive feature engineering"""
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Data preprocessing
        df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
        df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
        df['Treatment_Days'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
        
        # Feature engineering
        df['Age_Group'] = pd.cut(df['Age'], 
                               bins=[0, 18, 35, 50, 65, 100], 
                               labels=['Child', 'Young Adult', 'Adult', 'Middle Age', 'Senior'])
        
        df['Cost_Category'] = pd.cut(df['Billing Amount'], 
                                   bins=5, 
                                   labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Temporal features
        df['Month'] = df['Date of Admission'].dt.month
        df['Year'] = df['Date of Admission'].dt.year
        df['DayOfWeek'] = df['Date of Admission'].dt.dayofweek
        df['Quarter'] = df['Date of Admission'].dt.quarter
        
        # Season mapping
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Fall', 10: 'Fall', 11: 'Fall'}
        df['Season'] = df['Month'].map(season_map)
        
        # Risk scoring
        df['Risk_Score'] = (
            df['Age'] * 0.3 + 
            (df['Billing Amount'] / df['Billing Amount'].max()) * 100 * 0.4 +
            df['Admission Type'].map({'Emergency': 50, 'Urgent': 30, 'Elective': 10}) * 0.3
        )
        
        logger.info(f"Successfully loaded and processed {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def perform_data_quality_assessment(df):
    """Comprehensive data quality assessment with detailed metrics"""
    logger.info("Performing comprehensive data quality assessment")
    
    quality_metrics = {
        'completeness': {},
        'validity': {},
        'consistency': {},
        'uniqueness': {},
        'timeliness': {}
    }
    
    # Completeness analysis
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        quality_metrics['completeness'][col] = {
            'missing_count': missing_count,
            'missing_percentage': round(missing_pct, 2),
            'status': 'Excellent' if missing_pct == 0 else 'Good' if missing_pct < 5 else 'Warning' if missing_pct < 15 else 'Poor'
        }
    
    # Validity checks
    quality_metrics['validity'] = {
        'age_range': {
            'invalid_count': len(df[(df['Age'] < 0) | (df['Age'] > 120)]),
            'status': 'Good' if len(df[(df['Age'] < 0) | (df['Age'] > 120)]) == 0 else 'Poor'
        },
        'billing_amount': {
            'negative_count': len(df[df['Billing Amount'] < 0]),
            'zero_count': len(df[df['Billing Amount'] == 0]),
            'status': 'Good' if len(df[df['Billing Amount'] <= 0]) == 0 else 'Poor'
        },
        'treatment_days': {
            'negative_count': len(df[df['Treatment_Days'] < 0]),
            'zero_count': len(df[df['Treatment_Days'] == 0]),
            'status': 'Good' if len(df[df['Treatment_Days'] < 0]) == 0 else 'Poor'
        }
    }
    
    # Uniqueness analysis
    quality_metrics['uniqueness'] = {
        'duplicate_records': len(df) - len(df.drop_duplicates()),
        'duplicate_patients': len(df) - len(df['Name'].drop_duplicates()),
        'unique_hospitals': df['Hospital'].nunique(),
        'unique_doctors': df['Doctor'].nunique()
    }
    
    # Overall quality score
    completeness_score = sum([1 for col in quality_metrics['completeness'] 
                            if quality_metrics['completeness'][col]['status'] in ['Excellent', 'Good']]) / len(quality_metrics['completeness']) * 100
    
    validity_score = sum([1 for check in quality_metrics['validity'] 
                        if quality_metrics['validity'][check]['status'] == 'Good']) / len(quality_metrics['validity']) * 100
    
    overall_score = (completeness_score + validity_score) / 2
    
    quality_metrics['overall_score'] = round(overall_score, 2)
    
    return quality_metrics

def advanced_statistical_analysis(df):
    """Perform comprehensive statistical analysis with hypothesis testing"""
    logger.info("Performing advanced statistical analysis")
    
    results = {
        'descriptive_stats': {},
        'hypothesis_tests': {},
        'correlations': {},
        'regression_analysis': {}
    }
    
    # Descriptive statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results['descriptive_stats'] = df[numeric_cols].describe().to_dict()
    
    # Hypothesis testing
    # Test 1: Gender differences in billing amount
    male_billing = df[df['Gender'] == 'Male']['Billing Amount']
    female_billing = df[df['Gender'] == 'Female']['Billing Amount']
    
    # Perform t-test
    t_stat, p_value = ttest_ind(male_billing, female_billing)
    results['hypothesis_tests']['gender_billing_ttest'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'interpretation': f"{'Significant' if p_value < 0.05 else 'No significant'} difference in billing between genders"
    }
    
    # Test 2: Admission type vs Medical condition (Chi-square)
    contingency_table = pd.crosstab(df['Admission Type'], df['Medical Condition'])
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    results['hypothesis_tests']['admission_condition_chi2'] = {
        'chi2_statistic': chi2,
        'p_value': p_val,
        'degrees_of_freedom': dof,
        'significant': p_val < 0.05
    }
    
    # Correlation analysis
    correlation_matrix = df[numeric_cols].corr()
    results['correlations'] = correlation_matrix.to_dict()
    
    return results

def detect_anomalies(df):
    """Advanced anomaly detection using multiple methods"""
    logger.info("Detecting anomalies using multiple methods")
    
    anomalies = {
        'billing_outliers': {},
        'treatment_outliers': {},
        'age_outliers': {},
        'statistical_outliers': {}
    }
    
    # IQR method for billing amount
    Q1 = df['Billing Amount'].quantile(0.25)
    Q3 = df['Billing Amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    billing_outliers = df[(df['Billing Amount'] < lower_bound) | (df['Billing Amount'] > upper_bound)]
    anomalies['billing_outliers'] = {
        'count': len(billing_outliers),
        'percentage': round(len(billing_outliers) / len(df) * 100, 2),
        'total_amount': billing_outliers['Billing Amount'].sum(),
        'avg_amount': billing_outliers['Billing Amount'].mean()
    }
    
    # Z-score method for treatment days
    z_scores = np.abs(stats.zscore(df['Treatment_Days'].fillna(df['Treatment_Days'].mean())))
    treatment_outliers = df[z_scores > 3]
    anomalies['treatment_outliers'] = {
        'count': len(treatment_outliers),
        'percentage': round(len(treatment_outliers) / len(df) * 100, 2)
    }
    
    return anomalies

def patient_segmentation_analysis(df):
    """Advanced patient segmentation using clustering algorithms"""
    logger.info("Performing patient segmentation analysis")
    
    # Prepare features for clustering
    features = ['Age', 'Billing Amount', 'Treatment_Days']
    df_cluster = df[features].fillna(df[features].mean())
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_cluster)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features_scaled)
    
    # Analyze clusters
    cluster_analysis = df.groupby('Cluster').agg({
        'Age': ['mean', 'std'],
        'Billing Amount': ['mean', 'std'],
        'Treatment_Days': ['mean', 'std'],
        'Gender': lambda x: x.mode()[0] if not x.empty else 'Unknown',
        'Medical Condition': lambda x: x.mode()[0] if not x.empty else 'Unknown'
    }).round(2)
    
    return cluster_analysis, df

def create_professional_visualizations(df):
    """Create professional, publication-quality visualizations"""
    logger.info("Creating professional visualizations")
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Create comprehensive dashboard
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('HEALTHCARE ANALYTICS - EXECUTIVE DASHBOARD', fontsize=24, fontweight='bold', y=0.98)
    
    # 1. Risk Score Distribution
    plt.subplot(3, 4, 1)
    plt.hist(df['Risk_Score'], bins=30, color=COLORS['primary'], alpha=0.7, edgecolor='white')
    plt.title('Patient Risk Score Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Risk Score', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.grid(alpha=0.3)
    
    # 2. Age vs Billing Amount Scatter
    plt.subplot(3, 4, 2)
    scatter = plt.scatter(df['Age'], df['Billing Amount'], 
                         c=df['Risk_Score'], cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Risk Score')
    plt.title('Age vs Billing Amount\n(Colored by Risk Score)', fontsize=14, fontweight='bold')
    plt.xlabel('Age', fontweight='bold')
    plt.ylabel('Billing Amount ($)', fontweight='bold')
    
    # 3. Medical Condition Distribution
    plt.subplot(3, 4, 3)
    condition_counts = df['Medical Condition'].value_counts()
    plt.pie(condition_counts.values, labels=condition_counts.index, autopct='%1.1f%%',
            colors=sns.color_palette("husl", len(condition_counts)))
    plt.title('Medical Condition Distribution', fontsize=14, fontweight='bold')
    
    # 4. Seasonal Trends
    plt.subplot(3, 4, 4)
    seasonal_data = df.groupby('Season')['Billing Amount'].mean()
    bars = plt.bar(seasonal_data.index, seasonal_data.values, 
                   color=[COLORS['primary'], COLORS['success'], COLORS['warning'], COLORS['danger']])
    plt.title('Seasonal Healthcare Costs', fontsize=14, fontweight='bold')
    plt.xlabel('Season', fontweight='bold')
    plt.ylabel('Average Billing Amount ($)', fontweight='bold')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Hospital Performance Heatmap
    plt.subplot(3, 4, 5)
    hospital_metrics = df.groupby('Hospital').agg({
        'Billing Amount': 'mean',
        'Treatment_Days': 'mean',
        'Age': 'mean'
    }).head(10)
    
    sns.heatmap(hospital_metrics.T, annot=True, fmt='.0f', cmap='RdYlBu_r',
                cbar_kws={'label': 'Value'})
    plt.title('Top 10 Hospital Performance Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Hospital', fontweight='bold')
    plt.ylabel('Metrics', fontweight='bold')
    
    # 6. Insurance Provider Analysis
    plt.subplot(3, 4, 6)
    insurance_data = df.groupby('Insurance Provider')['Billing Amount'].mean().sort_values(ascending=True)
    plt.barh(range(len(insurance_data)), insurance_data.values, color=COLORS['info'])
    plt.yticks(range(len(insurance_data)), insurance_data.index)
    plt.title('Average Cost by Insurance Provider', fontsize=14, fontweight='bold')
    plt.xlabel('Average Billing Amount ($)', fontweight='bold')
    
    # 7. Treatment Duration vs Cost
    plt.subplot(3, 4, 7)
    plt.scatter(df['Treatment_Days'], df['Billing Amount'], alpha=0.5, color=COLORS['purple'])
    plt.title('Treatment Duration vs Cost', fontsize=14, fontweight='bold')
    plt.xlabel('Treatment Days', fontweight='bold')
    plt.ylabel('Billing Amount ($)', fontweight='bold')
    
    # Add trend line
    z = np.polyfit(df['Treatment_Days'].fillna(0), df['Billing Amount'], 1)
    p = np.poly1d(z)
    plt.plot(df['Treatment_Days'].fillna(0), p(df['Treatment_Days'].fillna(0)), 
             "r--", alpha=0.8, linewidth=2)
    
    # 8. Gender-based Analysis
    plt.subplot(3, 4, 8)
    gender_data = df.groupby(['Gender', 'Medical Condition'])['Billing Amount'].mean().unstack()
    gender_data.plot(kind='bar', ax=plt.gca(), color=[COLORS['primary'], COLORS['secondary']])
    plt.title('Gender-based Cost Analysis\nby Medical Condition', fontsize=14, fontweight='bold')
    plt.xlabel('Gender', fontweight='bold')
    plt.ylabel('Average Billing Amount ($)', fontweight='bold')
    plt.legend(title='Medical Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    # 9. Admission Type Trends
    plt.subplot(3, 4, 9)
    admission_trends = df.groupby(['Year', 'Admission Type']).size().unstack(fill_value=0)
    admission_trends.plot(kind='line', ax=plt.gca(), marker='o', linewidth=2)
    plt.title('Admission Type Trends Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontweight='bold')
    plt.ylabel('Number of Admissions', fontweight='bold')
    plt.legend(title='Admission Type')
    plt.grid(alpha=0.3)
    
    # 10. Age Group Analysis
    plt.subplot(3, 4, 10)
    age_group_data = df.groupby('Age_Group')['Billing Amount'].mean()
    plt.bar(age_group_data.index, age_group_data.values, color=COLORS['success'])
    plt.title('Healthcare Costs by Age Group', fontsize=14, fontweight='bold')
    plt.xlabel('Age Group', fontweight='bold')
    plt.ylabel('Average Billing Amount ($)', fontweight='bold')
    plt.xticks(rotation=45)
    
    # 11. Correlation Heatmap
    plt.subplot(3, 4, 11)
    numeric_cols = ['Age', 'Billing Amount', 'Treatment_Days', 'Risk_Score']
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    # 12. Monthly Admission Patterns
    plt.subplot(3, 4, 12)
    monthly_admissions = df.groupby('Month').size()
    plt.plot(monthly_admissions.index, monthly_admissions.values, 
             marker='o', linewidth=3, markersize=8, color=COLORS['danger'])
    plt.title('Monthly Admission Patterns', fontsize=14, fontweight='bold')
    plt.xlabel('Month', fontweight='bold')
    plt.ylabel('Number of Admissions', fontweight='bold')
    plt.grid(alpha=0.3)
    plt.xticks(range(1, 13))
    
    plt.tight_layout()
    plt.savefig('healthcare_analytics_dashboard.png', dpi=300, bbox_inches='tight')
    logger.info("Professional dashboard saved as 'healthcare_analytics_dashboard.png'")
    
    return fig

def generate_executive_summary(df, quality_metrics, statistical_results, anomalies):
    """Generate comprehensive executive summary with actionable insights"""
    logger.info("Generating executive summary")
    
    summary = {
        'overview': {
            'total_patients': len(df),
            'total_revenue': df['Billing Amount'].sum(),
            'average_cost_per_patient': df['Billing Amount'].mean(),
            'data_quality_score': quality_metrics['overall_score'],
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'key_findings': [],
        'recommendations': [],
        'risk_assessment': {},
        'financial_impact': {}
    }
    
    # Key findings
    summary['key_findings'] = [
        f"Analyzed {len(df):,} patient records with {quality_metrics['overall_score']:.1f}% data quality score",
        f"Total healthcare spending: ${df['Billing Amount'].sum():,.2f}",
        f"Identified {anomalies['billing_outliers']['count']} potential billing anomalies ({anomalies['billing_outliers']['percentage']:.1f}%)",
        f"Average treatment duration: {df['Treatment_Days'].mean():.1f} days",
        f"Most common condition: {df['Medical Condition'].mode()[0]}",
        f"Peak admission season: {df.groupby('Season')['Billing Amount'].sum().idxmax()}"
    ]
    
    # Risk assessment
    high_risk_patients = df[df['Risk_Score'] > df['Risk_Score'].quantile(0.8)]
    summary['risk_assessment'] = {
        'high_risk_count': len(high_risk_patients),
        'high_risk_percentage': len(high_risk_patients) / len(df) * 100,
        'average_risk_score': df['Risk_Score'].mean(),
        'risk_distribution': df['Risk_Score'].describe().to_dict()
    }
    
    # Financial impact
    summary['financial_impact'] = {
        'potential_fraud_savings': anomalies['billing_outliers']['total_amount'] - (df['Billing Amount'].mean() * anomalies['billing_outliers']['count']),
        'cost_optimization_opportunity': df.groupby('Medical Condition')['Billing Amount'].std().sum(),
        'seasonal_variation': df.groupby('Season')['Billing Amount'].mean().max() - df.groupby('Season')['Billing Amount'].mean().min()
    }
    
    # Strategic recommendations
    summary['recommendations'] = [
        "Implement AI-powered fraud detection system for billing anomalies",
        "Develop personalized care plans for high-risk patient segments",
        "Optimize staffing and resources based on seasonal admission patterns",
        "Standardize treatment protocols to reduce cost variation",
        "Focus preventive care initiatives on identified high-risk populations",
        "Improve data quality processes to achieve >95% completeness",
        "Establish real-time monitoring dashboards for key performance indicators"
    ]
    
    return summary

def main():
    """Main execution function"""
    print("="*80)
    print("🏥 ADVANCED HEALTHCARE ANALYTICS PLATFORM")
    print("="*80)
    print("Professional healthcare data analysis with advanced statistical insights")
    print("Version: 2.0.0 | GitHub Ready")
    print("="*80)
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()
        
        # Perform comprehensive analysis
        quality_metrics = perform_data_quality_assessment(df)
        statistical_results = advanced_statistical_analysis(df)
        anomalies = detect_anomalies(df)
        cluster_analysis, df_clustered = patient_segmentation_analysis(df)
        
        # Create visualizations
        dashboard_fig = create_professional_visualizations(df_clustered)
        
        # Generate executive summary
        executive_summary = generate_executive_summary(df_clustered, quality_metrics, statistical_results, anomalies)
        
        # Display results
        print("\n📊 EXECUTIVE SUMMARY")
        print("-" * 50)
        for finding in executive_summary['key_findings']:
            print(f"• {finding}")
        
        print("\n🎯 STRATEGIC RECOMMENDATIONS")
        print("-" * 50)
        for i, rec in enumerate(executive_summary['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\n💰 FINANCIAL IMPACT")
        print("-" * 50)
        print(f"• Potential fraud savings: ${executive_summary['financial_impact']['potential_fraud_savings']:,.2f}")
        print(f"• Cost optimization opportunity: ${executive_summary['financial_impact']['cost_optimization_opportunity']:,.2f}")
        print(f"• Seasonal cost variation: ${executive_summary['financial_impact']['seasonal_variation']:,.2f}")
        
        print("\n🔍 DATA QUALITY ASSESSMENT")
        print("-" * 50)
        print(f"• Overall quality score: {quality_metrics['overall_score']:.1f}%")
        print(f"• Duplicate records: {quality_metrics['uniqueness']['duplicate_records']}")
        print(f"• Unique hospitals: {quality_metrics['uniqueness']['unique_hospitals']}")
        print(f"• Unique doctors: {quality_metrics['uniqueness']['unique_doctors']}")
        
        # Save results
        with open('executive_summary.json', 'w') as f:
            json.dump(executive_summary, f, indent=2, default=str)
        
        with open('quality_metrics.json', 'w') as f:
            json.dump(quality_metrics, f, indent=2, default=str)
        
        print("\n✅ Analysis complete! Results saved:")
        print("• healthcare_analytics_dashboard.png - Professional visualizations")
        print("• executive_summary.json - Comprehensive summary")
        print("• quality_metrics.json - Data quality assessment")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()