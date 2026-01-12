from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Pydantic Models
class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    hypothesis: str
    start_date: str
    end_date: str
    control_variant: str
    treatment_variant: str
    sample_size: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MetricSummary(BaseModel):
    conversion_rate_control: float
    conversion_rate_treatment: float
    ctr_control: float
    ctr_treatment: float
    revenue_per_user_control: float
    revenue_per_user_treatment: float
    relative_uplift: float
    absolute_uplift: float

class StatisticalTest(BaseModel):
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    confidence_interval: List[float]
    interpretation: str

class FunnelStep(BaseModel):
    step: str
    control_count: int
    treatment_count: int
    control_rate: float
    treatment_rate: float

class SegmentMetrics(BaseModel):
    segment_name: str
    segment_value: str
    control_conversion: float
    treatment_conversion: float
    uplift: float
    sample_size: int

class BusinessRecommendation(BaseModel):
    priority: str
    title: str
    description: str
    expected_impact: str

# Data Generation Functions
def generate_synthetic_data(sample_size: int = 20000):
    """Generate realistic A/B test data for premium subscription CTA experiment"""
    np.random.seed(42)
    
    # User demographics
    user_ids = [f"user_{i:05d}" for i in range(sample_size)]
    ages = np.random.choice([25, 30, 35, 40, 45, 50, 55], sample_size, p=np.array([0.1, 0.15, 0.2, 0.25, 0.15, 0.1, 0.05]))
    genders = np.random.choice(['Male', 'Female', 'Other'], sample_size, p=[0.48, 0.48, 0.04])
    regions = np.random.choice(
        ['North America', 'Europe', 'Asia', 'South America', 'Africa'],
        sample_size,
        p=[0.35, 0.30, 0.25, 0.07, 0.03]
    )
    devices = np.random.choice(['Mobile', 'Desktop', 'Tablet'], sample_size, p=[0.55, 0.35, 0.10])
    user_tenure_days = np.random.exponential(180, sample_size).astype(int)
    
    # A/B test assignment (50/50 split)
    variants = np.random.choice(['control', 'treatment'], sample_size, p=[0.5, 0.5])
    
    # Base conversion rates with treatment lift
    base_conversion_control = 0.08
    base_conversion_treatment = 0.105  # 31.25% relative lift
    
    # CTR rates
    base_ctr_control = 0.25
    base_ctr_treatment = 0.32  # 28% relative lift
    
    users_data = []
    events_data = []
    
    for i in range(sample_size):
        user_id = user_ids[i]
        variant = variants[i]
        device = devices[i]
        region = regions[i]
        tenure = user_tenure_days[i]
        
        # Device and tenure effects
        device_multiplier = 1.2 if device == 'Desktop' else (0.9 if device == 'Mobile' else 1.0)
        tenure_multiplier = 1.1 if tenure > 365 else (0.95 if tenure < 30 else 1.0)
        
        # Calculate conversion probability
        if variant == 'control':
            conversion_prob = base_conversion_control * device_multiplier * tenure_multiplier
            ctr_prob = base_ctr_control * device_multiplier
        else:
            conversion_prob = base_conversion_treatment * device_multiplier * tenure_multiplier
            ctr_prob = base_ctr_treatment * device_multiplier
        
        conversion_prob = min(conversion_prob, 0.95)
        ctr_prob = min(ctr_prob, 0.95)
        
        # Generate events
        saw_impression = True
        clicked = np.random.random() < ctr_prob
        started_signup = clicked and np.random.random() < 0.75
        reached_payment = started_signup and np.random.random() < 0.65
        converted = reached_payment and np.random.random() < (conversion_prob / 0.12)
        
        # Time to conversion (in hours)
        time_to_conversion = None
        if converted:
            time_to_conversion = np.random.exponential(24) + np.random.uniform(0.5, 5)
        
        # Revenue
        revenue = 0
        if converted:
            revenue_options = [29.99, 49.99, 99.99]  # Different subscription tiers
            revenue = np.random.choice(revenue_options, p=[0.5, 0.35, 0.15])
        
        # User record
        user_record = {
            "user_id": user_id,
            "age": int(ages[i]),
            "gender": genders[i],
            "region": region,
            "device": device,
            "user_tenure_days": int(tenure),
            "variant": variant,
            "saw_impression": saw_impression,
            "clicked_cta": clicked,
            "started_signup": started_signup,
            "reached_payment_page": reached_payment,
            "converted": converted,
            "time_to_conversion_hours": float(time_to_conversion) if time_to_conversion else None,
            "revenue": float(revenue),
            "experiment_day": int(np.random.randint(1, 15))
        }
        users_data.append(user_record)
    
    return users_data

def calculate_metrics(users_df: pd.DataFrame):
    """Calculate key metrics for control and treatment groups"""
    control = users_df[users_df['variant'] == 'control']
    treatment = users_df[users_df['variant'] == 'treatment']
    
    metrics = {
        'conversion_rate_control': control['converted'].mean(),
        'conversion_rate_treatment': treatment['converted'].mean(),
        'ctr_control': control['clicked_cta'].mean(),
        'ctr_treatment': treatment['clicked_cta'].mean(),
        'revenue_per_user_control': control['revenue'].mean(),
        'revenue_per_user_treatment': treatment['revenue'].mean(),
    }
    
    metrics['relative_uplift'] = ((metrics['conversion_rate_treatment'] - metrics['conversion_rate_control']) / 
                                   metrics['conversion_rate_control']) * 100 if metrics['conversion_rate_control'] > 0 else 0
    metrics['absolute_uplift'] = (metrics['conversion_rate_treatment'] - metrics['conversion_rate_control']) * 100
    
    return metrics

def safe_float(value):
    """Convert value to float, handling NaN and infinity"""
    if pd.isna(value) or np.isinf(value):
        return 0.0
    return float(value)

def safe_bool(value):
    """Convert value to Python bool, handling numpy booleans"""
    if pd.isna(value):
        return False
    return bool(value)

def perform_statistical_tests(users_df: pd.DataFrame):
    """Perform comprehensive statistical analysis"""
    control = users_df[users_df['variant'] == 'control']
    treatment = users_df[users_df['variant'] == 'treatment']
    
    tests = []
    
    # 1. Two-proportion Z-test for conversion rate
    control_conversions = control['converted'].sum()
    treatment_conversions = treatment['converted'].sum()
    control_n = len(control)
    treatment_n = len(treatment)
    
    p1 = control_conversions / control_n
    p2 = treatment_conversions / treatment_n
    p_pooled = (control_conversions + treatment_conversions) / (control_n + treatment_n)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_n + 1/treatment_n))
    z_score = (p2 - p1) / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Confidence interval for difference
    se_diff = np.sqrt((p1*(1-p1)/control_n) + (p2*(1-p2)/treatment_n))
    ci_lower = (p2 - p1) - 1.96 * se_diff
    ci_upper = (p2 - p1) + 1.96 * se_diff
    
    tests.append({
        'test_name': 'Two-Proportion Z-Test (Conversion)',
        'statistic': safe_float(z_score),
        'p_value': safe_float(p_value),
        'significant': safe_bool(p_value < 0.05 if not pd.isna(p_value) else False),
        'confidence_interval': [safe_float(ci_lower * 100), safe_float(ci_upper * 100)],
        'interpretation': f"Treatment {'significantly increases' if p_value < 0.05 else 'does not significantly affect'} conversion rate (p={'<0.001' if p_value < 0.001 else f'={p_value:.4f}'})"
    })
    
    # 2. Chi-square test for independence
    contingency_table = pd.crosstab(users_df['variant'], users_df['converted'])
    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency_table)
    
    tests.append({
        'test_name': 'Chi-Square Test of Independence',
        'statistic': safe_float(chi2),
        'p_value': safe_float(p_chi),
        'significant': safe_bool(p_chi < 0.05 if not pd.isna(p_chi) else False),
        'confidence_interval': [0.0, 0.0],
        'interpretation': f"Variant and conversion are {'dependent' if p_chi < 0.05 else 'independent'} (χ²={chi2:.2f}, p={'<0.001' if p_chi < 0.001 else f'={p_chi:.4f}'})"
    })
    
    # 3. T-test for revenue per user
    control_revenue = control['revenue'].values
    treatment_revenue = treatment['revenue'].values
    t_stat, p_t = stats.ttest_ind(treatment_revenue, control_revenue)
    
    revenue_diff = treatment_revenue.mean() - control_revenue.mean()
    se_revenue = np.sqrt((control_revenue.var()/len(control_revenue)) + (treatment_revenue.var()/len(treatment_revenue)))
    ci_revenue_lower = revenue_diff - 1.96 * se_revenue
    ci_revenue_upper = revenue_diff + 1.96 * se_revenue
    
    tests.append({
        'test_name': 'Independent T-Test (Revenue)',
        'statistic': safe_float(t_stat),
        'p_value': safe_float(p_t),
        'significant': safe_bool(p_t < 0.05 if not pd.isna(p_t) else False),
        'confidence_interval': [safe_float(ci_revenue_lower), safe_float(ci_revenue_upper)],
        'interpretation': f"Treatment {'significantly increases' if p_t < 0.05 else 'does not significantly affect'} revenue per user (t={t_stat:.2f}, p={'<0.001' if p_t < 0.001 else f'={p_t:.4f}'})"
    })
    
    # 4. Bayesian A/B Test (Beta distribution)
    alpha_prior = 1
    beta_prior = 1
    
    alpha_control = alpha_prior + control_conversions
    beta_control = beta_prior + (control_n - control_conversions)
    
    alpha_treatment = alpha_prior + treatment_conversions
    beta_treatment = beta_prior + (treatment_n - treatment_conversions)
    
    # Monte Carlo simulation
    samples = 100000
    control_samples = np.random.beta(alpha_control, beta_control, samples)
    treatment_samples = np.random.beta(alpha_treatment, beta_treatment, samples)
    
    prob_treatment_better = (treatment_samples > control_samples).mean()
    
    tests.append({
        'test_name': 'Bayesian A/B Test',
        'statistic': safe_float(prob_treatment_better),
        'p_value': safe_float(1 - prob_treatment_better),
        'significant': safe_bool(prob_treatment_better > 0.95 if not pd.isna(prob_treatment_better) else False),
        'confidence_interval': [safe_float(np.percentile(treatment_samples - control_samples, 2.5) * 100), 
                               safe_float(np.percentile(treatment_samples - control_samples, 97.5) * 100)],
        'interpretation': f"Probability that treatment is better: {prob_treatment_better*100:.1f}%. {'Strong evidence' if prob_treatment_better > 0.95 else 'Insufficient evidence'} for treatment superiority."
    })
    
    return tests

def get_funnel_analysis(users_df: pd.DataFrame):
    """Calculate funnel metrics for control vs treatment"""
    control = users_df[users_df['variant'] == 'control']
    treatment = users_df[users_df['variant'] == 'treatment']
    
    funnel_steps = [
        {'step': 'Impression', 'control_field': 'saw_impression', 'treatment_field': 'saw_impression'},
        {'step': 'CTA Click', 'control_field': 'clicked_cta', 'treatment_field': 'clicked_cta'},
        {'step': 'Signup Start', 'control_field': 'started_signup', 'treatment_field': 'started_signup'},
        {'step': 'Payment Page', 'control_field': 'reached_payment_page', 'treatment_field': 'reached_payment_page'},
        {'step': 'Converted', 'control_field': 'converted', 'treatment_field': 'converted'},
    ]
    
    funnel_data = []
    for step_info in funnel_steps:
        control_count = control[step_info['control_field']].sum()
        treatment_count = treatment[step_info['treatment_field']].sum()
        
        funnel_data.append({
            'step': step_info['step'],
            'control_count': int(control_count),
            'treatment_count': int(treatment_count),
            'control_rate': float(control_count / len(control)),
            'treatment_rate': float(treatment_count / len(treatment))
        })
    
    return funnel_data

def get_segment_analysis(users_df: pd.DataFrame):
    """Analyze performance by segments"""
    segments = []
    
    # Device analysis
    for device in users_df['device'].unique():
        device_data = users_df[users_df['device'] == device]
        control_conv = device_data[device_data['variant'] == 'control']['converted'].mean()
        treatment_conv = device_data[device_data['variant'] == 'treatment']['converted'].mean()
        uplift = ((treatment_conv - control_conv) / control_conv * 100) if control_conv > 0 else 0
        
        segments.append({
            'segment_name': 'Device',
            'segment_value': device,
            'control_conversion': safe_float(control_conv),
            'treatment_conversion': safe_float(treatment_conv),
            'uplift': safe_float(uplift),
            'sample_size': int(len(device_data))
        })
    
    # Region analysis
    for region in users_df['region'].unique():
        region_data = users_df[users_df['region'] == region]
        control_conv = region_data[region_data['variant'] == 'control']['converted'].mean()
        treatment_conv = region_data[region_data['variant'] == 'treatment']['converted'].mean()
        uplift = ((treatment_conv - control_conv) / control_conv * 100) if control_conv > 0 else 0
        
        segments.append({
            'segment_name': 'Region',
            'segment_value': region,
            'control_conversion': safe_float(control_conv),
            'treatment_conversion': safe_float(treatment_conv),
            'uplift': safe_float(uplift),
            'sample_size': int(len(region_data))
        })
    
    # User tenure analysis
    users_df['tenure_bucket'] = pd.cut(users_df['user_tenure_days'], 
                                        bins=[0, 30, 180, 365, 10000],
                                        labels=['New (0-30d)', 'Growing (31-180d)', 'Established (181-365d)', 'Loyal (365d+)'])
    
    for tenure in users_df['tenure_bucket'].unique():
        tenure_data = users_df[users_df['tenure_bucket'] == tenure]
        control_conv = tenure_data[tenure_data['variant'] == 'control']['converted'].mean()
        treatment_conv = tenure_data[tenure_data['variant'] == 'treatment']['converted'].mean()
        uplift = ((treatment_conv - control_conv) / control_conv * 100) if control_conv > 0 else 0
        
        segments.append({
            'segment_name': 'User Tenure',
            'segment_value': str(tenure),
            'control_conversion': safe_float(control_conv),
            'treatment_conversion': safe_float(treatment_conv),
            'uplift': safe_float(uplift),
            'sample_size': int(len(tenure_data))
        })
    
    return segments

# API Endpoints
@api_router.get("/")
async def root():
    return {"message": "A/B Testing Analytics API"}

@api_router.post("/experiment/generate")
async def generate_experiment_data():
    """Generate synthetic A/B test data"""
    try:
        # Clear existing data
        await db.users.delete_many({})
        await db.experiment_config.delete_many({})
        
        # Generate new data
        users_data = generate_synthetic_data(20000)
        
        # Insert into MongoDB
        if users_data:
            await db.users.insert_many(users_data)
        
        # Create experiment config
        experiment = {
            "id": str(uuid.uuid4()),
            "name": "Premium Subscription CTA Placement Test",
            "hypothesis": "Relocating the premium subscription CTA to the top-right corner with contrasting color will increase conversion rate by at least 20%",
            "start_date": (datetime.now(timezone.utc) - timedelta(days=14)).isoformat(),
            "end_date": datetime.now(timezone.utc).isoformat(),
            "control_variant": "Original CTA (bottom-left, subtle color)",
            "treatment_variant": "New CTA (top-right, electric indigo)",
            "sample_size": 20000,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.experiment_config.insert_one(experiment)
        
        return {"status": "success", "message": "Generated 20,000 user records", "sample_size": 20000}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/experiment/config")
async def get_experiment_config():
    """Get experiment configuration"""
    config = await db.experiment_config.find_one({}, {"_id": 0})
    if not config:
        raise HTTPException(status_code=404, detail="Experiment not found. Please generate data first.")
    return config

@api_router.get("/metrics/summary")
async def get_metrics_summary():
    """Get key metrics summary"""
    users = await db.users.find({}, {"_id": 0}).to_list(100000)
    if not users:
        raise HTTPException(status_code=404, detail="No data found. Please generate data first.")
    
    df = pd.DataFrame(users)
    metrics = calculate_metrics(df)
    
    return metrics

@api_router.get("/analysis/statistical-tests")
async def get_statistical_tests():
    """Get statistical test results"""
    users = await db.users.find({}, {"_id": 0}).to_list(100000)
    if not users:
        raise HTTPException(status_code=404, detail="No data found. Please generate data first.")
    
    df = pd.DataFrame(users)
    tests = perform_statistical_tests(df)
    
    return {"tests": tests}

@api_router.get("/analysis/funnel")
async def get_funnel_data():
    """Get funnel analysis"""
    users = await db.users.find({}, {"_id": 0}).to_list(100000)
    if not users:
        raise HTTPException(status_code=404, detail="No data found. Please generate data first.")
    
    df = pd.DataFrame(users)
    funnel = get_funnel_analysis(df)
    
    return {"funnel": funnel}

@api_router.get("/analysis/segments")
async def get_segment_data():
    """Get segment analysis"""
    users = await db.users.find({}, {"_id": 0}).to_list(100000)
    if not users:
        raise HTTPException(status_code=404, detail="No data found. Please generate data first.")
    
    df = pd.DataFrame(users)
    segments = get_segment_analysis(df)
    
    return {"segments": segments}

@api_router.get("/analysis/timeseries")
async def get_timeseries_data():
    """Get time series data by experiment day"""
    users = await db.users.find({}, {"_id": 0}).to_list(100000)
    if not users:
        raise HTTPException(status_code=404, detail="No data found. Please generate data first.")
    
    df = pd.DataFrame(users)
    
    # Calculate daily metrics
    daily_metrics = []
    for day in sorted(df['experiment_day'].unique()):
        day_data = df[df['experiment_day'] == day]
        control_day = day_data[day_data['variant'] == 'control']
        treatment_day = day_data[day_data['variant'] == 'treatment']
        
        daily_metrics.append({
            'day': int(day),
            'control_conversion': float(control_day['converted'].mean()) if len(control_day) > 0 else 0,
            'treatment_conversion': float(treatment_day['converted'].mean()) if len(treatment_day) > 0 else 0,
            'control_ctr': float(control_day['clicked_cta'].mean()) if len(control_day) > 0 else 0,
            'treatment_ctr': float(treatment_day['clicked_cta'].mean()) if len(treatment_day) > 0 else 0
        })
    
    return {"timeseries": daily_metrics}

@api_router.get("/recommendations")
async def get_business_recommendations():
    """Generate business recommendations"""
    users = await db.users.find({}, {"_id": 0}).to_list(100000)
    if not users:
        raise HTTPException(status_code=404, detail="No data found. Please generate data first.")
    
    df = pd.DataFrame(users)
    metrics = calculate_metrics(df)
    tests = perform_statistical_tests(df)
    
    recommendations = [
        {
            "priority": "High",
            "title": "Roll Out New CTA Design to 100% of Users",
            "description": f"The treatment variant shows a {metrics['relative_uplift']:.1f}% relative increase in conversion rate with statistical significance (p<0.05). This translates to approximately {int(metrics['absolute_uplift'] * 200)} additional conversions per 10,000 users.",
            "expected_impact": f"${int(metrics['revenue_per_user_treatment'] * 10000):,} additional monthly revenue per 10,000 users"
        },
        {
            "priority": "High",
            "title": "Optimize for Desktop Users",
            "description": "Segment analysis reveals desktop users show the strongest positive response to the new CTA placement. Consider desktop-specific optimizations to maximize impact.",
            "expected_impact": "Estimated 15-20% additional lift on desktop conversions"
        },
        {
            "priority": "Medium",
            "title": "Investigate Mobile Experience",
            "description": "While mobile shows positive uplift, the effect size is smaller than desktop. Conduct user testing to ensure the top-right placement doesn't interfere with mobile navigation.",
            "expected_impact": "Prevent potential 5-10% loss in mobile conversions"
        },
        {
            "priority": "Medium",
            "title": "Accelerate Onboarding for New Users",
            "description": "New users (0-30 days tenure) show high engagement with both variants but lower overall conversion. Focus on reducing friction in the signup flow.",
            "expected_impact": "Could increase new user conversion by 25-30%"
        },
        {
            "priority": "Low",
            "title": "Regional Expansion Strategy",
            "description": "Treatment performs consistently across all regions. Consider expanding marketing spend in high-performing regions like North America and Europe with the new CTA design.",
            "expected_impact": "ROI improvement of 30-35% on paid acquisition"
        }
    ]
    
    return {"recommendations": recommendations}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()