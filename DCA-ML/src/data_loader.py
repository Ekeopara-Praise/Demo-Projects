import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_monthly_dca_data(months=60, seed=42):
    np.random.seed(seed)
    
    # 1. Monthly Dates (First of each month for 5 years / 60 months)
    start_date = datetime(2021, 1, 1)
    dates = [start_date + pd.DateOffset(months=i) for i in range(months)]
    
    # 2. Cumulative Days & Produced Days per Month
    calendar_days = [d.days_in_month for d in dates]
    cum_calendar_days = np.cumsum(calendar_days)
    
    # Simulate realistic uptime / downtime (e.g., 25 to 31 days produced/month)
    produced_days = [
        int(cd - np.random.choice([0, 0, 0, 1, 2, 5])) for cd in calendar_days
    ]
    # Simulate a major workover shutdown in Month 24
    produced_days[23] = 8 
    
    # 3. True Continuous Rate (Hyperbolic Arps: qi=1200 BOPD, Di=0.003/day, b=0.75)
    qi, Di, b = 1200.0, 0.003, 0.75
    daily_rates_pure = qi / ((1 + b * Di * cum_calendar_days) ** (1 / b))
    
    # 4. Add Noise & Wellhead Pressure (THP) / Choke Dynamics
    noise = np.random.normal(loc=1.0, scale=0.03, size=months)
    actual_bopd = daily_rates_pure * noise
    
    # Monthly Total Volumes = Effective Daily Rate * Total Produced Days
    monthly_oil = np.round(actual_bopd * produced_days, 2)
    
    # Gas-Oil Ratio (GOR) increasing over time (800 -> 1800 scf/STB)
    gor = 800 + (1000 * (cum_calendar_days / cum_calendar_days[-1]) ** 0.5)
    monthly_gas = np.round((monthly_oil * gor) / 1000.0, 2)
    
    # Water Cut climbing (5% -> 40%)
    wcut = 0.05 + 0.35 * (1 - np.exp(-0.0015 * cum_calendar_days))
    monthly_water = np.round((monthly_oil / (1 - wcut)) * wcut, 2)
    
    # THP & Choke Dynamics (THP declines as reservoir pressure drops)
    thp_psi = np.round(850 * np.exp(-0.0008 * cum_calendar_days) + np.random.normal(0, 10, months), 1)
    bean_size_64ths = [32 if days > 20 else 16 for days in produced_days]
    
    # Cumulative Oil Calculation (MSTB)
    cum_oil_mstb = np.round(np.cumsum(monthly_oil) / 1000.0, 3)
    
    # Build DataFrame
    df = pd.DataFrame({
        'Date': [d.strftime('%m/%d/%Y') for d in dates],
        'Total_Produced_Days': produced_days,
        'Monthly_Oil_STB': monthly_oil,
        'Monthly_Gas_MSCF': monthly_gas,
        'Monthly_Water_STB': monthly_water,
        'Bean_Size_64ths': bean_size_64ths,
        'THP_psi': thp_psi,
        'Cum_Oil_MSTB': cum_oil_mstb
    })
    
    return df

# Generate and save
df_monthly = generate_monthly_dca_data()
df_monthly.to_csv("data/raw/monthly_production_data.csv", index=False)
print("Monthly dataset saved to data/raw/monthly_production_data.csv!")