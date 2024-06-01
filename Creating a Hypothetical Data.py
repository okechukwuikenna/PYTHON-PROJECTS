#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate dates
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = pd.date_range(start_date, end_date)

# Generate synthetic data
data = {
    'Date': np.random.choice(date_range, size=1000),
    'Store_ID': np.random.randint(1, 6, size=1000),
    'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Groceries', 'Furniture'], size=1000),
    'Sales_Revenue': np.random.uniform(10.0, 1000.0, size=1000),
    'Num_Customers': np.random.randint(1, 50, size=1000)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('sales_data.csv', index=False)
print(data)


