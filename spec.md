# ML Studio

## Current State
The app is a Machine Learning Studio with a tab-based navigation (no sidebar), CSV upload limited to housing data (sqft/bedrooms/bathrooms/age/price fields), and in-browser linear regression. It has 11 sections accessed via top tabs.

## Requested Changes (Diff)

### Add
- Collapsible sidebar navigation with icons and labels replacing top tab navigation
- Universal CSV uploader that accepts any CSV dataset (auto-detects all columns/features)
- Automatic target column selection (user picks which column to predict)
- Multi-feature training: trains on all numeric columns except target
- More content: dataset statistics cards, correlation info, feature importance chart, model comparison section, data preview table with pagination, training progress visualization
- Professional header with dataset name and status indicators

### Modify
- CSV parsing: detect all numeric columns automatically, not just housing-specific fields
- Training logic: use all selected numeric columns as features, not hardcoded sqft/bedrooms/etc
- Prediction panel: show inputs for each detected feature column
- Data summary: show stats for all columns (min, max, mean, std)
- Sample data: keep as fallback but clearly labeled as demo data

### Remove
- Hardcoded sqft-only prediction input
- Top tab navigation

## Implementation Plan
1. Add sidebar component with icons, collapse toggle, section labels
2. Refactor CSV parser to auto-detect all numeric columns
3. Add target column selector (dropdown after upload)
4. Refactor ML training to use dynamic feature set
5. Update prediction form to render inputs per detected feature
6. Add dataset stats cards (rows, columns, missing values, numeric cols)
7. Add data preview table with first 10 rows
8. Add feature importance bar chart (coefficient magnitudes)
9. Add model metrics section (R squared, MAE, RMSE)
10. Wire sidebar navigation to all sections
