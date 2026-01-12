# OTT Streaming Analytics and Recommendation System

## Problem Statement

OTT streaming platforms face significant challenges in understanding user behavior, predicting churn, and delivering personalized content recommendations. High churn rates directly impact revenue, while poor recommendation quality reduces user engagement and satisfaction. Without data-driven insights, platforms struggle to identify at-risk users, understand content preferences, and optimize their content library strategy.

Success in this domain is measured by accurately predicting which users will churn, enabling proactive retention strategies, and providing recommendations that increase user engagement and reduce churn probability.

## Objective

This project aims to build a comprehensive analytics and recommendation system that:
- Predicts user churn with high accuracy using machine learning models
- Identifies distinct user segments through unsupervised clustering
- Generates personalized content recommendations using hybrid filtering approaches
- Provides actionable insights through interactive visualizations

The system assumes access to user demographic data, watch history, and content metadata. It is designed to work with synthetic data for demonstration purposes but can be adapted to real-world streaming platform data.

## Dataset

**Dataset Name**: Synthetic OTT Streaming Platform Data

**Type**: Structured tabular data (time-series behavioral data)

**Size**: 
- 10,000 users
- 2,000 content titles
- 150,000+ watch history records

**Key Features**:

1. **User Dataset** (`ott_users.csv`):
   - Demographics: age, gender, country
   - Subscription: plan type (basic, standard, premium), device type
   - Behavioral: average watch time, sessions per week, preferred genre
   - Churn: binary target variable (0/1)
   - Temporal: join date, last active date

2. **Watch History Dataset** (`ott_watch_history.csv`):
   - User-content interactions: user_id, title_id
   - Engagement: watch_time_minutes, rating (1-5 scale)
   - Temporal: watch_date

3. **Content Dataset** (`ott_titles.csv`):
   - Metadata: title_name, genre, release_year, age_rating
   - Engagement: popularity_score
   - Descriptive: text descriptions for content-based filtering

**Data Preprocessing Steps**:
- Handled missing values using median imputation for numerical features and mode for categorical features
- Removed duplicate records across all datasets
- Standardized categorical variables (genre, device_type, subscription_plan)
- Converted date columns to datetime format with proper timezone handling
- Merged datasets to create unified feature sets for analysis
- Filtered sparse interactions (users/titles with fewer than 3 ratings) for collaborative filtering

## Approach

The solution follows a modular pipeline architecture:

1. **Data Generation**: Synthetic data generation using statistical distributions (normal, exponential, categorical) to simulate realistic user behavior patterns

2. **Exploratory Data Analysis**: Comprehensive analysis including:
   - Genre-level engagement metrics
   - User segmentation via KMeans clustering
   - Country-wise behavioral patterns
   - Time-based engagement trends
   - User Lifetime Value (LTV) estimation

3. **Churn Prediction**: Supervised learning approach with:
   - RFM (Recency, Frequency, Duration) feature engineering
   - Multiple model comparison (Logistic Regression, Random Forest, Gradient Boosting)
   - Stratified train-test split (80-20) to handle class imbalance
   - Feature scaling for linear models

4. **Recommendation System**: Hybrid approach combining:
   - Content-based filtering using TF-IDF vectorization and cosine similarity
   - Collaborative filtering using matrix factorization (SVD/NMF)
   - Weighted hybrid combination (40% content-based, 60% collaborative)

5. **Visualization and Dashboard**: Interactive Streamlit dashboard with Plotly visualizations for real-time insights

## Model & Techniques Used

**Machine Learning Models**:
- **Logistic Regression**: Linear classification for churn prediction with L2 regularization
- **Random Forest Classifier**: Ensemble method with 100 trees, handles non-linear relationships
- **Gradient Boosting Classifier**: Sequential ensemble with 100 estimators, optimized for imbalanced classes
- **KMeans Clustering**: Unsupervised learning with 4 clusters for user segmentation
- **SVD (Singular Value Decomposition)**: Matrix factorization for collaborative filtering (50 latent factors)
- **NMF (Non-negative Matrix Factorization)**: Alternative matrix factorization when Surprise library unavailable
- **TF-IDF Vectorization**: Text feature extraction for content-based recommendations

**Statistical Techniques**:
- StandardScaler for feature normalization
- LabelEncoder for categorical variable encoding
- Stratified sampling for balanced train-test splits
- Cosine similarity for content matching
- User-item matrix construction for collaborative filtering

**Libraries and Frameworks**:
- Python 3.8+
- Pandas, NumPy for data manipulation
- Scikit-learn for machine learning algorithms
- Surprise library for collaborative filtering (with NMF fallback)
- Matplotlib, Seaborn, Plotly for visualization
- Streamlit for interactive dashboard

## Evaluation Metrics

**Churn Prediction Metrics**:
- **Accuracy**: Overall classification correctness
- **Precision**: Proportion of predicted churners who actually churn (reduces false positives)
- **Recall**: Proportion of actual churners correctly identified (reduces false negatives)
- **F1-Score**: Harmonic mean of precision and recall (primary metric for imbalanced data)
- **ROC-AUC**: Area under ROC curve measuring model's ability to distinguish between classes

These metrics were chosen because churn prediction is a binary classification problem with class imbalance. F1-Score is prioritized as it balances precision and recall, which is critical when the cost of missing a churner (false negative) is high.

**Recommendation System Metrics**:
- **RMSE (Root Mean Squared Error)**: Measures prediction error for rating predictions
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual ratings

RMSE and MAE are standard metrics for collaborative filtering systems as they measure the quality of rating predictions on a continuous scale.

**Validation Strategy**:
- 80-20 stratified train-test split for churn prediction (maintains class distribution)
- 5-fold cross-validation for model selection (when applicable)
- Temporal validation using time-based splits for watch history data

## Results

**Churn Prediction Performance**:

Best Model: Gradient Boosting Classifier
- Accuracy: 92.00%
- Precision: 82.21%
- Recall: 58.16%
- F1-Score: 68.13%
- ROC-AUC: 0.8844

Model Comparison:
- Logistic Regression: 91.90% accuracy, 66.53% F1-Score, 0.8868 ROC-AUC
- Random Forest: 91.70% accuracy, 66.12% F1-Score, 0.8649 ROC-AUC
- Gradient Boosting: 92.00% accuracy, 68.13% F1-Score, 0.8844 ROC-AUC

**Key Insights**:
- Gradient Boosting achieved the highest F1-Score, making it the best choice for identifying churners while minimizing false positives
- All models show high accuracy (>91%) but moderate recall (~55-58%), indicating room for improvement in identifying all churners
- RFM features (recency, frequency, duration) are strong predictors of churn
- Subscription plan and device type show significant correlation with churn behavior

**User Segmentation Results**:
- Identified 4 distinct user segments through KMeans clustering:
  - High-engagement users (frequent, long sessions)
  - Casual viewers (moderate engagement)
  - Occasional users (low frequency, short sessions)
  - Premium power users (high watch time, premium plans)

**Recommendation System Performance**:
- Collaborative filtering achieves RMSE < 1.0 and MAE < 0.8 on rating predictions
- Hybrid approach provides more diverse recommendations than individual methods
- Content-based filtering handles cold-start problem for new users

**Limitations**:
- Recall of 58% means 42% of churners are not identified, requiring further feature engineering or model tuning
- Synthetic data may not capture all real-world behavioral nuances
- Recommendation system performance depends on data sparsity; sparse matrices reduce prediction quality
- Model assumes stationarity in user behavior patterns

## Business / Real-World Impact

**Practical Applications**:
- **Churn Prevention**: Marketing teams can target at-risk users (predicted churners) with retention campaigns, discounts, or personalized content recommendations before they cancel subscriptions
- **Content Strategy**: Genre engagement analysis helps content acquisition teams identify which genres drive the most engagement and inform licensing decisions
- **User Segmentation**: Product teams can design personalized experiences for different user segments, improving overall platform engagement
- **Recommendation Engine**: Increases content discovery, watch time, and user satisfaction, directly impacting platform metrics like Average Revenue Per User (ARPU) and Monthly Active Users (MAU)

**Stakeholders Who Benefit**:
- **Product Managers**: Data-driven insights for feature prioritization and user experience improvements
- **Marketing Teams**: Targeted retention campaigns based on churn predictions and user segments
- **Content Teams**: Genre and title performance metrics for content acquisition and production decisions
- **Data Science Teams**: Reusable pipeline architecture for similar analytics projects

**Decision Enablement**:
- Prioritize retention efforts on users with high churn probability scores
- Allocate marketing budget to high-value user segments
- Adjust content library mix based on genre engagement trends
- Personalize homepage and recommendation carousels for individual users

## Project Structure

```
StreamPulse-Analytics/
├── data/                          # Input datasets
│   ├── ott_users.csv
│   ├── ott_watch_history.csv
│   └── ott_titles.csv
├── notebooks/                      # Jupyter notebook for end-to-end execution
│   └── advanced_analysis.ipynb
├── src/                            # Python source code modules
│   ├── generate_data.py           # Synthetic data generation
│   ├── data_cleaning.py           # Data preprocessing pipeline
│   ├── eda.py                      # Exploratory data analysis
│   ├── visualization.py           # Chart generation
│   ├── churn_model.py             # Churn prediction models
│   ├── content_based_recommender.py
│   ├── collaborative_filtering.py
│   ├── hybrid_recommender.py
│   └── path_utils.py              # Path management utilities
├── dashboard/                      # Streamlit interactive dashboard
│   └── app.py
├── outputs/                        # Generated outputs
│   ├── charts/                    # Visualization images
│   ├── cleaned_users.csv
│   ├── cleaned_watch_history.csv
│   ├── churn_model_metrics.txt
│   ├── sample_recommendations.csv
│   └── hybrid_engine_results.txt
├── images/                         # README documentation images
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## How to Run This Project

**Prerequisites**: Python 3.8 or higher

**Step 1: Clone the Repository**
```bash
git clone <repository-url>
cd StreamPulse-Analytics
```

**Step 2: Create and Activate Virtual Environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with the `surprise` library (particularly on Python 3.14+), the system will automatically fall back to scikit-learn's NMF implementation.

**Step 4: Generate Data (if not already present)**
```bash
python src/generate_data.py
```

**Step 5: Run End-to-End Analysis**
```bash
jupyter notebook notebooks/advanced_analysis.ipynb
```
Execute all cells in sequence. The notebook will:
- Clean and preprocess data
- Perform exploratory data analysis
- Train churn prediction models
- Generate recommendations
- Create visualizations

**Step 6: Launch Interactive Dashboard**
```bash
streamlit run dashboard/app.py
```
The dashboard will open in your default web browser at `http://localhost:8501`

**Alternative: Run Individual Scripts**
```bash
# Data cleaning
python src/data_cleaning.py

# EDA and clustering
python src/eda.py

# Churn prediction
python src/churn_model.py

# Recommendation systems
python src/content_based_recommender.py
python src/collaborative_filtering.py
python src/hybrid_recommender.py
```

## Future Improvements

**Model Enhancements**:
- Implement deep learning models (Neural Collaborative Filtering, LSTM for sequential patterns) to improve recommendation accuracy
- Add ensemble methods (stacking, voting) to combine churn prediction models and boost recall
- Experiment with XGBoost or LightGBM for potentially better churn prediction performance
- Implement time-series forecasting models to predict future engagement trends

**Data Improvements**:
- Incorporate additional features: device information, network quality, content metadata (directors, actors)
- Add temporal features: seasonality, day-of-week patterns, holiday effects
- Include external data: competitor pricing, market trends, social media sentiment
- Handle concept drift by implementing online learning or periodic model retraining

**Deployment and Scaling**:
- Containerize the application using Docker for consistent deployment
- Deploy dashboard on cloud platforms (AWS, GCP, Azure) with auto-scaling
- Implement model serving API using Flask/FastAPI for real-time predictions
- Set up automated model retraining pipeline with MLflow or Kubeflow
- Add A/B testing framework to compare recommendation strategies

**Feature Engineering**:
- Create interaction features between user demographics and content attributes
- Implement feature selection techniques to reduce dimensionality
- Add embedding-based features using word2vec or doc2vec for content descriptions

## Key Learnings

**Technical Learnings**:
- RFM (Recency, Frequency, Duration) features are highly predictive for churn in subscription-based services
- Hybrid recommendation systems outperform single-method approaches by combining content-based and collaborative signals
- Class imbalance in churn prediction requires careful metric selection (F1-Score over Accuracy)
- Matrix factorization techniques (SVD, NMF) effectively handle sparse user-item interaction matrices
- Feature scaling is critical for linear models but less important for tree-based ensemble methods

**Data Science Process Learnings**:
- Synthetic data generation requires careful consideration of statistical distributions to maintain realism
- Modular pipeline architecture enables easier debugging, testing, and maintenance
- Interactive dashboards significantly improve stakeholder engagement compared to static reports
- End-to-end notebooks are valuable for reproducibility but require robust path handling for different execution contexts
- Fallback mechanisms (NMF when Surprise unavailable) improve system robustness and deployment flexibility

**Domain-Specific Insights**:
- User engagement patterns vary significantly by genre, requiring genre-specific recommendation strategies
- Device type and subscription plan are strong indicators of user value and churn risk
- Time-based features (days since last watch, session frequency) capture behavioral changes that precede churn
- Content popularity scores alone are insufficient; user preferences and historical interactions drive better recommendations

## References

**Papers and Research**:
- Ricci, F., Rokach, L., & Shapira, B. (2015). "Recommender Systems Handbook" - Foundation for collaborative filtering approaches
- Koren, Y., Bell, R., & Volinsky, C. (2009). "Matrix Factorization Techniques for Recommender Systems" - SVD and matrix factorization methods
- Verbraken, T., Verbeke, W., & Baesens, B. (2013). "A Novel Profit-Based Metric for Evaluating Classification Models" - Churn prediction evaluation metrics

**Libraries and Tools**:
- Scikit-learn Documentation: https://scikit-learn.org/stable/
- Surprise Library: http://surpriselib.com/ (Collaborative filtering)
- Streamlit Documentation: https://docs.streamlit.io/
- Plotly Documentation: https://plotly.com/python/

**Datasets**:
- Synthetic data generated using statistical distributions based on industry benchmarks for OTT platforms
- Data generation approach inspired by Netflix Prize dataset structure and real-world streaming platform characteristics

## Output

Visualizations and charts generated by the analysis pipeline are stored in the `outputs/charts/` directory. Key visualizations include:

- Engagement heatmaps showing user activity patterns
- Genre popularity trends over time
- Churn prediction confusion matrices and ROC curves
- User segmentation cluster visualizations
- Device-type watch pattern distributions
- Rating distribution analysis

To include these visualizations in documentation, copy relevant charts to the `images/` directory and reference them in markdown using:

```markdown
![Chart Description](images/chart_filename.png)
```
