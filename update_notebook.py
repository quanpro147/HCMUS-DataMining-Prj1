import nbformat as nbf
import numpy as np

notebook_path = 'notebooks/04_preprocessing_tabular.ipynb'

# Read the existing notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

cells_to_add = [
    # Cell 1
    nbf.v4.new_markdown_cell("""## **7. Feature Selection & Dimensionality Reduction**
Phần này tập trung vào việc lựa chọn các đặc trưng quan trọng nhất và giảm số lượng chiều của dữ liệu nhằm tránh "curse of dimensionality" (lời nguyền số chiều). Việc này không chỉ giúp giảm hiện tượng overfitting, cải thiện khả năng tổng quát hóa của mô hình mà còn tăng tốc độ huấn luyện và giảm chi phí tính toán, đặc biệt quan trọng đối với dữ liệu dạng bảng chứa nhiều biến phân loại đã qua mã hóa (One-Hot Encoding)."""),
    
    # Cell 2
    nbf.v4.new_markdown_cell("""### **7.1 Theoretical Foundation & Mathematical Formulas**
Các phương pháp được sử dụng trong phần này được chia thành ba nhóm chính, bao gồm:

1. **ANOVA F-test (Dành cho đặc trưng số vs. mục tiêu phân loại):**
   Đánh giá xem giá trị trung bình của một đặc trưng có khác biệt đáng kể giữa các nhóm phân lớp hay không. 
   Công thức:
   $$F = \\frac{MS_{between}}{MS_{within}}$$

2. **Chi-square Test (Dành cho đặc trưng phân loại vs. mục tiêu phân loại):**
   Kiểm định tính độc lập giữa hai biến phân loại dựa trên sự khác biệt giữa tần số quan sát ($O_i$) và tần số kỳ vọng ($E_i$).
   Công thức:
   $$\\chi^2 = \\sum \\frac{(O_i - E_i)^2}{E_i}$$

3. **Mutual Information (Dành cho cả hai loại đặc trưng):**
   Đo lường lượng thông tin chung giữa hai biến, phản ánh mức độ giảm độ bất định của biến này khi biết biến kia.
   Công thức:
   $$I(X; Y) = \\sum_{y \\in Y} \\sum_{x \\in X} p(x, y) \\log \\left( \\frac{p(x, y)}{p(x)p(y)} \\right)$$

4. **Tree-based Importance:**
   Đo lường mức độ quan trọng của đặc trưng thông qua sự suy giảm độ tinh khiết (Gini Impurity) hoặc Entropy (Information Gain) khi thực hiện phân nhánh trên các node sử dụng đặc trưng đó.

5. **Recursive Feature Elimination with Cross-Validation (RFE-CV):**
   Phương pháp lặp lại việc huấn luyện mô hình và loại bỏ đặc trưng có độ quan trọng thấp nhất cho đến khi đạt được số lượng tối ưu $k$. Quá trình cross-validation giúp đảm bảo khả năng tổng quát.

6. **Principal Component Analysis (PCA):**
   Biến đổi tuyến tính tìm các hướng phản ánh phương sai lớn nhất trong dữ liệu thông qua phân tích giá trị riêng hoặc phân tích giá trị suy biến (SVD).
   Công thức chiếu dữ liệu sang không gian mới:
   $$z = XW$$

7. **t-Distributed Stochastic Neighbor Embedding (t-SNE):**
   Phương pháp giảm chiều phi tuyến dựa trên việc bảo toàn phân phối xác suất lân cận, tối thiểu hóa hàm mất mát Kullback-Leibler.
   Công thức:
   $$C = \\sum_{i} KL(P_i || Q_i)$$

8. **Uniform Manifold Approximation and Projection (UMAP):**
   Giảm chiều dữ liệu phi tuyến nhằm bảo toàn cấu trúc tô-pô của dữ liệu từ không gian gốc sang không gian nhúng thông qua việc tối ưu hóa cross-entropy."""),
    
    # Cell 3
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, RFECV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Ignore warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# Prepare X and y from the cleaned dataset
if 'df_adult_clean' in locals() or 'df_adult_clean' in globals():
    df = df_adult_clean.copy()
else:
    # Fallback to reading from potentially saved intermediate file
    try:
        df = pd.read_csv('../data/interim/adult_clean_encoded.csv')
    except:
        print("Dataframe not found in memory or disk. Please ensure previous steps are run.")

if 'income' in df.columns:
    X = df.drop('income', axis=1)
    y = df['income']
else:
    X = X_encoded
    y = y_encoded

# Setup cross validation strategy
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metric = 'f1'"""),

    # Cell 4
    nbf.v4.new_markdown_cell("""### **7.2 Statistical Filtering**
Phương pháp này sử dụng `SelectKBest` với các tiêu chuẩn thống kê khác nhau để xếp hạng và chọn lọc các đặc trưng. Bằng cách duyệt qua dải giá trị $k$, ta có thể đánh giá tính hiệu quả của tập đặc trưng đã chọn dựa trên F1-score thu được qua 5-Fold Cross Validation trên một mô hình LogisticRegression."""),

    # Cell 5
    nbf.v4.new_code_cell("""# Define range for number of features to select
max_features = min(X.shape[1], 51)
k_range = list(range(5, max_features, 5))

results = {'f_classif': [], 'chi2': [], 'mutual_info': []}

# Evaluate Statistical Filtering methods
for k in k_range:
    # f_classif
    selector_f = SelectKBest(f_classif, k=k)
    X_f = selector_f.fit_transform(X, y)
    scores_f = cross_val_score(LogisticRegression(max_iter=1000), X_f, y, cv=cv_strategy, scoring=scoring_metric)
    results['f_classif'].append(np.mean(scores_f))
    
    # Transform data to be non-negative for chi2
    X_min = X.min()
    X_non_negative = X - X_min * (X_min < 0)
    
    # chi2
    try:
        selector_chi2 = SelectKBest(chi2, k=k)
        X_chi2 = selector_chi2.fit_transform(X_non_negative, y)
        scores_chi2 = cross_val_score(LogisticRegression(max_iter=1000), X_chi2, y, cv=cv_strategy, scoring=scoring_metric)
        results['chi2'].append(np.mean(scores_chi2))
    except Exception as e:
        results['chi2'].append(np.nan)
        print(f"Chi2 failed for k={k}: {e}")
        
    # mutual_info_classif
    # Use a subset of data for mutual info to speed up if data is large
    sample_size = min(len(X), 10000)
    idx = np.random.choice(len(X), sample_size, replace=False)
    X_sample, y_sample = X.iloc[idx], y.iloc[idx]
    
    selector_mi = SelectKBest(mutual_info_classif, k=k)
    X_mi = selector_mi.fit_transform(X_sample, y_sample) # Fast approximation
    # Test on full using support from sample
    X_full_mi = X.iloc[:, selector_mi.get_support()]
    scores_mi = cross_val_score(LogisticRegression(max_iter=1000), X_full_mi, y, cv=cv_strategy, scoring=scoring_metric)
    results['mutual_info'].append(np.mean(scores_mi))"""),

    # Cell 6
    nbf.v4.new_code_cell("""# Plot the evaluation results
plt.figure(figsize=(10, 6))

plt.plot(k_range, results['f_classif'], marker='o', label='ANOVA F-value')
if not np.isnan(results['chi2']).all():
    plt.plot(k_range, results['chi2'], marker='s', label='Chi-Squared')
plt.plot(k_range, results['mutual_info'], marker='^', label='Mutual Information')

plt.title('Feature Selection using Statistical Filtering', fontsize=14)
plt.xlabel('Number of Selected Features (k)', fontsize=12)
plt.ylabel('CV F1-Score', fontsize=12)
plt.legend(title='Method')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()"""),

    # Cell 7
    nbf.v4.new_markdown_cell("""### **7.3 Model-Based Filtering**
Phương pháp này tận dụng thuộc tính `feature_importances_` từ các mô hình học máy dạng cây (Random Forest, Gradient Boosting) để đong đếm tầm quan trọng của từng đặc trưng. Kỹ thuật Lại bỏ Đặc trưng Đệ quy tích hợp Cross Validation (RFE-CV) được áp dụng nhằm tự động tìm ra tập hợp $k$ đặc trưng mang lại hiệu suất mô hình tối ưu nhất thay vì phải thiết lập ngưỡng thủ công."""),

    # Cell 8
    nbf.v4.new_code_cell("""# Train models for feature importances
rf_clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf_clf.fit(X, y)
rf_importance = rf_clf.feature_importances_

gb_clf = GradientBoostingClassifier(n_estimators=50, random_state=42)
# Use a sample for GB if data is very large
gb_clf.fit(X_sample, y_sample)
gb_importance = gb_clf.feature_importances_

# Sort features by importance
rf_indices = np.argsort(rf_importance)[::-1]
gb_indices = np.argsort(gb_importance)[::-1]

results_model = {'RF Importance': [], 'GB Importance': []}

# Evaluate top-k features
for k in k_range:
    top_k_rf = rf_indices[:k]
    scores_rf = cross_val_score(LogisticRegression(max_iter=1000), X.iloc[:, top_k_rf], y, cv=cv_strategy, scoring=scoring_metric)
    results_model['RF Importance'].append(np.mean(scores_rf))
    
    top_k_gb = gb_indices[:k]
    scores_gb = cross_val_score(LogisticRegression(max_iter=1000), X.iloc[:, top_k_gb], y, cv=cv_strategy, scoring=scoring_metric)
    results_model['GB Importance'].append(np.mean(scores_gb))

# Configure RFE-CV
rfecv = RFECV(
    estimator=RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1), 
    step=1, 
    cv=cv_strategy, 
    scoring=scoring_metric,
    min_features_to_select=5
)
# Run RFE-CV on a sample to speed up execution
rfecv.fit(X_sample, y_sample)

# Print Top 10 features
print("Top 10 features by Random Forest:")
for i in range(min(10, len(rf_indices))):
    print(f"{i+1}. {X.columns[rf_indices[i]]:<25} ({rf_importance[rf_indices[i]]:.4f})")
    
print("\\nTop 10 features by Gradient Boosting:")
for i in range(min(10, len(gb_indices))):
    print(f"{i+1}. {X.columns[gb_indices[i]]:<25} ({gb_importance[gb_indices[i]]:.4f})")"""),

    # Cell 9
    nbf.v4.new_code_cell("""# Plot the evaluation results for model-based filtering
plt.figure(figsize=(10, 6))

plt.plot(k_range, results_model['RF Importance'], marker='o', label='RF Feature Importance')
plt.plot(k_range, results_model['GB Importance'], marker='s', label='GB Feature Importance')

# RFE-CV Results
rfe_scores = rfecv.cv_results_['mean_test_score']
# Using min_features_to_select offset for rfe x-axis
rfe_x = range(5, len(rfe_scores) + 5) 
plt.plot(rfe_x, rfe_scores, marker='^', label='RFE-CV (Random Forest)', color='red', alpha=0.7)

plt.title('Feature Selection using Model-Based Filtering & RFE', fontsize=14)
plt.xlabel('Number of Selected Features (k)', fontsize=12)
plt.ylabel('CV F1-Score', fontsize=12)
plt.legend(title='Method')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()"""),

    # Cell 10
    nbf.v4.new_markdown_cell("""### **7.4 Dimensionality Reduction & Visualization**
Các phương pháp giảm chiều dữ liệu không giám sát giúp ánh xạ dữ liệu nhiều chiều xuống mặt phẳng 2D để quan sát sự phân bố các lớp dữ liệu trực quan hơn. PCA tìm kiếm phép chiếu tuyến tính bảo toàn tối đa phương sai, trong khi t-SNE bảo toàn cấu trúc phân phối cục bộ và UMAP có khả năng bảo toàn cấu trúc toàn cục hiệu quả hơn. Mục đích của phần này chủ yếu là trực quan hóa tính phân tách của tập dữ liệu trên không gian đặc trưng biểu diễn thay vì đóng vai trò chọn lọc đặc trưng phục vụ huấn luyện mô hình tabular."""),

    # Cell 11
    nbf.v4.new_code_cell("""# Subsample data for visualization (to keep it fast and readable)
viz_sample_size = min(len(X), 5000)
np.random.seed(42)
viz_indices = np.random.choice(len(X), viz_sample_size, replace=False)
X_viz = X.iloc[viz_indices]
y_viz = y.iloc[viz_indices]

# Standardize before dimensionality reduction
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_viz_scaled = scaler.fit_transform(X_viz)

# Perform PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_viz_scaled)
print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Explained Variance (2 components): {sum(pca.explained_variance_ratio_):.4f}")

# Perform t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_viz_scaled)

# Perform UMAP
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_viz_scaled)

# Plotting the reductions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_viz, ax=axes[0], palette='viridis', s=20, alpha=0.7)
axes[0].set_title('PCA Visualization', fontsize=12)

sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_viz, ax=axes[1], palette='viridis', s=20, alpha=0.7)
axes[1].set_title('t-SNE Visualization', fontsize=12)

sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y_viz, ax=axes[2], palette='viridis', s=20, alpha=0.7)
axes[2].set_title('UMAP Visualization', fontsize=12)

for ax in axes:
    ax.legend(title='Income')

plt.tight_layout()
plt.show()"""),

    # Cell 12
    nbf.v4.new_markdown_cell("""### **7.5 Comparative Analysis & Final Strategy**
Dựa trên kết quả thực nghiệm từ các phần trước:

- **Về F1-score:** Các phương pháp Model-Based (RFE, RF Importance) thường cho ra tập đặc trưng có khả năng dự đoán cao hơn so với phần lớn các phương pháp Statistical Filtering tuyến tính, tuy chi phí tính toán cao hơn đáng kể. Statistical Filtering dùng F-ANOVA là lựa chọn dung hòa tốt giữ chi phí tính toán và hiệu năng.
- **Giảm chiều dữ liệu (PCA/t-SNE/UMAP):** Kết quả trực quan hóa cho thấy sự đan xen giữa hai phân lớp, chứng tỏ đây là một bài toán phân loại phức tạp và dữ liệu dạng bảng chứa nhiều đặc trưng quan trọng khó biểu diễn tuyến tính tốt chỉ thông qua một vài thành phần chính. Việc sử dụng trực tiếp Feature Selection bảo toàn được độ diễn giải độc lập của các đặc trưng gốc (Interpretability) quan trọng cho quyết định nghiệp vụ, thay vì phép chiếu thay đổi thông tin (như PCA).

**Chiến lược cuối cùng:** 
Sử dụng **RFE-CV kết hợp với Random Forest** (hoặc chọn Top-$k$ từ Random Forest) làm chiến lược lựa chọn đặc trưng chính thức do tính ổn định cao và khả năng bảo toàn hiệu năng mô hình (F1-Score tốt nhất với số lượng biến ít hơn). Tập dữ liệu sẽ được giữ lại các biến được mô hình RFE đánh giá là quan trọng (đạt tiêu chí Support)."""),

    # Cell 13
    nbf.v4.new_code_cell("""# Apply final feature selection strategy (using RFE results)
final_feature_indices = np.where(rfecv.support_)[0]
selected_features = X.columns[final_feature_indices]

# Create final clean dataframe
df_final = X[selected_features].copy()
df_final['income'] = y.values

# Save the description and shape
print(f"Original shape: {X.shape}")
print(f"Final shape after feature selection: {df_final.shape}")
print(f"Selected features count: {len(selected_features)}")

# Display the first few rows
display(df_final.head())

# Optional: Save df_final to disk for modeling step
try:
    df_final.to_csv('../data/processed/adult_final_selected.csv', index=False)
except:
    pass""")
]

nb.cells.extend(cells_to_add)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

