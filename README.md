# HCMUS Data Mining - Project 1

## 1. Thành viên nhóm

| STT | Họ và tên | MSSV |
|---|---|---|
| 1 | Lê Hà Thanh Chương | 23120195 |
| 2 | Lê Thượng Đế | 23120232 |
| 3 | Vũ Nguyễn Trung Hiếu | 23122028 |
| 4 | Châu Văn Minh Khoa | 23122035 |
| 5 | Phan Ngọc Quân | 23122046 |

## 2. Cấu trúc bài nộp

### 2.1 Folder data
data/
Chứa các tập dữ liệu raw và processed

### 2.2 Folder notebooks
notebooks/
Chứa các file notebook để chạy code
Cụ thể trong notebooks sẽ:
- Trình bày sơ lược lý thuyết của từng phần
- Trình bày code và kết quả
- Đưa ra nhận xét và phân tích

### 2.3 Folder report
docs/
Chứa báo cáo kết quả
Cụ thể trình bày:
- Các biểu đồ, kết quả đạt được qua từng bước
- Đưa ra phân tích chi tiết cụ thể hơn
- Rút ra quyết định cuối cùng cho từng bước tiền xử lý của từng phần
- Tổng hợp lại các quyết định và nhận xét về điểm mạnh, điểm yếu của từng phần

## 3. Mô tả tập dữ liệu

Project sử dụng 3 loại dữ liệu chính:

### 2.1. Image dataset
- Đường dẫn: `data/raw/image_data/`
- Cấu trúc chia tập: `train/`, `val/`, `test/`

- Số lớp: 10 lớp ảnh rau/củ/quả
	- `asparagus`, `banana`, `broccoli`, `carrot`, `corn`, `eggplant`, `orange`, `pineapple`, `potato`, `tomato`
- Quy mô dữ liệu:
	- `train`: 7,000 ảnh (700 ảnh/lớp)
	- `val`: 1,500 ảnh (150 ảnh/lớp)
	- `test`: 1,500 ảnh (150 ảnh/lớp)
	- Tổng: 10,000 ảnh
- Đặc điểm: dữ liệu cân bằng giữa các lớp ở cả 3 split.
- Dạng bài toán: phân loại ảnh nhiều lớp (multiclass classification).
- Cách dữ liệu được thu thập tham khảo trong notebook: https://www.kaggle.com/code/chinhde/crawl-images-for-vegetable-classification
- Nguồn dữ liệu gốc: https://www.freepik.com/search?ai=excluded&format=search&last_filter=people&last_value=exclude&people=exclude&sort=relevance&type=photo

### 2.2. Tabular dataset
- Đường dẫn: `data/raw/tabular/adult.csv`
- Loại dữ liệu: dạng bảng, nhiều thuộc tính nhân khẩu học/kinh tế.
- Kích thước: 32,561 records, 15 thuộc tính.
- Nhãn mục tiêu: `income` với 2 lớp `<=50K` và `>50K`.
- Phân bố lớp: khoảng `77.36%` và `22.64%`.
- Các biến số (numeric) chính:
	- `age`, `fnlwgt`, `education.num`, `capital.gain`, `capital.loss`, `hours.per.week`
- Các biến phân loại (categorical) chính:
	- `workclass`, `education`, `marital.status`, `occupation`, `relationship`, `race`, `sex`, `native.country`
- Bài toán: phân loại thu nhập nhị phân (binary classification).
- Link Kaggle: https://www.kaggle.com/datasets/uciml/adult-census-income
- Cách dữ liệu được thu thập: được lấy từ "1994 Census bureau database"

### 2.3. Text dataset
- Đường dẫn: `data/raw/text/IMDB Dataset.csv`
- Nội dung: review phim tiếng Anh.
- Kích thước: 50,000 mẫu, 2 cột (`review`, `sentiment`).
- Nhãn: `positive`, `negative` (cân bằng 50%-50%).
- Dạng bài toán: phân loại cảm xúc văn bản (sentiment classification).
- Link Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- Cách dữ liệu được thu thập tham khảo trong paper: http://ai.stanford.edu/~amaas/data/sentiment/

## 3. Hướng dẫn cài đặt môi trường
Link github: https://github.com/quanpro147/HCMUS-DataMining-Prj1.git

> Yêu cầu chung: Python >= 3.10 và đang đứng tại thư mục gốc project.

### 3.1. Cách 1 - Conda

```bash
conda create -n datamining-prj1 python=3.10 -y
conda activate datamining-prj1
pip install -r requirements.txt
python -m ipykernel install --user --name datamining-prj1 --display-name "Python (datamining-prj1)"
```

### 3.2. Cách 2 - uv

```bash
# Cài uv (nếu máy chưa có)
pip install uv

# Tạo và kích hoạt môi trường
uv venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Cài dependencies
uv pip install -r requirements.txt

# Đăng ký kernel cho Jupyter
python -m ipykernel install --user --name datamining-prj1-uv --display-name "Python (datamining-prj1-uv)"
```

### 3.3. Cách 3 - venv

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name datamining-prj1-venv --display-name "Python (datamining-prj1-venv)"
```

## 4. Hướng dẫn chạy notebook

### 4.1 Tải Dataset
- Trong file nộp đã có sẵn tabular, còn image dataset và text dataset do khá lớn nên sẽ cần tải thủ công từ GG Drive theo link trong folder data/raw/image và data/raw/text.
- Đối với image dataset thì sau khi tải xong thì đặt các tập train/val/test vào thư mục image_data là được
- Đối với text dataset thì sau khi tải xong thì đặt file IMDB Dataset.csv vào thư mục text là được

### 4.2. Mở Jupyter

```bash
jupyter notebook
```

Hoặc:

```bash
jupyter lab
```

Sau đó mở thư mục `notebooks/` và chạy theo thứ tự gợi ý:

1. `01_EDA_image.ipynb`
2. `02_preprocessing_image.ipynb`
3. `03_EDA_tabular.ipynb`
4. `04_preprocessing_tabular.ipynb`
5. `05_EDA_text.ipynb`
6. `06_preprocessing_text.ipynb`

Lưu ý:
- Chọn đúng kernel đã tạo ở bước cài môi trường.

## 5. Bảng phân công công việc

| Họ và tên | Công việc | Mức độ hoàn thành |
|---|---|---|
| Lê Hà Thanh Chương | Phần 2 (Tabular): Toàn bộ EDA và tiền xử lý (a, b, c) | 100% |
| Lê Thượng Đế | Phần 2 (Tabular): Tiền xử lý (d, e, f), viết báo cáo | 100% |
| Vũ Nguyễn Trung Hiếu | Phần 1 (Image): Toàn bộ EDA, tiền xử lý (a, b, c, d) | 100% |
| Châu Văn Minh Khoa | Phần 3 (Text): Toàn bộ EDA, tiền xử lý (a, b, c, d) | 100% |
| Phan Ngọc Quân | Phần 1 (Image): Tiền xử lý (e, f); Phần 3 (Text): Tiền xử lý (e, f), viết báo cáo | 100% |
