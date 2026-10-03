# Demo lớp học — Phân loại bệnh tim

Bộ tài liệu dạy Machine Learning trên `heart_disease.csv` (10,000 dòng × 21 cột),
gồm **ứng dụng Streamlit** để trình chiếu và **Jupyter Notebook** để học sinh tự chạy.

## Cài đặt

```bash
pip install -r requirements.txt              # chỉ chạy ứng dụng
pip install -r requirements-notebook.txt     # thêm, nếu muốn chạy notebook
```

## Chạy ứng dụng Streamlit

```bash
cd /Users/ts00018/DataScience/heart-disease
streamlit run app.py
```

Trình duyệt tự mở tại `http://localhost:8501`. Dừng bằng `Ctrl + C`.

**Chế độ sáng / tối:** menu ☰ góc trên phải → *Settings* → *Appearance*. Hai bảng màu
được khai báo trong `.streamlit/config.toml`; giao diện tự khớp ngay, không cần tải lại.

Thư mục `artifacts/` đã có sẵn mô hình huấn luyện nên app chạy được ngay.
Nếu muốn huấn luyện lại từ đầu:

```bash
python train.py    # 8 mô hình + thí nghiệm đối chứng   (~1 phút)
python tune.py     # GridSearchCV cho 4 mô hình          (~2 phút)
```

## Chạy Jupyter Notebook

```bash
jupyter notebook notebooks/heart_disease_walkthrough.ipynb
```

Notebook độc lập hoàn toàn với `src/` — mọi bước được viết tường minh để học sinh
đọc và sửa trực tiếp. Chạy lại toàn bộ bằng *Kernel → Restart & Run All*.

## Deploy lên Streamlit Community Cloud

Streamlit Cloud chỉ deploy từ **repo GitHub**, nên bước đầu là đẩy dự án lên GitHub.
Thư mục này đã được `git init` và commit sẵn.

### 1. Tạo repo trên GitHub

Vào https://github.com/new, đặt tên (ví dụ `heart-disease-demo`), **không** tích
"Add a README file", rồi bấm *Create repository*.

### 2. Đẩy mã nguồn lên

```bash
cd /Users/ts00018/DataScience/heart-disease
git remote add origin https://github.com/<tên-tài-khoản>/heart-disease-demo.git
git push -u origin main
```

### 3. Deploy

1. Vào https://share.streamlit.io và đăng nhập bằng GitHub.
2. Bấm **Create app** → **Deploy a public app from GitHub**.
3. Điền: *Repository* = repo vừa tạo, *Branch* = `main`, *Main file path* = `app.py`.
4. Bấm **Deploy**. Lần đầu mất khoảng 3–5 phút để cài thư viện.

Bộ phụ thuộc đã được kiểm thử trên **cả Python 3.13 lẫn 3.14**, nên không cần chọn
phiên bản Python cụ thể trong *Advanced settings*.

### Những tệp Streamlit Cloud cần

| Tệp | Vai trò |
|---|---|
| `requirements.txt` | Thư viện Python, **ghim chính xác phiên bản** |
| `packages.txt` | Gói hệ thống Linux — `libgomp1` cho LightGBM |
| `.streamlit/config.toml` | Theme sáng/tối |
| `artifacts/*.joblib` | Mô hình đã huấn luyện, commit kèm để app chạy ngay |

### Sự cố đã gặp: pyarrow không build được trên Python 3.14

Lần deploy đầu thất bại với:

```
× Failed to download and build `pyarrow==21.0.0`
  error: command 'cmake' failed: No such file or directory
ERROR: Could not build wheels for pyarrow
```

Chuỗi nguyên nhân:

1. Streamlit Cloud dựng môi trường bằng **Python 3.14**.
2. `streamlit==1.51.0` ghim `pyarrow<22`.
3. pyarrow chỉ có wheel cho Python 3.14 **từ bản 22 trở lên**.
4. Không có wheel phù hợp → pip build từ mã nguồn → thiếu `cmake` → hỏng.

Cách khắc phục: nâng lên `streamlit==1.65.0` (cho phép `pyarrow<26`) và ghim
`pyarrow==25.0.1`. Bộ pin hiện tại đã chạy thử trên cả 3.13 lẫn 3.14.

Lưu ý khi tuỳ biến giao diện: Streamlit ≤1.5x đánh dấu tab bằng
`[data-baseweb="tab"]`, còn ≥1.6x dùng `[role="tab"]`. `src/theme.py` khai báo cả hai
nên CSS không vỡ khi nâng cấp.

### Vì sao phải ghim phiên bản chính xác

`artifacts/*.joblib` là **pickle** của các đối tượng scikit-learn và LightGBM. Nếu
Cloud cài phiên bản khác lúc huấn luyện, việc nạp mô hình có thể thất bại. Vì vậy
`requirements.txt` dùng `==` chứ không dùng `>=`.

Nếu sau này bạn nâng cấp thư viện ở máy, nhớ chạy lại `python train.py` và
`python tune.py` rồi cập nhật `requirements.txt` cho khớp.

### Lưu ý khi dùng trên lớp

- **App ngủ khi không ai dùng.** Lần truy cập đầu sau một thời gian dài sẽ mất
  khoảng 30 giây để thức dậy. Trước giờ dạy nên mở app trước vài phút.
- **Repo public thì dữ liệu cũng public.** Bộ `heart_disease.csv` là dữ liệu nhân tạo
  nên không có vấn đề riêng tư, nhưng hãy nhớ điều này nếu sau đó bạn thay bằng dữ
  liệu thật. Streamlit Cloud có hỗ trợ repo private, bạn kiểm tra hạn mức của tài
  khoản mình trong phần Settings.
- **Cập nhật app**: chỉ cần `git push`, Cloud tự deploy lại.

## Về tập train / test

Dự án chỉ có **một nguồn dữ liệu duy nhất**: `Dataset/heart_disease.csv`.
Việc chia train/test diễn ra **trong code** bằng `train_test_split(stratify=y)`,
không phải bằng hai file có sẵn như các cuộc thi Kaggle.

`train.py` là *tên script huấn luyện*, không phải file dữ liệu. Để nhìn thấy kết quả
chia tách, script xuất ra hai tệp tham khảo (không phải đầu vào của chương trình):

| Tệp | Nội dung |
|---|---|
| `artifacts/train_set.csv` | 8,000 dòng — mô hình học từ đây |
| `artifacts/test_set.csv` | 2,000 dòng — mô hình **chưa từng nhìn thấy** |

Cả hai giữ nguyên tỷ lệ mắc bệnh 20.0% nhờ `stratify=y`.

## Cấu trúc

| Tệp | Vai trò |
|---|---|
| `app.py` | Ứng dụng Streamlit, 6 tab |
| `notebooks/heart_disease_walkthrough.ipynb` | Notebook 57 ô, 7 bài học, 5 bài tập |
| `src/data.py` | Nạp dữ liệu (xử lý bẫy `"None"`) + pipeline tiền xử lý |
| `src/models.py` | Định nghĩa 8 mô hình đem so sánh |
| `train.py` | Huấn luyện, cross-validation, thí nghiệm đối chứng |
| `tune.py` | Tinh chỉnh siêu tham số bằng `GridSearchCV` |
| `src/theme.py` | Bảng màu, CSS và template biểu đồ dùng chung |
| `.streamlit/config.toml` | Theme sáng/tối của Streamlit |
| `requirements.txt` | Thư viện cho ứng dụng (Streamlit Cloud đọc file này) |
| `requirements-notebook.txt` | Thêm thư viện cho notebook |
| `packages.txt` | Gói hệ thống Linux cho Streamlit Cloud |
| `artifacts/` | Mô hình đã huấn luyện, `metrics.csv`, `tuning.csv`, tập train/test |

## Sáu tab của ứng dụng

1. **Tổng quan** — thẻ số liệu, phân bố nhãn, giá trị thiếu, phân phối từng biến
2. **Tiền xử lý** — quy tắc theo nhóm biến, pipeline chống rò rỉ dữ liệu
3. **So sánh mô hình** — bảng chỉ số, ROC, ma trận nhầm lẫn, độ quan trọng của biến
4. **Tinh chỉnh** — `GridSearchCV`, mức cải thiện so với biên độ nhiễu
5. **Chẩn đoán dữ liệu** — ba kiểm định về tín hiệu trong dữ liệu
6. **Dự đoán** — form nhập liệu và đối chiếu hai hồ sơ cực đoan

Dashboard chiếm trọn chiều rộng màn hình, trình bày thuần số liệu và biểu đồ. Toàn bộ
phần giảng giải nằm trong notebook để giáo viên chủ động dẫn dắt trên lớp.

Mọi thành phần tự vẽ trong `src/theme.py` đều không phụ thuộc theme: nền dùng `rgba`
trung tính, chữ dùng `inherit`, nền biểu đồ để trong suốt. Nhờ vậy đổi sáng/tối là
tức thì. (Không dùng `st.context.theme` vì API này báo sai đúng lúc người dùng chuyển
theme — xem streamlit#11920.)

## Kết quả huấn luyện

Mọi mô hình đều đạt ROC-AUC ≈ 0.50. Đây **không phải lỗi code** — bộ dữ liệu là dữ
liệu nhân tạo sinh ngẫu nhiên, các biến đầu vào độc lập thống kê với nhãn:

- Accuracy của mô hình tốt nhất bằng đúng Baseline đoán bừa (80%)
- Huấn luyện trên **nhãn xáo trộn** cho ROC-AUC 0.518 — *cao hơn* nhãn thật (0.498)
- Tương quan của mọi biến số với nhãn đều nằm trong ±0.02
- Random Forest sau `GridSearchCV` còn **tệ hơn** trên tập test (−0.0275)

Bộ dữ liệu này vì thế phù hợp để dạy **quy trình và tư duy phản biện**, không phù hợp
để rút ra kết luận y học.

## Một con số dễ đọc nhầm

Bộ dữ liệu chỉ thiếu **500 ô trên 200,000 ô**, ảnh hưởng **500 dòng (5.0%)**.

Nếu đọc bằng `pd.read_csv()` mặc định, chuỗi `"None"` ở cột `Alcohol Consumption` bị
hiểu thành `NaN` và con số phồng lên **2,933 dòng (29.3%)** — gấp gần 6 lần. Đây là ví
dụ tốt để dạy: một lỗi đọc dữ liệu làm sai lệch toàn bộ đánh giá chất lượng dữ liệu.

## Gợi ý khi dạy

Chạy tab 1→5 như một buổi ML bình thường để học sinh tự thấy con số 80% và thấy nó
"ngon". Tab 6 mới lật bài. Cú sốc đó dạy được nhiều hơn là nói trước từ đầu.
