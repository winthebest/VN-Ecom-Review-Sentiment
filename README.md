# Vietnamese Sentiment Analysis Platform

Nền tảng phân tích cảm xúc tiếng Việt sử dụng nhiều kiến trúc (CNN/GRU/LSTM/XGBoost/PhoBERT). Hệ thống gồm pipeline huấn luyện tự động, tracking bằng MLflow, và dịch vụ Flask cho suy luận trực tiếp với PhoBERT fine-tune.

## Tính năng chính
- Tiền xử lý chuyên sâu cho tiếng Việt (chuẩn hóa teencode, VnCoreNLP word segmentation, loại bỏ stopwords).
- Huấn luyện hàng loạt mô hình qua `pipeline.py`, log chỉ số và artifact lên MLflow.
- Lưu mô hình và dữ liệu đặc trưng trong các thư mục `models/` và `Data/processed/` để tái sử dụng.
- API Flask (`app/models/app.py`) cung cấp REST endpoint `/predict` và trang web demo để phân tích cảm xúc với PhoBERT.
- Dockerfile cho môi trường suy luận độc lập, giúp triển khai nhanh lên bất kỳ hạ tầng nào.

## Kiến trúc thư mục (rút gọn)
```
D:/Project
├── app/models/            # Flask service + Docker build context
├── configs/               # Cấu hình YAML cho từng mô hình
├── Data/
│   ├── raw/               # Dữ liệu gốc
│   └── processed/         # Numpy & torch tensors sau preprocessing
├── models/                # Checkpoints cho CNN/GRU/LSTM/XGBoost/PhoBERT
├── mlruns/                # MLflow tracking (file:// backend)
├── src/                   # Script preprocessing & train từng mô hình
└── pipeline.py            # Điều phối toàn bộ pipeline
```

## Chuẩn bị môi trường
1. Cài Python 3.10+ và pip.
2. Cài đặt phụ thuộc:
   ```bash
   pip install -r requirements.txt
   pip install -r app/models/requirements.txt   # nếu chạy service riêng
   ```
3. (Tuỳ chọn) đặt biến `JAVA_HOME` và tải mô hình VnCoreNLP như trong `app/models/utils.py` để chạy word segmentation.
4. Tải PhoBERT sẵn vào `cache/` theo đường dẫn khai báo nếu muốn tránh tải lại từ HuggingFace.

## Chạy pipeline huấn luyện
```bash
python pipeline.py
```
- Script tự chuyển vào `src/`, lần lượt chạy `preprocessing.py`, `CNN_train.py`, `GRU_train.py`, `LSTM_train.py`, `XGBoost_train.py`.
- Sau khi chạy xong mở MLflow UI:
  ```bash
  mlflow ui --backend-store-uri file:///D:/Project/mlruns
  ```
  rồi truy cập `http://localhost:5000` để xem kết quả.

### Huấn luyện từng mô hình thủ công
```bash
cd src
python preprocessing.py
python CNN_train.py
python GRU_train.py
python LSTM_train.py
python XGBoost_train.py
# PhoBERT_train.py có thể bật thêm nếu cần fine-tune lại
```

## Dịch vụ PhoBERT API
```bash
cd app/models
python app.py
```
- `POST /predict` với JSON `{ "text": "..." }` trả về nhãn `NEG/NEU/POS` và xác suất.
- `GET /` cung cấp form web đơn giản để thử nghiệm nhanh.

## Docker hoá dịch vụ
```bash
cd app/models
docker build -t vn-sentiment-phobert .
docker run -p 8000:8000 vn-sentiment-phobert
```
- Container đọc file `phobert_sentiment_classifier.pth` đặt cùng thư mục.
- Có thể đẩy image lên Docker Hub bằng `docker tag` và `docker push` (đã thực hiện trước đó).

## Đưa mã nguồn lên GitHub
1. Đăng nhập GitHub, tạo repository mới (ví dụ `vn-sentiment-analysis`), để trống README vì ta đã có.
2. Trên máy cục bộ:
   ```bash
   cd D:/Project
   git init
   git add .
   git commit -m "Initial commit: Vietnamese sentiment analysis platform"
   git remote add origin https://github.com/<username>/vn-sentiment-analysis.git
   git push -u origin main
   ```
3. Nếu repository đã tồn tại, chỉ cần `git remote set-url origin ...` trước khi push.

## Ghi chú
- Nên giữ riêng dữ liệu thô hoặc mô hình quá lớn (ví dụ `.pth`, `.npy`) bằng `.gitignore` nếu không muốn đẩy lên GitHub.
- Thư mục `mlruns/` có thể rất lớn; cân nhắc lưu trữ ngoài GitHub hoặc dùng DVC/MLflow Artifact Store.
- Khi triển khai production, thiết lập GPU/CPU phù hợp và kiểm soát version của tokenizer/model để đảm bảo tính tái lập.

