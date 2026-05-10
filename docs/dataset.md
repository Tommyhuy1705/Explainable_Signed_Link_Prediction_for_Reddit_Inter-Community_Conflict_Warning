# Dataset Description

## Nguồn dữ liệu
- SNAP Reddit Hyperlinks
- File gốc: `soc-redditHyperlinks-body.tsv`, `soc-redditHyperlinks-title.tsv`

## Mô tả cột dữ liệu
- source_subreddit: cộng đồng nguồn.
- target_subreddit: cộng đồng đích.
- timestamp: thời gian liên kết.
- sentiment/label: dấu liên kết (positive/negative).
- post/link metadata: thông tin bài viết liên quan (nếu có).

## Quy ước xử lý
- Chuẩn hóa tên subreddit về lowercase.
- Loại bỏ bản ghi thiếu cột cốt lõi.
- Tách train/validation/test theo thời gian.
