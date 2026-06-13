# BÁO CÁO PHÂN TÍCH CHUYÊN SÂU: TRỌNG SỐ, THÔNG SỐ ĐÁNH GIÁ & KỸ THUẬT CHỐNG OVERFITTING TẠI CÁC MÔ HÌNH DL
*(Tài liệu hướng dẫn chi tiết phục vụ báo cáo và trả lời câu hỏi phản biện của Giảng viên hướng dẫn)*

Báo cáo này cung cấp cái nhìn chi tiết và chuyên sâu về mặt kỹ thuật, toán học và kiến trúc của hai mô hình Học sâu (Deep Learning) thuộc hai phân khu đồ án:
1. **Dự án 1 (DAMH_01 - OMR):** Mô hình **CNN** phân loại nhị phân ô tô trắc nghiệm (Marked / Unmarked).
2. **Dự án 2 (DAMH_02 - LSTM):** Mô hình **Bidirectional LSTM + Attention** phân loại 3 lớp cảm xúc văn bản song ngữ (Tích cực, Tiêu cực, Bình thường).

---

## PHẦN 1: PHÂN TÍCH CHI TIẾT KIẾN TRÚC & TRỌNG SỐ (MODEL ARCHITECTURE & WEIGHTS)

Trọng số của mô hình (Model Weights) chính là các tham số khả học (trainable parameters) gồm ma trận trọng số liên kết (kernels/weights) và vectơ chệch (biases). Trọng số được khởi tạo ngẫu nhiên và tối ưu hóa qua thuật toán lan truyền ngược (Backpropagation).

### 1.1. Mô hình OMR CNN (Phân loại ô trắc nghiệm)
Mô hình CNN được thiết kế nhằm học các đặc trưng không gian (spatial features) từ ảnh đầu vào xám cỡ $64 \times 64 \times 1$ (kênh đơn).

#### 1. Các tầng Convolutional (Conv2D)
Mỗi bộ lọc (filter) trong tầng Conv2D hoạt động như một cửa sổ trượt (sliding window) áp dụng phép nhân ma trận chập trên ảnh đầu vào để tạo ra các Bản đồ đặc trưng (Feature Maps).
*   **Conv2D 1:** $32$ bộ lọc kích thước $3 \times 3$, hàm kích hoạt **ReLU**.
    *   *Số lượng trọng số:* Kích thước nhân nhân với số kênh đầu vào nhân với số bộ lọc cộng với bias cho mỗi bộ lọc.
        $$\text{Params} = (3 \times 3 \times 1 + 1) \times 32 = 320 \text{ tham số}$$
    *   *Nhiệm vụ đặc trưng:* Học các cạnh cơ bản (mép đường bao, độ cong của ô tròn trắc nghiệm).
*   **Conv2D 2:** $64$ bộ lọc kích thước $3 \times 3$, hàm kích hoạt **ReLU**.
    *   *Số lượng trọng số:* Nhận đầu vào là $32$ kênh đặc trưng từ tầng trước.
        $$\text{Params} = (3 \times 3 \times 32 + 1) \times 64 = 18,496 \text{ tham số}$$
    *   *Nhiệm vụ đặc trưng:* Học các tổ hợp đặc trưng phức tạp hơn như các góc chéo, vết bút chì tô lem nhẹ bên trong đường bao.
*   **Conv2D 3:** $128$ bộ lọc kích thước $3 \times 3$, hàm kích hoạt **ReLU**.
    *   *Số lượng trọng số:* Nhận đầu vào là $64$ kênh đặc trưng từ tầng trước.
        $$\text{Params} = (3 \times 3 \times 64 + 1) \times 128 = 73,856 \text{ tham số}$$
    *   *Nhiệm vụ đặc trưng:* Trích xuất đặc trưng bậc cao về mật độ pixel tối của chì, sự phân bố không gian biểu diễn vết tô hoàn chỉnh hay trống.

#### 2. Các tầng MaxPooling2D
MaxPooling2D kích thước $2 \times 2$ quét qua các Feature Maps đầu ra của Conv2D và chỉ giữ lại giá trị pixel lớn nhất (đặc trưng nổi bật nhất) trong mỗi vùng $2 \times 2$.
*   *Tác dụng:* Giảm kích thước không gian đi một nửa (từ $64 \times 64$ xuống $32 \times 32$, rồi $16 \times 16$, rồi $8 \times 8$). Điều này làm giảm số lượng trọng số cần học ở các tầng tiếp theo xuống 4 lần, hạn chế sự bùng nổ tham số và chống overfitting.
*   *Lưu ý:* MaxPooling **không chứa trọng số khả học (0 trainable parameters)**.

#### 3. Tầng Flatten & Tầng Dense (Fully Connected)
*   **Flatten:** Chuyển đổi ma trận đặc trưng đầu ra $8 \times 8 \times 128$ thành một vectơ phẳng 1 chiều kích thước $8192$ để chuẩn bị đưa vào lớp Dense Classifier. Không chứa trọng số khả học.
*   **Dense 1 (128 units):** Nhào trộn toàn bộ $8192$ chiều đặc trưng phẳng để tìm các mối liên kết phi tuyến tính.
    $$\text{Params} = (8192 + 1) \times 128 = 1,048,704 \text{ tham số}$$
    Hàm kích hoạt **ReLU** ($f(x) = \max(0, x)$) được sử dụng để lọc bỏ các giá trị âm (tín hiệu không kích hoạt/nhiễu), giúp mô hình hội tụ nhanh hơn.
*   **Dense Output (1 unit):** Dự đoán xác suất nhị phân đầu ra.
    $$\text{Params} = (128 + 1) \times 1 = 129 \text{ tham số}$$
    Hàm kích hoạt **Sigmoid** ($f(x) = \frac{1}{1 + e^{-x}}$) nén đầu ra về khoảng $(0, 1)$.
    *   Tiệm cận về $0$: Xác suất cao là ô tròn đã được tô (Marked) - do giá trị pixel tối chiếm ưu thế.
    *   Tiệm cận về $1$: Xác suất cao là ô tròn trống (Unmarked) - do giá trị pixel sáng chiếm ưu thế.

---

### 1.2. Mô hình LSTM Emotion (Phân loại cảm xúc văn bản)
Mô hình này làm việc với dữ liệu dạng chuỗi thời gian (Sequential data) để phân tích cảm xúc từ các câu văn bản song ngữ có chiều dài chuẩn hóa $T = 150$ từ.

#### 1. Tầng Embedding
Mỗi từ trong từ điển được đại diện bằng một chỉ số số nguyên (token ID). Tầng Embedding chuyển đổi token ID này thành một vectơ số thực phân phối liên tục trong không gian $32$ chiều.
*   *Kích thước từ điển:* $V = 10,000$ từ.
*   *Số lượng trọng số:*
    $$\text{Params} = V \times \text{Embedding Dim} = 10,000 \times 32 = 320,000 \text{ tham số}$$
*   *Ý nghĩa:* Ma trận này học mối quan hệ ngữ nghĩa giữa các từ. Các từ có ngữ cảnh sử dụng tương đương (ví dụ: "vui", "hạnh phúc", "happy") sẽ được huấn luyện để kéo lại gần nhau trong không gian vectơ 32 chiều này.

#### 2. Tầng Bidirectional LSTM
Mô hình sử dụng mạng LSTM hai chiều (Bidirectional LSTM) gồm một mạng LSTM đọc xuôi từ đầu đến cuối câu (Forward LSTM) và một mạng LSTM đọc ngược từ cuối lên đầu câu (Backward LSTM). Điều này giúp mô hình nắm bắt ngữ cảnh từ cả quá khứ và tương lai tại bất kỳ thời điểm nào.
*   *Số lượng nơ-ron trạng thái ẩn (hidden units):* $h = 32$.
*   *Cấu trúc bên trong mỗi phần tử LSTM:* Gồm $4$ cổng chính để kiểm soát luồng thông tin đi qua thời gian:
    *   **Cổng quên (Forget gate - $f_t$):** Quyết định bỏ đi thông tin cũ nào từ trạng thái tế bào trước ($C_{t-1}$).
    *   **Cổng vào (Input gate - $i_t$):** Quyết định nạp thêm thông tin mới nào vào trạng thái tế bào hiện tại.
    *   **Trạng thái tế bào ứng viên ($\tilde{C}_t$):** Tạo thông tin mới để chuẩn bị đưa vào tế bào.
    *   **Cổng ra (Output gate - $o_t$):** Quyết định giá trị đầu ra ẩn $h_t$.
*   *Số lượng trọng số của một LSTM đơn chiều:*
    $$\text{Params} = 4 \times [(\text{Embedding Dim} + h) \times h + h] = 4 \times [(32 + 32) \times 32 + 32] = 8,320 \text{ tham số}$$
*   *Vì chạy hai chiều (Bidirectional) độc lập:* Tổng tham số là $8,320 \times 2 = 16,640$. Đầu ra của tầng này tại mỗi bước thời gian là vectơ ghép có kích thước $32 \times 2 = 64$ chiều.

#### 3. Cơ chế Self-Attention (Tự chú ý)
Cơ chế này gán một trọng số chú ý $\alpha_t \in [0, 1]$ cho trạng thái ẩn $h_t$ của từ tại vị trí thứ $t$ trong câu, sao cho tổng các trọng số bằng 1: $\sum_{t=1}^{T} \alpha_t = 1$.
*   *Ý nghĩa:* Giúp mô hình tập trung sự chú ý vào các từ mang tính cảm xúc cốt lõi (như "tuyệt vời", "chán", "sad", "bad") và giảm bớt sự ảnh hưởng của các từ nối, từ dừng ít mang thông tin cảm xúc ("là", "thì", "and", "the").
*   *Trọng số khả học:* Các ma trận chiếu $W_{att}$ dùng để tính toán điểm tương quan giữa các trạng thái ẩn.

#### 4. GlobalMaxPooling1D & Dense Classifier
*   **GlobalMaxPooling1D:** Chỉ lấy giá trị đặc trưng lớn nhất trên toàn bộ chiều thời gian của câu sau Attention.
*   **Dense 1 (32 units):** Tầng ẩn hoàn toàn kết nối với hàm kích hoạt **ReLU** để tổng hợp các đặc trưng ngữ nghĩa phức tạp.
    $$\text{Params} = (64 + 1) \times 32 = 2,080 \text{ tham số}$$
*   **Dense Output (3 units):** Phân loại cảm xúc vào 3 lớp (Tích cực, Tiêu cực, Trung tính).
    $$\text{Params} = (32 + 1) \times 3 = 99 \text{ tham số}$$
    Sử dụng hàm kích hoạt **Softmax**:
    $$P(y = i | x) = \frac{e^{z_i}}{\sum_{j=1}^{3} e^{z_j}}$$
    Softmax ép các giá trị đầu ra (logits) thành một phân phối xác suất hợp lệ (tổng xác suất của 3 lớp đúng bằng 100%). Lớp nào có xác suất cao nhất sẽ được chọn làm nhãn cảm xúc dự đoán.

---

## PHẦN 2: CÁC THÔNG SỐ ĐÁNH GIÁ MÔ HÌNH (EVALUATION METRICS)

Để đánh giá một cách khoa học hiệu năng của các mô hình Deep Learning trên tập kiểm thử độc lập (Test Set), chúng ta dựa vào các chỉ số toán học sau:

### 2.1. Hàm mất mát (Loss Function)
Loss function đo lường sự sai lệch giữa giá trị dự đoán của mô hình và nhãn thực tế. Mục tiêu huấn luyện là giảm thiểu tối đa giá trị này thông qua Gradient Descent.
*   **Binary Crossentropy (Dành cho OMR CNN):**
    $$L = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
    *Trong đó:* $y_i \in \{0, 1\}$ là nhãn thực tế (đã tô hoặc trống), $\hat{y}_i \in [0, 1]$ là xác suất dự đoán trống. Hàm này phạt nặng các dự đoán sai lệch lớn bằng hàm logarit.
*   **Sparse Categorical Crossentropy (Dành cho LSTM Emotion):**
    $$L = - \frac{1}{N} \sum_{i=1}^{N} \log(P(y_i | x_i))$$
    Sử dụng cho bài toán phân loại đa lớp khi nhãn thực tế được mã hóa dưới dạng số nguyên trực tiếp ($0, 1, 2$) thay vì vectơ one-hot. Giúp tiết kiệm bộ nhớ khi tính toán loss cho tập từ điển lớn hoặc nhiều lớp.

### 2.2. Ma trận nhầm lẫn (Confusion Matrix)
Là bảng biểu diễn chi tiết kết quả phân loại của mô hình đối với các mẫu kiểm thử:
*   **True Positive (TP):** Thực tế dương tính (Marked / Positive), mô hình đoán đúng là dương tính.
*   **True Negative (TN):** Thực tế âm tính (Unmarked / Negative), mô hình đoán đúng là âm tính.
*   **False Positive (FP - Sai lầm loại 1):** Thực tế âm tính, mô hình đoán sai thành dương tính (Báo động giả).
*   **False Negative (FN - Sai lầm loại 2):** Thực tế dương tính, mô hình đoán sai thành âm tính (Bỏ sót nhãn).

Từ Ma trận nhầm lẫn này, ta tính toán các chỉ số chất lượng:

#### 1. Độ chính xác chung (Accuracy)
Tỷ lệ số dự đoán đúng trên tổng số dự đoán.
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
*Hạn chế:* Chỉ phản ánh chính xác khi tập dữ liệu hoàn toàn cân bằng giữa các lớp. Nếu dữ liệu bị lệch (imbalanced data), chỉ số này sẽ bị ảo (Accuracy Paradox).

#### 2. Độ chuẩn xác (Precision)
Tỷ lệ số mẫu thực sự dương tính trong tổng số mẫu được mô hình dự đoán là dương tính.
$$\text{Precision} = \frac{TP}{TP + FP}$$
*Ý nghĩa:* Đánh giá độ tin cậy của dự đoán dương tính. Precision cao nghĩa là mô hình rất ít khi bị báo động giả (FP thấp).

#### 3. Độ phủ / Độ nhạy (Recall / Sensitivity)
Tỷ lệ mẫu được dự đoán đúng là dương tính trên tổng số mẫu thực sự dương tính trong tập dữ liệu.
$$\text{Recall} = \frac{TP}{TP + FN}$$
*Ý nghĩa:* Đánh giá khả năng tìm kiếm của mô hình. Recall cao nghĩa là mô hình rất ít khi bỏ sót các nhãn dương tính thực tế (FN thấp).

#### 4. Điểm F1 (F1-Score)
Trung bình điều hòa (harmonic mean) giữa Precision và Recall.
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
*Ý nghĩa:* Đây là thước đo hiệu năng cốt lõi cho mô hình phân loại cảm xúc (LSTM Emotion) vì tập dữ liệu văn bản thường bị mất cân bằng (Neutral nhiều hơn Positive/Negative). Điểm F1-Score trung bình cao là minh chứng rõ ràng nhất rằng mô hình nhận diện tốt cả 3 lớp cảm xúc, không bị thiên vị nhãn nào.

---

## PHẦN 3: PHÂN TÍCH CÁC THÔNG SỐ CHỐNG OVERFITTING (CHỐNG HỌC VẸT)

Overfitting xảy ra khi mô hình học thuộc lòng cả các chi tiết nhiễu (noise) trong tập huấn luyện, làm cho loss trên tập Train rất thấp nhưng loss trên tập Validation/Test lại tăng vọt. Dưới đây là cách hoạt động cụ thể của các tham số cấu hình trong đồ án nhằm ngăn chặn hiện tượng này:

| Thông số / Kỹ thuật | Cơ chế hoạt động chống Overfitting | Giá trị áp dụng trong Đồ án |
| :--- | :--- | :--- |
| **Dropout** | Trong quá trình train, ngẫu nhiên ngắt kết nối (set đầu ra về 0) một tỷ lệ nơ-ron ẩn. Ép mô hình không được dựa vào một vài nơ-ron nổi trội, buộc các nơ-ron khác phải học cách cộng tác để tìm đặc trưng dự phòng. | `Dropout(0.5)` ở tầng Dense của cả hai mô hình; và `dropout=0.3` ở lớp LSTM. |
| **SpatialDropout1D** | Khác với Dropout thường (tắt các phần tử ngẫu nhiên trong vectơ embedding), SpatialDropout1D tắt hẳn toàn bộ kênh đặc trưng (cả vectơ từ). Ép mô hình phải học cảm xúc của cả câu từ ngữ cảnh chung chứ không phụ thuộc vào sự xuất hiện của một từ duy nhất. | `SpatialDropout1D(0.4)` ngay sau tầng Embedding của mô hình LSTM. |
| **Recurrent Dropout** | Áp dụng riêng cho các liên kết hồi quy (hidden state từ bước $t-1$ sang $t$) trong LSTM. Tắt ngẫu nhiên luồng thông tin nhớ qua thời gian để ngăn nơ-ron ghi nhớ quá sâu các chuỗi từ cụ thể của tập train. | `recurrent_dropout=0.3` cấu hình bên trong lớp LSTM. |
| **L2 Regularization (Weight Decay)** | Cộng thêm số hạng phạt bằng bình phương độ lớn của trọng số vào hàm Loss: $L_{new} = L + \lambda \sum w^2$. Ép các trọng số $w$ tiến dần về 0 và mịn hơn. Ngăn chặn trọng số cực đoan làm méo quyết định của mạng nơ-ron. | Cấu hình tham số $\lambda = 10^{-4}$ (`l2(1e-4)`) cho Embedding/LSTM kernels và $\lambda = 0.01$ (`l2(0.01)`) cho Dense 1. |
| **Embedding Dimension** | Số chiều biểu diễn từ. Nếu số chiều quá lớn (ví dụ 300), mô hình có quá nhiều không gian bộ nhớ để học thuộc lòng. Giới hạn số chiều nhỏ ép mô hình phải khái quát hóa các từ đồng nghĩa vào chung nhóm. | Cấu hình kích thước nhỏ gọn `EMBEDDING_DIM = 32`. |
| **Early Stopping** | Giám sát `val_loss` sau mỗi epoch. Nếu `val_loss` không giảm trong $p$ epochs liên tiếp (patience), dừng huấn luyện ngay lập tức và khôi phục lại bộ trọng số tốt nhất trước khi overfitting bắt đầu. | Cấu hình `patience=4` cho OMR CNN; và `patience=10` cho LSTM Emotion với `restore_best_weights=True`. |
| **Reduce Learning Rate on Plateau** | Tự động nhân Learning Rate với một hệ số (factor) khi `val_loss` đi ngang. Giúp mô hình không nhảy quá xa qua điểm cực tiểu mà tinh chỉnh mịn màng ở giai đoạn cuối, giúp đường loss hội tụ mượt mà. | Cấu hình `factor=0.5`, `patience=4` trong mô hình LSTM. |

---

## PHẦN 4: BỘ CÂU HỎI & TRẢ LỜI PHẢN BIỆN THƯỜNG GẶP (Q&A CHEAT SHEET)

Dưới đây là các câu hỏi mà các thầy cô trong Hội đồng phản biện thường xuyên đặt ra cho sinh viên làm về đề tài CNN và LSTM, kèm theo gợi ý trả lời thông minh, học thuật nhất:

### Q1: Trọng số của mô hình (Model Weights) được cập nhật như thế nào trong quá trình huấn luyện?
*   **Trả lời khoa học:** 
    > "Thưa thầy cô, quá trình huấn luyện diễn ra qua 3 bước lặp lại liên tục:
    > 1. **Lan truyền xuôi (Forward Propagation):** Dữ liệu truyền qua các lớp để tính ra dự đoán đầu ra $\hat{y}$ và giá trị sai số thông qua hàm mất mát (Loss).
    > 2. **Lan truyền ngược (Backpropagation):** Tính đạo hàm riêng của hàm mất mát đối với từng trọng số ở mỗi tầng ($\frac{\partial L}{\partial w}$) bằng quy tắc chuỗi (Chain Rule). Giá trị đạo hàm này chính là gradient (độ dốc).
    > 3. **Cập nhật trọng số:** Thuật toán tối ưu hóa (Optimizer - ví dụ như Adam) sẽ cập nhật các trọng số theo hướng ngược chiều với gradient để giảm thiểu loss:
    >    $$w_{new} = w_{old} - \eta \cdot \text{Update}(gradient)$$
    >    Trong đó $\eta$ là tỷ lệ học (learning rate)."

### Q2: Tại sao em lại sử dụng hàm kích hoạt ReLU ở tầng ẩn và Sigmoid/Softmax ở tầng đầu ra?
*   **Trả lời khoa học:**
    > "Thưa thầy cô:
    > *   **Hàm ReLU** ($y = \max(0, x)$) được dùng ở các tầng ẩn vì nó giúp giải quyết triệt để vấn đề tiêu biến gradient (vanishing gradient problem) đối với các giá trị dương, tính toán cực kỳ nhanh vì chỉ là phép toán so sánh, và tạo ra tính thưa (sparsity) cho mạng nơ-ron giúp lọc bỏ nhiễu.
    > *   **Hàm Sigmoid** ($y = \frac{1}{1 + e^{-x}}$) được dùng ở tầng đầu ra của bài toán phân loại nhị phân (OMR CNN) vì nó nén đầu ra về khoảng xác suất $(0, 1)$, rất phù hợp để phân loại 2 lớp (đã tô hoặc trống).
    > *   **Hàm Softmax** được dùng ở đầu ra bài toán phân loại đa lớp (LSTM Emotion) vì nó chuẩn hóa logits của các lớp thành một phân phối xác suất có tổng bằng 1. Xác suất của lớp nào lớn nhất sẽ đại diện cho nhãn dự đoán."

### Q3: Trong bài toán LSTM Emotion, tại sao em lại sử dụng SpatialDropout1D thay vì Dropout thông thường ngay sau tầng Embedding?
*   **Trả lời khoa học:**
    > "Thưa thầy cô, Dropout thông thường hoạt động bằng cách tắt ngẫu nhiên các phần tử độc lập trong vectơ biểu diễn từ (word embedding). Việc này chỉ làm mất đi một số chiều thông tin của từ đó nhưng từ đó vẫn tồn tại trong câu, dẫn đến việc mô hình vẫn có thể học thuộc lòng.
    > Ngược lại, **SpatialDropout1D** hoạt động bằng cách tắt hoàn toàn toàn bộ vectơ biểu diễn của từ đó (tắt toàn bộ kênh đặc trưng 1D). Việc này tương đương với việc xóa bỏ ngẫu nhiên một số từ khỏi câu trong quá trình huấn luyện. Điều này buộc mạng LSTM phải học cách suy luận cảm xúc dựa vào ngữ cảnh của các từ xung quanh chứ không được phụ thuộc vào sự xuất hiện của riêng một từ khóa nào, giúp mô hình chống học vẹt tốt hơn rất nhiều."

### Q4: L2 Regularization chống Overfitting bằng cách nào? Cơ chế toán học đằng sau nó là gì?
*   **Trả lời khoa học:**
    > "Thưa thầy cô, L2 Regularization (còn gọi` Weight Decay) hoạt động bằng cách cộng thêm một lượng phạt tỷ lệ thuận với bình phương chuẩn L2 của ma trận trọng số vào hàm mất mát:
    > $$L_{new} = L_{old} + \lambda \sum w^2$$
    > Khi thực hiện lan truyền ngược để cập nhật trọng số, gradient của hàm phạt này sẽ ép các trọng số khả học phải giảm giá trị tuyệt đối về gần 0:
    > $$w_{new} = (1 - 2\eta\lambda) w_{old} - \eta \frac{\partial L_{old}}{\partial w}$$
    > Điều này ngăn chặn việc xuất hiện các trọng số có giá trị quá lớn hay quá cực đoan. Một mạng nơ-ron có các trọng số nhỏ và phân bố mịn sẽ có đường biên quyết định (decision boundary) mượt mà hơn, không bị uốn éo theo các nhiễu của tập Train, nhờ đó khái quát hóa tốt trên dữ liệu mới."

### Q5: Tại sao trong bài toán phân loại cảm xúc (Emotion Classification), em lại chọn F1-Score làm chỉ số đánh giá chính thay vì Accuracy?
*   **Trả lời khoa học:**
    > "Thưa thầy cô, trong thực tế dữ liệu ngôn ngữ tự nhiên, số lượng câu mang cảm xúc trung tính (Neutral) thường chiếm tỷ lệ áp đảo so với các câu mang cảm xúc cực đoan (Tích cực hoặc Tiêu cực).
    > Nếu chúng ta có một tập dữ liệu gồm 80% câu Neutral và chỉ 20% câu Positive/Negative, một mô hình phân loại cực kỳ kém (chỉ cần đoán bừa mọi câu là Neutral) cũng sẽ đạt được độ chính xác (Accuracy) lên tới 80%. Điều này gây ra ảo tưởng về sức mạnh của mô hình.
    > Chỉ số **F1-Score** là trung bình điều hòa của Precision và Recall, đo lường cân bằng cả khả năng dự đoán đúng và khả năng tránh bỏ sót trên từng lớp cảm xúc cụ thể. Điểm F1-Score trung bình cao là minh chứng rõ ràng nhất rằng mô hình nhận diện tốt cả 3 lớp cảm xúc, không bị thiên vị nhãn đa số."

### Q6: Early Stopping hoạt động như thế nào? Em chọn thông số patience dựa trên cơ sở nào?
*   **Trả lời khoa học:**
    > "Thưa thầy cô, Early Stopping hoạt động bằng cách theo dõi hàm loss trên tập kiểm định (`val_loss`) sau mỗi epoch.
    > *   Trong giai đoạn đầu, cả `train_loss` và `val_loss` đều giảm đều.
    > *   Đến một epoch tối ưu nào đó, mô hình bắt đầu học vẹt, dẫn đến `train_loss` tiếp tục giảm nhưng `val_loss` chững lại và bắt đầu tăng lên (ngóc đầu lên).
    > *   Thông số `patience` là số lượng epoch tối đa mà chúng ta cho phép mô hình tiếp tục chạy kể từ khi `val_loss` ngừng cải thiện. Nếu hết số epoch này mà `val_loss` vẫn không giảm thêm, mô hình sẽ tự động dừng huấn luyện.
    > *   Chúng tôi chọn `patience=4` cho CNN vì mô hình này hội tụ rất nhanh, và `patience=10` cho LSTM vì mô hình xử lý chuỗi phức tạp hơn và cần nhiều epoch hơn để khẳng định xem loss có thực sự chững lại hay không. Chúng tôi đặt `restore_best_weights=True` để lấy lại đúng bộ trọng số tại epoch có `val_loss` thấp nhất."
