# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:** Python3, Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Có thay đổi nội dung của các mục là các câu trả lời thay vì các câu hỏi như nguồn ban đầu.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

Nguồn: day1_lab_outputs/classification_predictions.json và visuals/classification_top5.png, ảnh sample_id = traffic (coco_image_id 210273, kích thước 640×428), dòng rank = 1.

1. Record hạng 1:

    class_id=468 ; class_name=cab ; rank=1 ; score=0.510915 ; taxonomy_name=ImageNet-1K

    (Record do yolo11n-cls.pt sinh ra, task = image_classification.)

2. Record này nói về cả bức ảnh hay một vật thể riêng?
   
    Record nói về cả bức ảnh. Lý do:

    Record chỉ có sample_id, kích thước ảnh, rank, class_id, class_name và score. Không có dòng nào xác định vị trí kiểu như là bbox_xyxy hay polygon_xy, nên không chỉ ra vị trí của vật nào.
    Cả 5 record của traffic cùng thuộc một ảnh và khác nhau ở rank. Nên có thể chúng là 5 ứng viên nhãn xếp hạng cho cùng một ảnh, không phải 5 vật thể khác nhau.

    Trong ảnh có nhiều xe buýt, ô tô con và người đi bộ nữa, nhưng đơn vị nhãn của tác vụ này vẫn là một lớp cho cả ảnh.

4. Class list do đâu mà có?

    Class list đi kèm taxonomy của checkpoint, không phải do model tự nghĩ ra. Record ghi taxonomy_name = ImageNet-1K và model_file = yolo11n-cls.pt. Model ở đây chỉ chấm điểm cho các lớp đã có sẵn trong danh sách ImageNet-1K và xếp hạng chúng. Nó không thể trả về một lớp nằm ngoài danh sách. Vì vậy mọi nhãn trong top-5 (cab, minibus, police_van, recreational_vehicle, streetcar) đều lấy từ taxonomy này.

5. Vì sao cần ghi cả class_id, class_name và taxonomy_name?

    Vì class_id (468) chỉ là số thứ tự trong một danh sách cụ thể. Không biết danh sách nào thì con số này không có nghĩa. Ví dụ id 468 của ImageNet-1K không trùng nghĩa với id 468 trong một taxonomy khác.
    class_name (cab) giúp người đọc hiểu được ngay, nhưng tên có thể trùng hoặc khác cách viết giữa các taxonomy.
    taxonomy_name (ImageNet-1K) cho biết cặp id–tên đó thuộc bộ lớp nào, nên giúp người khác có thể kiểm tra và tái lập lại được.

    Nên việc ghi đủ ba trường thì nhãn mới truy vết được và không bị hiểu nhầm giữa các bộ lớp vì mỗi trường sẽ có mỗi vai trò và mối quan hệ với nhau.

6. Ảnh có nhiều vật/chủ đề thì guideline cần nói rõ điều gì?
   
    Ảnh traffic có xe buýt, ô tô con, người đi bộ, xe máy. Để annotator chọn đúng một lớp, guideline cần nêu rõ:

    Tiêu chí chọn chủ thể chính: vật chiếm diện tích lớn nhất, nổi bật nhất hay ở trung tâm, hoặc nhãn theo cảnh chung (ví dụ “đường phố”) thay vì theo từng vật.
   
    Quy tắc xử lý khi hai chủ đề ngang nhau (ví dụ ưu tiên theo thứ tự cố định).
   
    Có cho phép lớp “khác / không rõ” hay không, và khi nào dùng.
   
    Quy tắc chọn khi taxonomy có nhiều lớp gần nghĩa (ví dụ như ở đây là cab, minibus, police_van...).

8. Vì sao không được lấy score của model làm ground truth?

    score (0.510915) chỉ là mức độ “tin” của model với dự đoán của nó. Biểu đồ cũng ghi trục là “độ tin cậy của mô hình (không phải chất  lượng ground truth)”.
   
    Ground truth là nhãn do người gán theo guideline rồi được reviewer kiểm tra. Score không phản ánh việc nhãn có đúng quy tắc hay không.
    Trên ảnh traffic, model chỉ tin khoảng 51% vào cab, còn minibus được 0.16. Mức tin này không cao. Ngay cả khi score cao, model vẫn có thể sai, nên score không thay được bước người kiểm tra.

    Điểm mơ hồ quan sát được: Ảnh traffic chủ yếu là xe buýt xanh và ô tô con, nhưng top-1 là cab (0.51), còn top-5 đều là các lớp xe cộ gần nghĩa. Đây là ví dụ cho thấy ảnh nhiều vật thể khó chọn đúng một lớp nếu guideline không nói rõ cách chọn.

## 2. Phát hiện vật thể

Nguồn: day1_lab_outputs/detection_predictions.json và visuals/detection_predictions.png, ảnh sample_id = kitchen (coco_image_id 397133, kích thước 640×427).

1. Record person(người đứng giữa trong ảnh):

    class_name=person ; score=0.912625 ; bbox_xyxy=[385.33, 69.24, 498.92, 348.92] ; bbox_width=113.58 ; bbox_height=279.68

    (Ngoài ra record còn ghi class_id=0, taxonomy_name=COCO-80, model_file=yolo11n.pt, score_threshold=0.35.)

2. Giải thích box bằng lời thường

    bbox_xyxy là 4 số theo thứ tự [x1, y1, x2, y2], đơn vị là pixel (coordinate_unit = pixel). Gốc tọa độ (0, 0)ở góc trên-trái của ảnh, x tăng sang phải, y tăng xuống dưới.

    Góc trên-trái là (x1, y1) = (385.33, 69.24): từ góc trên-trái của ảnh, đi sang phải khoảng 385 pixel rồi đi xuống khoảng 69 pixel thì tới góc này.
   
    Góc dưới-phải là (x2, y2) = (498.92, 348.92): từ góc trên-trái của ảnh, đi sang phải khoảng 499 pixel rồi đi xuống khoảng 349 pixel.
   
    Box rộng khoảng 113.58 pixel (498.92 − 385.33) và cao khoảng 279.68 pixel (348.92 − 69.24). Số có thể lệch một chút so với file vì đã làm tròn.
   
    Box cao và hẹp, khớp với người đang đứng thẳng trong ảnh. 

4. Hai ngưỡng và số prediction (lấy từ output ô so sánh ngưỡng trong notebook, ảnh kitchen)

    Ngưỡng	Số prediction
   
    0.20	17
   
    0.35 	11
   
    0.60	6

    Hạ ngưỡng từ 0.60 xuống 0.20 thì box tăng từ 6 lên 17. Sau đó đếm lại trong JSON thì ở 0.35 đúng 11 record, khớp với notebook.

5. Khi hạ ngưỡng, reviewer phải làm thêm gì?

    Hạ ngưỡng nghĩa là giữ thêm nhiều box mà model ít chắc chắn. Nhiều box hơn thì reviewer phải xem thêm từng cái. Việc thêm gồm:

    Kiểm tra box có thật sự là vật thể hay model đoán bừa (nhiều khả năng các box score thấp sai nhiều hơn).
    Kiểm tra tên lớp có đúng không, ví dụ cup hay bowl.
    Kiểm tra box có ôm sát vật không và xóa box trùng nhau hoặc box thừa.

    Ngưỡng cao thì ít việc kiểm tra hơn nhưng có thể sót vật thật. Ngưỡng thấp thì ngược lại. Theo tôi, ngưỡng chỉ là cách lọc prediction cho dễ xem, không phải quy tắc để quyết định cái gì là đúng.

6. Một quy tắc box để annotator làm theo

    “Mỗi vật thể thuộc danh sách lớp chỉ vẽ một box hình chữ nhật, ôm sát phần nhìn thấy của vật đó từ mép ngoài cùng bên này sang bên kia, không chừa khoảng trống thừa và không lấn sang vật khác.”

    Quy tắc này áp dụng được cho mọi ảnh vì nói về cách vẽ box, không nhắc đến ảnh kitchen cụ thể.

7. Vật bị che hoặc bị cắt mép trên ảnh kitchen

    Ở mép trái ảnh có box person 0.61, tọa độ [0.12, 263.18, 61.55, 310.78]. Box này nằm sát mép trái (x1 gần 0) và trên hình chỉ thấy một phần nhỏ giống cánh tay/bàn tay, không thấy cả người. Box oven 0.69 cũng bắt đầu từ x1 = 0.11, tức cũng sát mép.

    Nên chia thành hai phần:

    Nếu viết thành quy tắc trong guideline: vật bị cắt ở mép ảnh thì chỉ vẽ box quanh phần nhìn thấy, không đoán và vẽ thêm phần ngoài ảnh.
   
    Đưa lên hỏi Lab Coach (escalation): trường hợp chỉ thấy một bàn tay thì có tính là person hay không. Vì chưa biết guideline cho phép lấy bao nhiêu phần vật thể thì được gán nhãn. Nếu tự quyết thì khác gì tự bịa quy tắc, và mỗi annotator có thể làm một kiểu.

Vì thế tôi chọn escalation cho ca cụ thể này, rồi sau khi có câu trả lời mới viết thêm vào guideline.

## 3. Phân đoạn theo từng đối tượng

Nguồn: day1_lab_outputs/segmentation_predictions.json và visuals/segmentation_prediction.png, ảnh sample_id = kitchen (coco_image_id 397133, 640×427). 

Chọn lại người đứng giữa ảnh (record person 0.90) để dễ so với box ở Mục 2.

1. Record:

    instance_id=kitchen-001 ; class_name=person ; score=0.899318 ; polygon_point_count=348 ;
    5 điểm đầu của polygon_xy=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]

    (Record còn có class_id=0, taxonomy_name=COCO-80, model_file=yolo11n-seg.pt, bbox_xyxy=[385.45, 66.44, 498.02, 348.58], coordinate_unit=pixel.)

2. So với box, polygon cho thêm thông tin gì về hình dạng?

    Box chỉ cho biết vật nằm trong một hình chữ nhật. Polygon thì đi theo viền thật của vật nên cho biết hình dạng: đầu tròn, vai, hai tay chống hông, khoảng trống giữa hai chân.

    Nếu như thử tính diện tích để kiểm tra:

    Diện tích box ≈ 31,760 pixel².
   
    Diện tích đa giác ≈ 18,170 pixel² (tự tính bằng công thức diện tích đa giác từ 348 điểm).
   
    Vậy đa giác chỉ chiếm khoảng 57% box. Phần còn lại trong box (khoảng 43%) là nền hoặc đồ vật khác nằm sau người, ví dụ tường, xoong chảo, góc bếp.
   
    Nếu chỉ dùng box thì phần này bị tính nhầm là của người. Đây là chỗ polygon cho thông tin hơn box. Con số 57% là tự tính, không phải trường có sẵn trong file.

4. instance_id dùng để làm gì?

    instance_id là cái tên để phân biệt từng đối tượng riêng trong cùng một ảnh. Ví dụ trong ảnh kitchen có hai record cùng là person nhưng khác instance: kitchen-001 (người đứng giữa) và kitchen-009 (phần nhỏ ở mép trái). Có instance_id thì cho biết hai mask này là hai đối tượng chứ không phải một.

    Câu “không phải class ID và không phải tracking ID” tôi hiểu như sau:

    Không phải class ID: class ID (class_id=0) cho biết loại vật (person). Nhiều người khác nhau đều có cùng class ID = 0, còn instance_id thì mỗi người một cái.
   
    Không phải tracking ID: tracking ID dùng trong video để biết cùng một người ở frame này và frame sau. Ở đây chỉ có ảnh đơn, kitchen-001 chỉ là số thứ tự trong ảnh đó, không nói gì về việc “đó là cùng một người” ở ảnh khác.

6. Một quy tắc vẽ biên mask để annotator làm theo

    “Vẽ biên mask bám sát theo mép nhìn thấy của đối tượng, chỉ gồm các pixel của chính đối tượng đó. Không lấy phần nền hoặc vật khác ở giữa (ví dụ khoảng trống giữa hai chân, giữa tay và thân). Khi hai vật chạm nhau thì mỗi mask dừng ở đường tiếp xúc, không chồng lên nhau.”

    Quy tắc này nói về cách vẽ nên áp dụng được cho mọi ảnh.

7. Chỗ biên mờ / hai vật chạm nhau / bị che trên ảnh kitchen
   
    Trên ảnh segmentation_prediction.png, chỗ ở góc trái dưới: mask oven 0.49 (màu tím) và mask dining table 0.58 (màu xanh) nằm sát nhau và chồng lên nhau một phần, thêm các mask bowl và person 0.43 cũng dồn vào cùng chỗ.

   Theo record, oven có bbox_xyxy=[0.25, 188.95, 194.35, 294.62] và dining table có bbox_xyxy=[0.97, 234.74, 345.86, 423.87], hai box này cũng chồng nhau ở khoảng y = 235–295. Chỗ này khó vì nhiều vật chạm nhau và tối, khó thấy đường biên.

    Tôi chọn dùng guideline để xử lý chỗ này. Lý do là hai vật chạm nhau là chuyện gặp rất nhiều ở các ảnh khác, nên viết một quy tắc chung (mỗi mask dừng ở đường tiếp xúc, không chồng nhau) là đủ để annotator làm theo. Và chỉ escalation nếu gặp ca mà quy tắc này vẫn không trả lời được, ví dụ vật bị che gần hết nên không biết biên ở đâu thì không tự đoán.

## 4. Vòng đời và kiểm tra chất lượng

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho cả ảnh, lấy từ taxonomy (ImageNet-1K), do người chọn theo guideline. | Ảnh traffic: top-1 là cab (score 0.51) dù ảnh có nhiều xe buýt và ô tô con, còn minibus 0.16, police_van 0.09 là các lớp xe gần nghĩa. Ảnh kitchen: top-1 là gong (0.42), nghe không giống nhãn cho một bức ảnh nhà bếp. Score thấp hoặc lớp lạ nghĩa là model không chắc, không phải bằng chứng nhãn đúng. | Chọn một lớp theo guideline (tiêu chí chủ thể chính, cách xử lý ảnh nhiều vật). Không chép score hay top-1 của model làm nhãn. Ca khó thì escalation. | Xem lớp được gán có khớp guideline và có nằm trong taxonomy không. Xem các ảnh nhiều chủ đề có chọn nhất quán không. Xem có ai lấy prediction của model làm đáp án không. |
| Phát hiện vật thể | Mỗi vật thể một box [x1, y1, x2, y2] (pixel) kèm một lớp (COCO-80), do người vẽ theo guideline. | Ảnh kitchen: person 0.61 sát mép trái, bbox_xyxy=[0.12, 263.18, 61.55, 310.78], chỉ thấy một phần nhỏ giống bàn tay hoặc cánh tay, không rõ có tính là người không. cup 0.45 và bowl 0.50 chồng lên nhau ở giữa-trái ảnh, khó thấy rõ box nào ôm vật nào. Số box còn phụ thuộc ngưỡng: 17 box (0.20), 11 (0.35), 6 (0.60). | Vẽ một box ôm sát phần nhìn thấy của từng vật theo quy tắc box (không đoán phần ngoài ảnh). Gặp ca như “chỉ thấy bàn tay” thì hỏi Lab Coach, không tự quyết. | Xem có box nào thừa, bị sót hoặc trùng nhau không. Xem box có ôm sát vật không, tên lớp có đúng không (cup hay bowl). Không dùng ngưỡng để quyết định đúng sai. |
| Instance segmentation | Mỗi đối tượng một polygon (mask) bám theo viền cùng một lớp, các đối tượng cùng lớp được tách riêng bằng instance_id. | Ảnh kitchen: mask potted plant 0.63 (kitchen-004) nằm ở góc trên-trái, trên hình giống phủ lên bó lá khô treo trên cao hơn là cây chậu. Mask oven 0.49 và dining table 0.58 chạm và chồng nhau ở góc trái dưới. Với người đứng giữa (kitchen-001), đa giác chiếm khoảng 57% diện tích box (tự tính), nên box thừa nhiều nền. | Vẽ biên bám theo mép thấy được, chỉ gồm pixel của chính vật đó. Chỗ hai vật chạm nhau thì mỗi mask dừng ở đường tiếp xúc. Gán đúng lớp và tách từng instance riêng. | Xem biên mask có bám đúng mép không (có lấn ra nền hoặc sót phần vật không). Xem có mask nào chồng nhau không. Xem lớp có đúng không (ví dụ potted plant). Xem hai vật cùng lớp có bị gộp thành một instance không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của tôi cho bài lab này (1 câu):

    Tôi chỉ dùng ba ảnh COCO công khai mà notebook đã cố định (có kiểm tra checksum), không đưa ảnh riêng, dữ liệu khách hàng hay thông tin cá nhân (họ tên, MSSV, email, số điện thoại) vào báo cáo, file output hay repo, và họ tên/MSSV chỉ nằm trong tên repository.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, mình sẽ dừng và báo cho ai?

    Tôi sẽ dừng lại, không chạy tiếp, không commit hay đẩy lên GitHub, rồi báo cho Lab Coach (hoặc giảng viên phụ trách) để hỏi cách xử lý. Không tự quyết hay tự xóa/sửa dữ liệu đó.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
