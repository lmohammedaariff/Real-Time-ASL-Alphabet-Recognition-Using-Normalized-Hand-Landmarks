# Model and evaluation

Training compares Logistic Regression, RBF SVM, Random Forest, and an MLP on the same stratified 80/20 split. Scaling is contained in sklearn pipelines for linear, SVM, and MLP models so it is fitted only on training data. The best macro-F1 model is exported as `models/best_model.joblib` with class metadata.

Reports include accuracy, macro precision/recall/F1, per-class classification report, confusion matrices, model comparison CSV, training time, and batch-average inference latency. A workspace run on 8,333 extracted public-dataset landmarks selected SVM with 89.62% holdout accuracy and 89.41% macro F1; exact results are in `reports/`. Generated model and dataset files stay local and are excluded from Git by default. Webcam FPS additionally depends on camera, processor, and MediaPipe runtime.

The current MLP uses early stopping. It does not include a neural-network framework checkpoint because the scikit-learn MLP is lightweight and comparable under a single reproducible sklearn pipeline.
