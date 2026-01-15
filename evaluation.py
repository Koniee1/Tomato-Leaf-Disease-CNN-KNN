# Line graphs
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(); plt.title('Line Graph of Accuracy')
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(); plt.title('Line graph of Loss')
plt.show()

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Bar Chart 
precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred)
x = np.arange(len(labels))
plt.bar(x - 0.2, precision, 0.2, label='Precision')
plt.bar(x, recall, 0.2, label='Recall')
plt.bar(x + 0.2, f1, 0.2, label='F1-score')
plt.xlabel('Class Label')
plt.ylabel('Metric Value')
plt.xticks(x, labels)
plt.legend()
plt.title("Metrics by Class")
plt.show()

# PCA visualisation
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Num of Components")
plt.ylabel("Explained Variance")
plt.grid(True)
plt.title("PCA Variance Explained")
plt.show()

# McNemar's Test on test data
cnn_probs = full_model.predict(test, verbose=1)
y_pred_A = np.argmax(cnn_probs, axis=1)
y_pred_B = y_test_pred

A_correct = (y_pred_A == y_test)
B_correct = (y_pred_B == y_test)

table = np.zeros((2, 2))
table[0, 0] = np.sum(A_correct & B_correct)
table[0, 1] = np.sum(A_correct & ~B_correct)
table[1, 0] = np.sum(~A_correct & B_correct)
table[1, 1] = np.sum(~A_correct & ~B_correct)

result = mcnemar(table, exact=True)
print("McNemar statistic:", result.statistic)
print("p-value:", result.pvalue)
