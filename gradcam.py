def generate_gradcam_p(model, img_tensor, layer_name, class_index, debug=False):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape(persistent=True) as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        second_derivative = tape.gradient(grads, conv_outputs)
        third_derivative = tape.gradient(second_derivative, conv_outputs)

    del tape  # 

    conv_outputs = conv_outputs[0]
    grads = grads[0]
    second_derivative = second_derivative[0]
    third_derivative = third_derivative[0]

    numerator = second_derivative
    denominator = 2.0 * second_derivative + third_derivative * conv_outputs
    denominator = tf.where(denominator != 0.0, denominator, tf.ones_like(denominator))
    alphas = numerator / denominator

    weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(0, 1))
    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)
    cam = tf.nn.relu(cam)

    cam = cam.numpy()
    cam = cv2.resize(cam, (img_tensor.shape[2], img_tensor.shape[1]))
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    if debug:
        print(f"[Grad-CAM++] Class: {class_index}, Score: {loss.numpy()[0]:.4f}")
        plt.imshow(cam, cmap="jet")
        plt.axis("off")
        plt.show()

    return heatmap, loss.numpy()[0]
def gradcam_overlay(img_array, heatmap, true_label_index, pred_class, confidence_score, alpha=0.25):
    if isinstance(img_array, tf.Tensor):
        img_array = img_array.numpy()

    img = img_array.copy()
    if np.max(img) <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if len(img.shape) == 3:
        if img.shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, leaf_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    leaf_mask = cv2.GaussianBlur(leaf_mask, (7, 7), 0)
    leaf_mask = leaf_mask / 255.0

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0])).astype(np.uint8)
    heatmap = (heatmap * leaf_mask[..., np.newaxis]).astype(np.uint8)

    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image", fontsize=13)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title(
        f"True: {labels[int(true_label_index)]}\n"
        f"Pred: {labels[int(pred_class)]}\n"
        f"Conf: {confidence_score:.2f}%", fontsize=12
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()
seen_labels = set()
count = 0
last_conv_layer_name = "top_conv"

for batch_imgs, batch_labels in val:
    for j in range(len(batch_imgs)):
        img = batch_imgs[j]
        label = batch_labels[j]

        if int(label) not in seen_labels:
            img_exp = tf.expand_dims(img, axis=0)

            cnn_preds = full_model.predict(img_exp, verbose=0)
            cnn_class = np.argmax(cnn_preds, axis=1)[0]

            features = feature_extractor.predict(img_exp, verbose=0)
            features_scaled = scaler.transform(features)
            features_pca = pca.transform(features_scaled)

            pred_class, confidence = knn_confidence_score(
                knn_model=grid_search.best_estimator_,
                feature_vector=features_pca,
                train_labels=y_res
            )

            heatmap, _ = generate_gradcam_p(
                full_model, img_exp, last_conv_layer_name, cnn_class
            )

            display_img = img * 255 if np.max(img) <= 1.0 else img
            gradcam_overlay(display_img, heatmap, label, pred_class, confidence)

            seen_labels.add(int(label))
            count += 1

        if count >= 4:
            break
    if count >= 4:
        break
