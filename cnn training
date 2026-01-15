

input_layer = Input(shape=(224,224,3))
base = EfficientNetV2B1(include_top=False, weights='imagenet', input_tensor=input_layer)


for layer in base.layers:
    layer.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(len(labels), activation='softmax')(x)

full_model = Model(inputs=input_layer, outputs=output_layer)
for layer in base.layers[:-20]:
    layer.trainable = False
for layer in base.layers[-20:]:
    layer.trainable = True

full_model.compile(
    optimizer=Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
full_model.summary()


y_train = train.classes
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = {i:w for i,w in enumerate(class_weights)}

checkpoint = ModelCheckpoint('cnn_best_model.h5', monitor='val_accuracy', save_best_only=True)
early = EarlyStopping(patience=7, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.2, min_lr=1e-6)


history = full_model.fit(
    train,
    validation_data=val,
    epochs=50,
    callbacks=[early, lr_reduce, checkpoint],
    class_weight=cw_dict
)


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
