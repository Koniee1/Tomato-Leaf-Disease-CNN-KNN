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


# Feature Extraction
feature_extractor = Model(inputs=full_model.input, outputs=full_model.get_layer("global_average_pooling2d").output)
X_train = feature_extractor.predict(train_gen, verbose=1)
y_train = train_gen.classes
X_val = feature_extractor.predict(val_gen, verbose=1)
y_val = val_gen.classes

# To flatten 
if X_train.ndim > 2:
     X_train = X_train.reshape((X_train.shape[0], -1))
     X_val = X_val.reshape((X_val.shape[0], -1))

