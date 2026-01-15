def knn_confidence_score(knn_model, feature_vector, train_labels):
    distances, indices = knn_model.kneighbors(feature_vector, n_neighbors=knn_model.n_neighbors)
    neighbor_labels = train_labels[indices[0]]

   
    votes = Counter(neighbor_labels)
    pred_class = votes.most_common(1)[0][0]

    total = sum(votes.values())
    confidence = (votes[pred_class] / total) * 100

    return pred_class, confidence
