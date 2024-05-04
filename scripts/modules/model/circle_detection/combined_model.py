def predict_with_combined_models(image_path, color_model, texture_model, color_scaler, texture_scaler):
    image = cv.imread(image_path)
    if image is None:
        print("Image non trouvée.")
        return

    color_features = extract_color_features(image, color_scaler)
    texture_features = extract_texture_features(image, texture_scaler)

    color_prediction = color_model.predict([color_features])
    texture_prediction = texture_model.predict([texture_features])

    if color_prediction == texture_prediction:
        final_prediction = color_prediction
    else:
        final_prediction = resolve_conflict(color_prediction, texture_prediction)
        # TODO
        
    print("Prédiction finale pour la pièce détectée :", final_prediction)


# TODO
# résoudre conflit, choisir model A ou B en fonction de ? en fonction de la luminosité ? d'un poids attribué au model jugé le plus efficace ?
def resolve_conflict(color_prediction, texture_prediction):
    pass