package com.googlemlkit.ganeshaghav;

public enum ButtonNames {

    ImageClassification ("ImageClassification"),
    FlowerIdentification ("FlowerIdentification"),
    ObjectDetection ("ObjectDetection"),
    FaceDetection ("FaceDetection"),
    BirdSoundIdentifier ("BirdSoundIdentifier"),
    SpamTextDetector ("SpamTextDetector"),
    PoseDetection ("PoseDetection"),
    VisitorAnalysis ("VisitorAnalysis"),
    FaceRecognition ("FaceRecognition");

    private final String name;

    private ButtonNames(String s) {
        name = s;
    }

    public boolean equalsName(String otherName) {
        return name.equals(otherName);
    }

    public String toString() {
        return this.name;
    }

}
