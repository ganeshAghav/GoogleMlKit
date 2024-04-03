package com.googlemlkit.ganeshaghav.helpers;

import static com.googlemlkit.ganeshaghav.helpers.PoseEmbedding.getPoseEmbedding;
import static com.googlemlkit.ganeshaghav.helpers.Utils.maxAbs;
import static com.googlemlkit.ganeshaghav.helpers.Utils.multiply;
import static com.googlemlkit.ganeshaghav.helpers.Utils.multiplyAll;
import static com.googlemlkit.ganeshaghav.helpers.Utils.subtract;
import static com.googlemlkit.ganeshaghav.helpers.Utils.sumAbs;
import static java.lang.Math.min;
import android.util.Pair;
import com.google.mlkit.vision.common.PointF3D;
import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseLandmark;
import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import static java.lang.Math.max;
import static java.lang.Math.min;


public class PoseClassifier {

  private static final String TAG = "PoseClassifier";
  private static final int MAX_DISTANCE_TOP_K = 30;
  private static final int MEAN_DISTANCE_TOP_K = 10;
  private static final PointF3D AXES_WEIGHTS = PointF3D.from(1, 1, 0.2f);

  private final List<PoseSample> poseSamples;
  private final int maxDistanceTopK;
  private final int meanDistanceTopK;
  private final PointF3D axesWeights;

  public PoseClassifier(List<PoseSample> poseSamples) {
    this(poseSamples, MAX_DISTANCE_TOP_K, MEAN_DISTANCE_TOP_K, AXES_WEIGHTS);
  }

  public PoseClassifier(List<PoseSample> poseSamples, int maxDistanceTopK,
                        int meanDistanceTopK, PointF3D axesWeights) {
    this.poseSamples = poseSamples;
    this.maxDistanceTopK = maxDistanceTopK;
    this.meanDistanceTopK = meanDistanceTopK;
    this.axesWeights = axesWeights;
  }

  private static List<PointF3D> extractPoseLandmarks(Pose pose) {
    List<PointF3D> landmarks = new ArrayList<>();
    for (PoseLandmark poseLandmark : pose.getAllPoseLandmarks()) {
      landmarks.add(poseLandmark.getPosition3D());
    }
    return landmarks;
  }


  public int confidenceRange() {
    return min(maxDistanceTopK, meanDistanceTopK);
  }

  public ClassificationResult classify(Pose pose) {
    return classify(extractPoseLandmarks(pose));
  }

  public ClassificationResult classify(List<PointF3D> landmarks) {
    ClassificationResult result = new ClassificationResult();

    if (landmarks.isEmpty()) {
      return result;
    }

    // We do flipping on X-axis so we are horizontal (mirror) invariant.
    List<PointF3D> flippedLandmarks = new ArrayList<>(landmarks);
    multiplyAll(flippedLandmarks, PointF3D.from(-1, 1, 1));

    List<PointF3D> embedding = getPoseEmbedding(landmarks);
    List<PointF3D> flippedEmbedding = getPoseEmbedding(flippedLandmarks);


    PriorityQueue<Pair<PoseSample, Float>> maxDistances = new PriorityQueue<>(
        maxDistanceTopK, (o1, o2) -> -Float.compare(o1.second, o2.second));
    for (PoseSample poseSample : poseSamples) {
      List<PointF3D> sampleEmbedding = poseSample.getEmbedding();

      float originalMax = 0;
      float flippedMax = 0;
      for (int i = 0; i < embedding.size(); i++) {
        originalMax =
            max(
                originalMax,
                maxAbs(multiply(subtract(embedding.get(i), sampleEmbedding.get(i)), axesWeights)));
        flippedMax =
            max(
                flippedMax,
                maxAbs(
                    multiply(
                        subtract(flippedEmbedding.get(i), sampleEmbedding.get(i)), axesWeights)));
      }
      // Set the max distance as min of original and flipped max distance.
      maxDistances.add(new Pair<>(poseSample, min(originalMax, flippedMax)));
      // We only want to retain top n so pop the highest distance.
      if (maxDistances.size() > maxDistanceTopK) {
        maxDistances.poll();
      }
    }

    // Keeps higher mean distances on top so we can pop it when top_k size is reached.
    PriorityQueue<Pair<PoseSample, Float>> meanDistances = new PriorityQueue<>(
        meanDistanceTopK, (o1, o2) -> -Float.compare(o1.second, o2.second));
    // Retrive top K poseSamples by least mean distance to remove outliers.
    for (Pair<PoseSample, Float> sampleDistances : maxDistances) {
      PoseSample poseSample = sampleDistances.first;
      List<PointF3D> sampleEmbedding = poseSample.getEmbedding();

      float originalSum = 0;
      float flippedSum = 0;
      for (int i = 0; i < embedding.size(); i++) {
        originalSum += sumAbs(multiply(
            subtract(embedding.get(i), sampleEmbedding.get(i)), axesWeights));
        flippedSum += sumAbs(
            multiply(subtract(flippedEmbedding.get(i), sampleEmbedding.get(i)), axesWeights));
      }
      // Set the mean distance as min of original and flipped mean distances.
      float meanDistance = min(originalSum, flippedSum) / (embedding.size() * 2);
      meanDistances.add(new Pair<>(poseSample, meanDistance));
      // We only want to retain top k so pop the highest mean distance.
      if (meanDistances.size() > meanDistanceTopK) {
        meanDistances.poll();
      }
    }

    for (Pair<PoseSample, Float> sampleDistances : meanDistances) {
      String className = sampleDistances.first.getClassName();
      result.incrementClassConfidence(className);
    }

    return result;
  }
}
