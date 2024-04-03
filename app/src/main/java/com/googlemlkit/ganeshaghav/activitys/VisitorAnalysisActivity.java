package com.googlemlkit.ganeshaghav.activitys;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.annotation.OptIn;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.lifecycle.Observer;

import com.google.android.material.floatingactionbutton.ExtendedFloatingActionButton;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.face.Face;
import com.googlemlkit.ganeshaghav.R;
import com.googlemlkit.ganeshaghav.helpers.AgeGenderEstimationProcessor;
import com.googlemlkit.ganeshaghav.helpers.GraphicOverlay;
import com.googlemlkit.ganeshaghav.helpers.VisionBaseProcessor;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class VisitorAnalysisActivity extends AppCompatActivity implements AgeGenderEstimationProcessor.AgeGenderCallback{

    private static final int REQUEST_CAMERA = 1001;
    protected PreviewView previewView;
    protected GraphicOverlay graphicOverlay;
    private TextView outputTextView;
    private Button addFaceButton;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private Executor executor = Executors.newSingleThreadExecutor();

    private VisionBaseProcessor processor;
    private ImageAnalysis imageAnalysis;



    private Interpreter ageModelInterpreter;
    private Interpreter genderModelInterpreter;
    private int facesCount;
    private int smilingCount;
    private int maleCount;
    private int femaleCount;
    private int kidsCount;
    private int youngCount;
    private int adultCount;
    private int agedCount;
    private Set<Integer> faceTrackingIdSet = new HashSet<>();



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_visitor_analysis);

        previewView = findViewById(R.id.camera_source_preview);
        graphicOverlay = findViewById(R.id.graphic_overlay);
        outputTextView = findViewById(R.id.output_text_view);
        addFaceButton = findViewById(R.id.button_add_face);

        cameraProviderFuture = ProcessCameraProvider.getInstance(getApplicationContext());

        processor = setProcessor();

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
        } else {
            initSource();
        }

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (processor != null) {
            processor.stop();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == REQUEST_CAMERA && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            initSource();
        }
    }

    protected void setOutputText(String text) {
        outputTextView.setText(text);
    }

    private void initSource() {

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException e) {
                // No errors need to be handled for this Future.
                // This should never be reached.
            }
        }, ContextCompat.getMainExecutor(getApplicationContext()));
    }

    void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        int lensFacing = getLensFacing();
        Preview preview = new Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(lensFacing)
                .build();

        imageAnalysis =
                new ImageAnalysis.Builder()
                        .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                        .build();

        setFaceDetector(lensFacing);
        cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalysis, preview);
    }


    private void setFaceDetector(int lensFacing) {
        previewView.getPreviewStreamState().observe(this, new Observer<PreviewView.StreamState>() {
            @Override
            public void onChanged(PreviewView.StreamState streamState) {
                if (streamState != PreviewView.StreamState.STREAMING) {
                    return;
                }

                View preview = previewView.getChildAt(0);
                float width = preview.getWidth() * preview.getScaleX();
                float height = preview.getHeight() * preview.getScaleY();
                float rotation = preview.getDisplay().getRotation();
                if (rotation == Surface.ROTATION_90 || rotation == Surface.ROTATION_270) {
                    float temp = width;
                    width = height;
                    height = temp;
                }

                imageAnalysis.setAnalyzer(
                        executor,
                        createFaceDetector((int) width, (int) height, lensFacing)
                );
                previewView.getPreviewStreamState().removeObserver(this);
            }
        });
    }

    @OptIn(markerClass = ExperimentalGetImage.class)
    private ImageAnalysis.Analyzer createFaceDetector(int width, int height, int lensFacing) {
        graphicOverlay.setPreviewProperties(width, height, lensFacing);
        return imageProxy -> {
            if (imageProxy.getImage() == null) {
                imageProxy.close();
                return;
            }
            int rotationDegrees = imageProxy.getImageInfo().getRotationDegrees();
            // converting from YUV format
            processor.detectInImage(imageProxy, toBitmap(imageProxy.getImage()), rotationDegrees);
            // after done, release the ImageProxy object
            imageProxy.close();
        };
    }

    private Bitmap toBitmap(Image image) {
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];
        //U and V are swapped
        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 75, out);

        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    protected int getLensFacing() {
        return CameraSelector.LENS_FACING_BACK;
    }

    public void makeAddFaceVisible() {
        addFaceButton.setVisibility(View.VISIBLE);
    }


    protected VisionBaseProcessor setProcessor() {
        try {
            ageModelInterpreter = new Interpreter(FileUtil.loadMappedFile(this, "model_lite_age_q.tflite"), new Interpreter.Options());
            genderModelInterpreter = new Interpreter(FileUtil.loadMappedFile(this, "model_lite_gender_q.tflite"), new Interpreter.Options());
        } catch (IOException e) {
            e.printStackTrace();
        }

        AgeGenderEstimationProcessor ageGenderEstimationProcessor = new AgeGenderEstimationProcessor(
                ageModelInterpreter,
                genderModelInterpreter,
                graphicOverlay,
                this
        );
        ageGenderEstimationProcessor.activity = this;
        return ageGenderEstimationProcessor;
    }

    public void setTestImage(Bitmap cropToBBox) {
        if (cropToBBox == null) {
            return;
        }
        runOnUiThread(() -> ((ImageView) findViewById(R.id.testImageView)).setImageBitmap(cropToBBox));
    }

    @Override
    public void onFaceDetected(Face face, int age, int gender) {
        if (!faceTrackingIdSet.contains(face.getTrackingId())) {
            facesCount++;

            if (face.getSmilingProbability() != null && face.getSmilingProbability() > .79f) {
                smilingCount++;
            }

            if (age < 12) {
                kidsCount++;
            } else if (age < 20) {
                youngCount++;
            } else if (age < 60) {
                adultCount++;
            } else {
                agedCount++;
            }

            if (gender == 0) {
                maleCount++;
            } else {
                femaleCount++;
            }

            StringBuilder builder = new StringBuilder();
            builder.append("Total faces: ").append(facesCount).append(", Smiling: ").append((int) ((smilingCount/(float) facesCount) * 100.0f)).append("%\n")
                    .append("Male: ").append(maleCount).append(", Female: ").append(femaleCount).append("\n")
                    .append("Kids: ").append(kidsCount).append(", Young: ").append(youngCount).append("\n")
                    .append("Adults: ").append(adultCount).append(", Aged: ").append(agedCount);

            setOutputText(builder.toString());
        }
    }
}