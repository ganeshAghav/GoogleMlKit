package com.googlemlkit.ganeshaghav;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.icu.text.SimpleDateFormat;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.exifinterface.media.ExifInterface;
import com.google.mlkit.common.model.LocalModel;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.label.ImageLabel;
import com.google.mlkit.vision.label.ImageLabeler;
import com.google.mlkit.vision.label.ImageLabeling;
import com.google.mlkit.vision.label.custom.CustomImageLabelerOptions;
import com.google.mlkit.vision.label.defaults.ImageLabelerOptions;
import com.google.mlkit.vision.objects.DetectedObject;
import com.google.mlkit.vision.objects.ObjectDetection;
import com.google.mlkit.vision.objects.ObjectDetector;
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions;
import com.googlemlkit.ganeshaghav.activitys.BirdSoundDetectorActivity;
import com.googlemlkit.ganeshaghav.activitys.FaceRecognitionActivity;
import com.googlemlkit.ganeshaghav.activitys.PoseDetectionActivity;
import com.googlemlkit.ganeshaghav.activitys.SpamTextDetectionActivity;
import com.googlemlkit.ganeshaghav.activitys.VisitorAnalysisActivity;
import com.googlemlkit.ganeshaghav.helpers.BoxWithText;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;


public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    public final String LOG_TAG = "MainActivity";
    private Button btnImageClassification,btnFlowerIdentification,btnObjectDetection,btnFaceDetection,
    btnBirdSoundIdentifier,btnSpamTextDetector,btnPoseDetection,btnVisitorAnalysis,btnFaceRecognition;

    //Permissions
    private int REQUEST_ID_MULTIPLE_PERMISSIONS = 23;

    private File photoFile;
    private final static int CAPTURE_IMAGE_ACTIVITY_REQUEST_CODE = 1034;
    private final static int PICK_IMAGE_ACTIVITY_REQUEST_CODE = 1064;

    private String selectedButton="";

    private ImageLabeler imageLabeler;
    private ObjectDetector objectDetector;
    private FaceDetector faceDetector;

    //Scan QR Code
    private AlertDialog.Builder alertDialogBuilder;
    private AlertDialog alertDialog;
    private ImageView imgResult;
    private TextView txtResult;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnImageClassification =(Button) findViewById(R.id.btnImageClassification);
        btnImageClassification.setOnClickListener(this);

        btnFlowerIdentification =(Button) findViewById(R.id.btnFlowerIdentification);
        btnFlowerIdentification.setOnClickListener(this);

        btnObjectDetection =(Button) findViewById(R.id.btnObjectDetection);
        btnObjectDetection.setOnClickListener(this);

        btnFaceDetection =(Button) findViewById(R.id.btnFaceDetection);
        btnFaceDetection.setOnClickListener(this);

        btnBirdSoundIdentifier =(Button) findViewById(R.id.btnBirdSoundIdentifier);
        btnBirdSoundIdentifier.setOnClickListener(this);

        btnSpamTextDetector =(Button) findViewById(R.id.btnSpamTextDetector);
        btnSpamTextDetector.setOnClickListener(this);

        btnPoseDetection =(Button) findViewById(R.id.btnPoseDetection);
        btnPoseDetection.setOnClickListener(this);

        btnVisitorAnalysis =(Button) findViewById(R.id.btnVisitorAnalysis);
        btnVisitorAnalysis.setOnClickListener(this);

        btnFaceRecognition =(Button) findViewById(R.id.btnFaceRecognition);
        btnFaceRecognition.setOnClickListener(this);


        GetPermissionDetails();

    }

    @Override
    public void onClick(View v) {

        GetPermissionDetails();

        int id =v.getId();

        if(id ==  R.id.btnImageClassification){

            imageLabeler=null;
            imageLabeler = ImageLabeling.getClient(new ImageLabelerOptions.Builder()
                    .setConfidenceThreshold(0.7f)
                    .build());

            selectedButton = ButtonNames.ImageClassification.name();
            CaptureImage();
        }
        else if(id == R.id.btnFlowerIdentification){

            imageLabeler=null;
            LocalModel localModel = new LocalModel.Builder().setAssetFilePath("model_flowers.tflite").build();
            CustomImageLabelerOptions options = new CustomImageLabelerOptions.Builder(localModel)
                    .setConfidenceThreshold(0.7f)
                    .setMaxResultCount(5)
                    .build();
            imageLabeler = ImageLabeling.getClient(options);

            selectedButton = ButtonNames.FlowerIdentification.name();
            CaptureImage();
        }
        else if(id == R.id.btnObjectDetection){

            ObjectDetectorOptions options =
                    new ObjectDetectorOptions.Builder()
                            .setDetectorMode(ObjectDetectorOptions.SINGLE_IMAGE_MODE)
                            .enableMultipleObjects()
                            .enableClassification()
                            .build();

            objectDetector = ObjectDetection.getClient(options);

            selectedButton = ButtonNames.ObjectDetection.name();
            CaptureImage();
        }
        else if(id == R.id.btnFaceDetection){
            FaceDetectorOptions highAccuracyOpts =
                    new FaceDetectorOptions.Builder()
                            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                            .enableTracking()
                            .build();

            faceDetector = FaceDetection.getClient(highAccuracyOpts);

            selectedButton = ButtonNames.FaceDetection.name();
            CaptureImage();
        }
        else if(id == R.id.btnBirdSoundIdentifier){
            selectedButton = ButtonNames.BirdSoundIdentifier.name();

            Intent intent=new Intent(MainActivity.this, BirdSoundDetectorActivity.class);
            startActivity(intent);
        }
        else if(id == R.id.btnSpamTextDetector){
            selectedButton = ButtonNames.SpamTextDetector.name();

            Intent intent=new Intent(MainActivity.this, SpamTextDetectionActivity.class);
            startActivity(intent);
        }
        else if(id == R.id.btnPoseDetection){
            selectedButton = ButtonNames.PoseDetection.name();


            Intent intent=new Intent(MainActivity.this, PoseDetectionActivity.class);
            startActivity(intent);
        }
        else if(id == R.id.btnVisitorAnalysis){
            selectedButton = ButtonNames.VisitorAnalysis.name();

            Intent intent=new Intent(MainActivity.this, VisitorAnalysisActivity.class);
            startActivity(intent);
        }
        else if(id == R.id.btnFaceRecognition){
            selectedButton = ButtonNames.FaceRecognition.name();

            Intent intent=new Intent(MainActivity.this, FaceRecognitionActivity.class);
            startActivity(intent);
        }
    }

    //////////////////////////////////// RUN TIME PERMISSION ///////////////////////////////////////
    public void  GetPermissionDetails() {

        boolean result=CheckPermissionsGranted();
        if(result==false)
        {
            requestPermission();
        }
        else {
            proceedAfterPermission();
        }
    }

    public boolean CheckPermissionsGranted() {

        int cameraPermission = ContextCompat.checkSelfPermission(this,android.Manifest.permission.CAMERA);
        int writePermission = ContextCompat.checkSelfPermission(this,android.Manifest.permission.WRITE_EXTERNAL_STORAGE);
        int readPermission = ContextCompat.checkSelfPermission(this,android.Manifest.permission.READ_EXTERNAL_STORAGE);
        int audioRecordPermission = ContextCompat.checkSelfPermission(this,android.Manifest.permission.RECORD_AUDIO);

        if(cameraPermission==0 && writePermission==0 && readPermission==0 && audioRecordPermission==0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    private void requestPermission() {

        try
        {
            int cameraPermission = ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA);
            int writePermission = ContextCompat.checkSelfPermission(this, android.Manifest.permission.WRITE_EXTERNAL_STORAGE);
            int readPermission = ContextCompat.checkSelfPermission(this, android.Manifest.permission.READ_EXTERNAL_STORAGE);
            int audioRecordPermission = ContextCompat.checkSelfPermission(this,android.Manifest.permission.RECORD_AUDIO);

            List<String> listPermissionsNeeded = new ArrayList<>();


            if (cameraPermission != PackageManager.PERMISSION_GRANTED)
            {
                listPermissionsNeeded.add(android.Manifest.permission.CAMERA);
            }
            if (writePermission != PackageManager.PERMISSION_GRANTED)
            {
                listPermissionsNeeded.add(android.Manifest.permission.WRITE_EXTERNAL_STORAGE);
            }
            if (readPermission != PackageManager.PERMISSION_GRANTED)
            {
                listPermissionsNeeded.add(android.Manifest.permission.READ_EXTERNAL_STORAGE);
            }
            if (audioRecordPermission != PackageManager.PERMISSION_GRANTED)
            {
                listPermissionsNeeded.add(android.Manifest.permission.RECORD_AUDIO);
            }


            if (!listPermissionsNeeded.isEmpty())
            {
                ActivityCompat.requestPermissions(this, listPermissionsNeeded.toArray(new String[listPermissionsNeeded.size()]), REQUEST_ID_MULTIPLE_PERMISSIONS);
            }
            else{
                proceedAfterPermission();
            }

        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if(requestCode == REQUEST_ID_MULTIPLE_PERMISSIONS)
        {
            if(grantResults.length >0 && grantResults[0] == PackageManager.PERMISSION_GRANTED)
            {
                proceedAfterPermission();
            }
            else
            {
                Toast.makeText(getApplicationContext(),"oops permission denied",Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void proceedAfterPermission() {
        try{

        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

    ////////////////////////////////////// GET IMAGE FROM CAMERA ///////////////////////////////////
    public void CaptureImage(){
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        photoFile = getPhotoFileUri(new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date()) + ".jpg");

        Uri fileProvider = FileProvider.getUriForFile(MainActivity.this, "com.iago.fileprovider1", photoFile);
        intent.putExtra(MediaStore.EXTRA_OUTPUT, fileProvider);

        if (intent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(intent, CAPTURE_IMAGE_ACTIVITY_REQUEST_CODE);
        }
    }

    public File getPhotoFileUri(String fileName) {
        File mediaStorageDir = new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), LOG_TAG);

        if (!mediaStorageDir.exists() && !mediaStorageDir.mkdirs()){
            Log.d(LOG_TAG, "failed to create directory");
        }

        File file = new File(mediaStorageDir.getPath() + File.separator + fileName);

        return file;
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == CAPTURE_IMAGE_ACTIVITY_REQUEST_CODE) {
            if (resultCode == RESULT_OK) {
                Bitmap bitmap = getCapturedImage();
                if(bitmap!=null){
                    rotateIfRequired(bitmap);

                    bitmapCaptured(bitmap);
                }
                else {
                    Toast.makeText(this, "Bitmap is null", Toast.LENGTH_SHORT).show();
                }

            } else { // Result was a failure
                Toast.makeText(this, "Picture wasn't taken!", Toast.LENGTH_SHORT).show();
            }
        } else if (requestCode == PICK_IMAGE_ACTIVITY_REQUEST_CODE) {
            if (resultCode == RESULT_OK) {
                Bitmap bitmap = loadFromUri(data.getData());

                if(bitmap!=null){
                    bitmapCaptured(bitmap);
                }
                else {
                    Toast.makeText(this, "Bitmap is null", Toast.LENGTH_SHORT).show();
                }
            } else { // Result was a failure
                Toast.makeText(this, "Picture wasn't selected!", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private Bitmap getCapturedImage() {

        int targetW = 300;
        int targetH = 300;

        BitmapFactory.Options bmOptions = new BitmapFactory.Options();
        bmOptions.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(photoFile.getAbsolutePath());
        int photoW = bmOptions.outWidth;
        int photoH = bmOptions.outHeight;
        int scaleFactor = Math.max(1, Math.min(photoW / targetW, photoH /targetH));

        bmOptions = new BitmapFactory.Options();
        bmOptions.inJustDecodeBounds = false;
        bmOptions.inSampleSize = scaleFactor;
        bmOptions.inMutable = true;
        return BitmapFactory.decodeFile(photoFile.getAbsolutePath(), bmOptions);
    }

    private void rotateIfRequired(Bitmap bitmap) {
        try {
            ExifInterface exifInterface = new ExifInterface(photoFile.getAbsolutePath());
            int orientation = exifInterface.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_UNDEFINED
            );

            if (orientation == ExifInterface.ORIENTATION_ROTATE_90) {
                rotateImage(bitmap, 90f);
            } else if (orientation == ExifInterface.ORIENTATION_ROTATE_180) {
                rotateImage(bitmap, 180f);
            } else if (orientation == ExifInterface.ORIENTATION_ROTATE_270) {
                rotateImage(bitmap, 270f);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private Bitmap rotateImage(Bitmap source, float angle) {
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(
                source, 0, 0, source.getWidth(), source.getHeight(),
                matrix, true
        );
    }

    protected Bitmap loadFromUri(Uri photoUri) {
        Bitmap image = null;

        try {
            if (Build.VERSION.SDK_INT > 27) {
                ImageDecoder.Source source = ImageDecoder.createSource(this.getContentResolver(), photoUri);
                image = ImageDecoder.decodeBitmap(source);
            } else {
                image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), photoUri);
            }
        } catch(IOException e) {
            e.printStackTrace();
        }

        return image;
    }


    ///////////////////////////////////// BITMAP CAPTURED //////////////////////////////////////////
    private void bitmapCaptured(Bitmap bitmap){

        if(selectedButton.equals(ButtonNames.ImageClassification.name())){

            InputImage inputImage = InputImage.fromBitmap(bitmap, 0);
            imageLabeler.process(inputImage).addOnSuccessListener(imageLabels -> {
                StringBuilder sb = new StringBuilder();
                for (ImageLabel label : imageLabels) {
                    sb.append(label.getText()).append(": ").append(label.getConfidence()).append("\n");
                }
                if (imageLabels.isEmpty()) {
                    DialogShow(bitmap,"Could not classify!!");
                } else {
                    DialogShow(bitmap,sb.toString());
                }
            }).addOnFailureListener(e -> {
                e.printStackTrace();
            });

        }
        else if(selectedButton.equals(ButtonNames.FlowerIdentification.name())){

            InputImage inputImage = InputImage.fromBitmap(bitmap, 0);
            imageLabeler.process(inputImage).addOnSuccessListener(imageLabels -> {
                StringBuilder sb = new StringBuilder();
                for (ImageLabel label : imageLabels) {
                    sb.append(label.getText()).append(": ").append(label.getConfidence()).append("\n");
                }
                if (imageLabels.isEmpty()) {
                    DialogShow(bitmap,"Could not classify!!");
                } else {
                    DialogShow(bitmap,sb.toString());
                }
            }).addOnFailureListener(e -> {
                e.printStackTrace();
            });

        }
        else if(selectedButton.equals(ButtonNames.ObjectDetection.name())){

            InputImage image = InputImage.fromBitmap(bitmap, 0);
            objectDetector.process(image)
                    .addOnSuccessListener(
                            detectedObjects -> {
                                // Task completed successfully
                                StringBuilder sb = new StringBuilder();
                                List<BoxWithText> list = new ArrayList<>();
                                for (DetectedObject object : detectedObjects) {
                                    for (DetectedObject.Label label : object.getLabels()) {
                                        sb.append(label.getText()).append(" : ")
                                                .append(label.getConfidence()).append("\n");
                                    }
                                    if (!object.getLabels().isEmpty()) {
                                        list.add(new BoxWithText(object.getLabels().get(0).getText(), object.getBoundingBox()));
                                    } else {
                                        list.add(new BoxWithText("Unknown", object.getBoundingBox()));
                                    }
                                };
                                if (detectedObjects.isEmpty()) {
                                    DialogShow(drawDetectionResult(bitmap, list),"Could not detect!!");
                                } else {
                                    DialogShow(drawDetectionResult(bitmap, list),sb.toString());
                                }
                            })
                    .addOnFailureListener(
                            e -> {
                                // Task failed with an exception
                                // ...
                                e.printStackTrace();
                            });
        }
        else if(selectedButton.equals(ButtonNames.FaceDetection.name())){

            Bitmap finalBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            InputImage image = InputImage.fromBitmap(finalBitmap, 0);

            faceDetector.process(image)
                    .addOnFailureListener(error -> {
                        error.printStackTrace();
                        DialogShow(finalBitmap, error.toString());
                    })
                    .addOnSuccessListener(faces -> {
                        if (faces.isEmpty()) {
                            DialogShow(finalBitmap,"No faces detected");
                        } else {
                            List<BoxWithText> boxes = new ArrayList();
                            for (Face face : faces) {
                                boxes.add(new BoxWithText(face.getTrackingId() + "", face.getBoundingBox()));
                            }
                            DialogShow(drawDetectionResult(finalBitmap, boxes),String.format("%d faces detected", faces.size()));
                        }
                    });

        }
        else if(selectedButton.equals(ButtonNames.BirdSoundIdentifier.name())){

        }
        else if(selectedButton.equals(ButtonNames.SpamTextDetector.name())){

        }
        else if(selectedButton.equals(ButtonNames.PoseDetection.name())){

        }
        else if(selectedButton.equals(ButtonNames.VisitorAnalysis.name())){

        }
        else if(selectedButton.equals(ButtonNames.FaceRecognition.name())){

        }
    }

    protected Bitmap drawDetectionResult(Bitmap bitmap, List<BoxWithText> detectionResults) {
        Bitmap outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(outputBitmap);
        Paint pen = new Paint();
        pen.setTextAlign(Paint.Align.LEFT);

        for (BoxWithText box : detectionResults) {
            // draw bounding box
            pen.setColor(Color.RED);
            pen.setStrokeWidth(8F);
            pen.setStyle(Paint.Style.STROKE);
            canvas.drawRect(box.rect, pen);

            Rect tagSize = new Rect(0, 0, 0, 0);

            // calculate the right font size
            pen.setStyle(Paint.Style.FILL_AND_STROKE);
            pen.setColor(Color.YELLOW);
            pen.setStrokeWidth(2F);

            pen.setTextSize(96F);
            pen.getTextBounds(box.text, 0, box.text.length(), tagSize);
            float fontSize = pen.getTextSize() * box.rect.width() / tagSize.width();

            // adjust the font size so texts are inside the bounding box
            if (fontSize < pen.getTextSize()) {
                pen.setTextSize(fontSize);
            }

            float margin = (box.rect.width() - tagSize.width()) / 2.0F;
            if (margin < 0F) margin = 0F;
            canvas.drawText(
                    box.text, box.rect.left + margin,
                    box.rect.top + tagSize.height(), pen
            );
        }
        return outputBitmap;
    }

    /////////////////////////////////////// ALERT DIALOG ///////////////////////////////////////////

    private void DialogShow(Bitmap bitmap,String result) {

        try {

            alertDialogBuilder = new AlertDialog.Builder(MainActivity.this,R.style.CustomAlertDialog);
            ViewGroup viewGroup = findViewById(android.R.id.content);
            View dialogView = LayoutInflater.from(this).inflate(R.layout.alert_dialog, viewGroup, false);

            imgResult=(ImageView) dialogView.findViewById(R.id.imgResultDialog);
            txtResult=(TextView) dialogView.findViewById(R.id.txtResultDialog);

            imgResult.setImageBitmap(bitmap);
            txtResult.setText(result);


            alertDialogBuilder.setView(dialogView);
            alertDialog = alertDialogBuilder.create();
            alertDialog.setCancelable(true);
            alertDialog.show();

        }
        catch (Exception e){
            Log.i(LOG_TAG,e.toString());
            if(alertDialog.isShowing()){
                alertDialog.dismiss();
            }
        }
    }

}