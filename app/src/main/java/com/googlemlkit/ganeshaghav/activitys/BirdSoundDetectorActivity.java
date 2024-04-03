package com.googlemlkit.ganeshaghav.activitys;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.AudioRecord;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.googlemlkit.ganeshaghav.R;

import org.tensorflow.lite.support.audio.TensorAudio;
import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.task.audio.classifier.AudioClassifier;
import org.tensorflow.lite.task.audio.classifier.Classifications;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

public class BirdSoundDetectorActivity extends AppCompatActivity {

    private Button btnStartRecording,btnStopRecording;
    private TextView outputTextView,specsTextView;

    public final static int REQUEST_RECORD_AUDIO = 2033;


    String modelPath = "my_birds_model.tflite";
    float probabilityThreshold = 0.3f;
    AudioClassifier classifier;
    private TensorAudio tensor;
    private AudioRecord record;
    private TimerTask timerTask;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_bird_sound_detector);
       
        btnStartRecording=(Button) findViewById(R.id.btnStartRecording);
        btnStopRecording=(Button) findViewById(R.id.btnStopRecording);

        outputTextView=(TextView) findViewById(R.id.outputTextView);
        specsTextView=(TextView) findViewById(R.id.specsTextView);


        btnStartRecording.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onStartRecording();
            }
        });

        btnStopRecording.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onStopRecording();
            }
        });


        btnStopRecording.setEnabled(false);

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
        }

    }
    public void onStartRecording() {
        btnStartRecording.setEnabled(false);
        btnStopRecording.setEnabled(true);

        try {
            // Loading the model from the assets folder
            classifier = AudioClassifier.createFromFile(this, modelPath);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Creating an audio recorder
        tensor = classifier.createInputTensorAudio();

        // showing the audio recorder specification
        TensorAudio.TensorAudioFormat format = classifier.getRequiredTensorAudioFormat();
        String specs = "Number of channels: " + format.getChannels() + "\n"
                + "Sample Rate: " + format.getSampleRate();
        specsTextView.setText(specs);

        // Creating and start recording
        record = classifier.createAudioRecord();
        record.startRecording();

        timerTask = new TimerTask() {
            @Override
            public void run() {
                Log.d(BirdSoundDetectorActivity.class.getSimpleName(), "timer task triggered");
                // Classifying audio data
                // val numberOfSamples = tensor.load(record)
                // val output = classifier.classify(tensor)
                int numberOfSamples = tensor.load(record);
                List<Classifications> output = classifier.classify(tensor);

                // Filtering out classifications with low probability
                List<Category> finalOutput = new ArrayList<>();
//                for (Classifications classifications : output) {
                for (Category category : output.get(0).getCategories()) {
                    if (category.getLabel().equals("Bird") && category.getScore() > probabilityThreshold) {
                        finalOutput.add(category);
                    }
                }
//                }

                if (finalOutput.isEmpty()) {
                    return;
                }

                finalOutput = new ArrayList<>();
                for (Category category : output.get(1).getCategories()) {
                    if (category.getScore() > probabilityThreshold) {
                        finalOutput.add(category);
                    }
                }

                // Sorting the results
                Collections.sort(finalOutput, (o1, o2) -> (int) (o1.getScore() - o2.getScore()));

                // Creating a multiline string with the filtered results
                StringBuilder outputStr = new StringBuilder();
                for (Category category : finalOutput) {
                    outputStr.append(category.getLabel())
                            .append(": ").append(category.getScore())
                            .append(", ").append(category.getDisplayName()).append("\n");
                }

                // Updating the UI
                List<Category> finalOutput1 = finalOutput;
                runOnUiThread(() -> {
                    if (finalOutput1.isEmpty()) {
                        outputTextView.setText("Could not identify the bird");
                    } else {
                        outputTextView.setText(outputStr.toString());
                    }
                });
            }
        };

        new Timer().scheduleAtFixedRate(timerTask, 1, 500);
    }

    public void onStopRecording() {
        btnStartRecording.setEnabled(true);
        btnStopRecording.setEnabled(false);

        timerTask.cancel();
        record.stop();
    }


}