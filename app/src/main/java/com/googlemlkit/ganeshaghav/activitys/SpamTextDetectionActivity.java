package com.googlemlkit.ganeshaghav.activitys;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.support.label.Category;
import org.tensorflow.lite.task.text.nlclassifier.NLClassifier;
import com.googlemlkit.ganeshaghav.R;

import java.io.IOException;
import java.util.List;

public class SpamTextDetectionActivity extends AppCompatActivity {

    private EditText edtSpamText;
    private Button btnSpamCheck;
    private TextView txtResult;


    private static final String MODEL_PATH = "model_spam.tflite";
    private NLClassifier classifier;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_spam_text_detection);

        edtSpamText=(EditText) findViewById(R.id.edtSpamText);
        btnSpamCheck=(Button) findViewById(R.id.btnSpamCheck);
        txtResult=(TextView) findViewById(R.id.txtSpamResult);

        btnSpamCheck.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                if(!edtSpamText.getText().toString().equals("")){

                    List<Category> apiResults = classifier.classify(edtSpamText.getText().toString());
                    float score = apiResults.get(1).getScore();
                    if (score > 0.8f) {
                        txtResult.setText("Detected as spam.\nSpam score: " + score);
                    } else {
                        txtResult.setText("Not detected as spam.\nSpam score: " + score);
                    }
                }
                else {
                    Toast.makeText(getApplicationContext(),"Please enter txt",Toast.LENGTH_SHORT).show();
                }

            }
        });

        getTitle();
        try {
            classifier = NLClassifier.createFromFile(this, MODEL_PATH);
        } catch (IOException e) {
            Log.e(SpamTextDetectionActivity.class.getSimpleName(), e.getMessage());
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        classifier.close();
        classifier = null;
    }

}