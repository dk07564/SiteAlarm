package com.e.sitealarm;

import androidx.appcompat.app.AppCompatActivity;

import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;

import org.json.JSONObject;
import org.w3c.dom.Text;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.URL;

public class News extends AppCompatActivity {
    String contents;
    TextView textView;
    Button button;
//    ProgressBar progressBar;
//    RelativeLayout loadingLayout;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_news);
        textView = findViewById(R.id.textView);
        button = findViewById(R.id.button);
//        progressBar = findViewById(R.id.progressBar);
//        loadingLayout = findViewById(R.id.loadingLayout);
//        loadingLayout.setVisibility(View.GONE);

//        progressBar.setVisibility(View.GONE);

        Intent intent = getIntent();
        contents = intent.getStringExtra("contents");
        Log.e("CONTENTS", contents);

        textView.setText(contents);

        Handler handler = new Handler(Looper.getMainLooper());
        handler.post(new Runnable() {
            @Override
            public void run() {
                //요약 버튼
                button.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
//                        progressBar.setVisibility(View.VISIBLE);
//                        loadingLayout.setVisibility(View.VISIBLE);
                        ProgressDialog progressDialog = new ProgressDialog(News.this);
                        progressDialog.setMessage("요약 중");
                        progressDialog.show();

                        News.AsyncT asyncT = new News.AsyncT();
                        asyncT.execute();
                    }
                });
            }
        });
    }
    //JSON 송수신
    class AsyncT extends AsyncTask<Void, Void, Void> {
        @Override
        protected Void doInBackground(Void... params) {
            try {
                //host: 컴퓨터 IP값
                String host = "192.168.0.9";
                URL url = new URL("http://" + host + ":5000");

                //JSON 송신
                HttpURLConnection httpURLConnection = (HttpURLConnection) url.openConnection();
                httpURLConnection.setDoOutput(true);
                httpURLConnection.setRequestMethod("POST");
                httpURLConnection.setRequestProperty("Content-Type", "application/json");
                httpURLConnection.connect();

                JSONObject jsonObject = new JSONObject();

                //크롤링으로 얻은 값을 JSON에 넣음(KEY: Contents, VALUE: Crawilng.getContents())
                jsonObject.put("contents", contents);
                Log.e("JSON", jsonObject.toString());

                DataOutputStream dataOutputStream = new DataOutputStream(httpURLConnection.getOutputStream());
                DataOutputStream out = new DataOutputStream(dataOutputStream);
                OutputStreamWriter osw = new OutputStreamWriter(out);
                osw.write(String.valueOf(jsonObject));
                osw.close();
                out.close();
                dataOutputStream.close();

                //JSON 수신
                BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(httpURLConnection.getInputStream(), "UTF-8"));
                StringBuilder stringBuilder = new StringBuilder();

                String line = null;

                try {
                    while ((line = bufferedReader.readLine()) != null) {
                        stringBuilder.append(line);
                    }

                    //result: 받은 JSON --> String
                    String result = null;
                    result = stringBuilder.toString();
                    Handler handler = new Handler(Looper.getMainLooper());
                    final String finalResult = result;
                    handler.post(new Runnable() {
                        @Override
                        public void run() {
                            if(finalResult != null){

                                TextView summaryText = new TextView(getApplicationContext());
                                summaryText.setText(finalResult);
                                summaryText.setTextSize(15);

                                AlertDialog.Builder builder = new AlertDialog.Builder(News.this);
                                builder.setTitle("요약")
//                                        .setMessage(finalResult)
                                        .setView(summaryText)
                                        .setPositiveButton("확인",
                                                new DialogInterface.OnClickListener() {
                                                    @Override
                                                    public void onClick(DialogInterface dialogInterface, int i) {
                                                        Intent intent = getIntent();
                                                        finish();
                                                        startActivity(intent);
                                                    }
                                                })
                                        .create()
                                        .show();
                            }
                        }
                    });

                    //UI 제어
                    //받은 JSON값을 textView에 출력
                } catch (IOException e) {
                    e.printStackTrace();
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
            return null;
        }
    }
    public void onBackPressed(){
        super.onBackPressed();
        Intent intent = new Intent(News.this, MainActivity.class);
        startActivity(intent);
    }
}