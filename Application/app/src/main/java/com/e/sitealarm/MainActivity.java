package com.e.sitealarm;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.preference.PreferenceManager;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.ListView;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    Context context;
    ProgressBar newsLoading;
//    RelativeLayout relativeLayout;
    String contents, selectedNews = null;
    ArrayList urlList = new ArrayList<>();
    Switch switchButton;
//    SharedPreferences sharedPreferences = getSharedPreferences("pref", 0);
    SharedPreferences sharedPreferences;
    SharedPreferences.Editor editor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
//        relativeLayout = findViewById(R.id.relativeLayout);
//        relativeLayout.setVisibility(View.GONE);

        context=getApplicationContext();
        context=getApplication();
        sharedPreferences = context.getSharedPreferences("pref", 0);
//        sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        editor = sharedPreferences.edit();

        MainActivity.AsyncT asyncT = new MainActivity.AsyncT();
        asyncT.execute();

        serviceOnOff();

    }

    public void listShow(ArrayList arrayList){
        try{
            ListView listView = findViewById(R.id.listView);
            ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                    android.R.layout.simple_list_item_1, arrayList);
            listView.setAdapter(adapter);

            listView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
                @Override
                public void onItemClick(AdapterView<?> adapterView, View view, int i, long l) {
//                    relativeLayout.setVisibility(View.VISIBLE);
                    selectedNews = urlList.get(i).toString();

                    MainActivity.contentsAsyncT contentsAsyncT = new MainActivity.contentsAsyncT();
                    contentsAsyncT.execute();
//                    newsLoading.setVisibility(View.VISIBLE);

                    ProgressDialog progressDialog = new ProgressDialog(MainActivity.this);
//                    progressDialog.setTitle("뉴스 기사 가져오는 중");
                    progressDialog.setMessage("뉴스 기사 가져오는 중");
                    progressDialog.show();

                }
            });
        }catch(Exception e){
            Log.e("surface ERROR", e.getMessage());
        }
    }

    //크롤링
    class AsyncT extends AsyncTask<Void, Void, String> {
        @Override
        protected String doInBackground(Void... params) {
            getURL();

            return null;
        }
    }

    public void getURL(){
        String url = "https://news.naver.com/main/list.nhn?mode=LS2D&mid=shm&sid1=105&sid2=230";
        Document doc = null;

        try {
            doc = Jsoup.connect(url).get();
        } catch (IOException e) {
            e.printStackTrace();
        }
        Elements elements = (Elements) doc.getElementsByClass("type06_headline");
        Elements articles;
        Elements title;
        Element newsContent;
        String newsLink;

        articles = elements.select("li");
        int cnt = articles.size();


        final ArrayList titleList = new ArrayList<>();


        for(int i = 0; i<cnt; i++){
            newsContent = articles.get(i);

            title = newsContent.select("a");
            newsLink = title.attr("href");

            titleList.add(title.text());
            urlList.add(newsLink);

//            if(i==0){
////                editor.putString("title", title.text());
//                editor.putString("title", "title");
//                editor.commit();
//            }
        }

        Handler handler = new Handler(Looper.getMainLooper());
        handler.post(new Runnable() {
            @Override
            public void run() {
                listShow(titleList);
            }
        });
    }

    class contentsAsyncT extends AsyncTask<Void, Void, String> {
        @Override
        protected String doInBackground(Void... params) {
            getContents(selectedNews);
            return null;
        }
    }

    public void getContents(String url){
        Document doc = null;

        try {
            doc = Jsoup.connect(url).get();
        } catch (IOException e) {
            e.printStackTrace();
        }
        Elements elements = (Elements) doc.getElementsByClass("article_body");
        contents = elements.select("div._article_body_contents").text();

        Log.e("CONTENTS", contents);

        Intent intent = new Intent(MainActivity.this, News.class);
        intent.putExtra("contents", contents);
        startActivity(intent);
    }

    public void serviceOnOff(){
        switchButton = findViewById(R.id.switchButton);
        Boolean switchOn;

        final Intent intent = new Intent(getApplicationContext(), MonitorService.class);


        final SharedPreferences.Editor editor = sharedPreferences.edit();
        switchOn = sharedPreferences.getBoolean("isChecked", true);

        switchButton.setChecked(switchOn);

        if(switchOn){
            startService(intent);
        }

        switchButton.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean isChecked) {
                if(isChecked){
                    Toast.makeText(MainActivity.this, "ON", Toast.LENGTH_SHORT).show();
                    startService(intent);
                }else{
                    stopService(intent);
                }
                editor.putBoolean("isChecked", isChecked);
                editor.commit();
            }
        });
    }
}