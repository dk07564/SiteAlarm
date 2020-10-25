package com.e.sitealarm;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.preference.PreferenceManager;
import android.util.Log;
import android.widget.Toast;

import androidx.core.app.NotificationCompat;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import java.io.IOException;
import java.util.ArrayList;

public class MonitorService extends Service implements Runnable{
    Thread thread;
    Boolean run=true;
    Context context;
    SharedPreferences sharedPreferences;
    SharedPreferences.Editor editor;
    public MonitorService() {

        Handler handler = new Handler(Looper.getMainLooper());
        handler.post(new Runnable() {
            @Override
            public void run() {
                context=getApplication();
                sharedPreferences = getSharedPreferences("pref", 0);
                editor = sharedPreferences.edit();
            }
        });


        thread = new Thread(this);
        thread.start();
    }

    @Override
    public IBinder onBind(Intent intent) {
        // TODO: Return the communication channel to the service.
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void run() {
        while(run){
            try {
                crawling();

                Log.e("LOG", "LOG");

                thread.sleep(10*60000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
    public void crawling() {
//        context = getApplicationContext();
//        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
//        SharedPreferences sharedPreferences = getSharedPreferences("pref", 0);
//        SharedPreferences.Editor editor = sharedPreferences.edit();

        String url = "https://news.naver.com/main/list.nhn?mode=LS2D&mid=shm&sid1=105&sid2=230";
        Document doc = null;

        try {
            doc = Jsoup.connect(url).get();
        } catch (IOException e) {
            e.printStackTrace();
        }

        Elements elements = (Elements) doc.getElementsByClass("type06_headline");
        Elements articles;
        String title, prevTitle, newsLink = null;
        Element newsContent;

        articles = elements.select("li");
        newsContent = articles.get(0);

        title = newsContent.select("a").text();
        newsLink = newsContent.select("a").attr("href");

        prevTitle = sharedPreferences.getString("title", "title");

        Log.e("TITLE", prevTitle);

        if (!title.equals(prevTitle)) {
            String channelId = "channel";
            String channelName = "Schedule";
            NotificationManager notificationManager = (NotificationManager) getApplicationContext().getSystemService(Context.NOTIFICATION_SERVICE);

            int importance = NotificationManager.IMPORTANCE_HIGH;
            NotificationChannel notificationChannel = null;
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                notificationChannel = new NotificationChannel(channelId, channelName, importance);
                notificationManager.createNotificationChannel(notificationChannel);
            }

            NotificationCompat.Builder builder = new NotificationCompat.Builder(getApplicationContext().getApplicationContext(), channelId);

//            Intent notificationIntent = new Intent(getApplicationContext(), MainActivity.class);
            Intent notificationIntent = new Intent(Intent.ACTION_VIEW, Uri.parse(newsLink));
            notificationIntent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);

            PendingIntent pendingIntent = PendingIntent.getActivity(getApplicationContext().getApplicationContext(), 0, notificationIntent,
                    PendingIntent.FLAG_UPDATE_CURRENT);

            builder.setContentTitle("새로운 기사가 등록되었습니다.")
                    .setContentText(title)
                    .setAutoCancel(true)
                    .setPriority(NotificationCompat.PRIORITY_HIGH)
                    .setSmallIcon(R.drawable.news)
                    .setContentIntent(pendingIntent);

            notificationManager.notify(0, builder.build());

            editor.putString("title", title);
            editor.commit();
        }
    }
}
