/*
 *    Copyright (C) 2017 MINDORKS NEXTGEN PRIVATE LIMITED
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package com.jason.facerecognition;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.PointF;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.content.Intent;

import com.flurgle.camerakit.CameraListener;
import com.flurgle.camerakit.CameraView;
import com.mindorks.tensorflowexample.R;

import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.lang.Math;

public class MainActivity extends AppCompatActivity {

    private static final int INPUT_SIZE = 160;
    private static final int IMAGE_MEAN = 117;
    private static final float IMAGE_STD = 1;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "embeddings";

    private static final String MODEL_FILE = "file:///android_asset/optimized_facenet.pb";
    private static final String LABEL_FILE = "file:///android_asset/recognition";

    private Classifier classifier;
    private Executor executor = Executors.newSingleThreadExecutor();
    private TextView textViewResult;
    private Button btnGetFace, btnToggleCamera, btnGetInto;
    private ImageView face1Result,face2Result;
    private CameraView cameraView;

    private List<Classifier.Recognition> mface1=null;
    private List<Classifier.Recognition> mface2=null;

    private int isGettingface=0;
    private double facedifference=1;
    private Bitmap bitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initView();//控件初始化

        cameraView.setCameraListener(new CameraListener() {
            @Override
            public void onPictureTaken(byte[] picture) {
                //从相机获取Byte流：picture
                super.onPictureTaken(picture);
                //解析picture为bitmap
                bitmap = BitmapFactory.decodeByteArray(picture, 0, picture.length);
                //获取人脸
                getFace(0);

                bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);

                //识别第一张人脸，获得128*1特征vector
                if(isGettingface%2==1){
                    face1Result.setImageBitmap(bitmap);
                    mface1 = classifier.recognizeImage(bitmap);
                }
                //识别第二张人脸，获得128*1特征vector
                else if(isGettingface%2==0){
                    face2Result.setImageBitmap(bitmap);
                    mface2 = classifier.recognizeImage(bitmap);
                }
                else{
                    face1Result.setImageResource(R.mipmap.ic_contact_picture);
                    face2Result.setImageResource(R.mipmap.ic_contact_picture);
                    mface1=null;
                    mface2=null;
                }

                if(null!=mface1 && null!=mface2){
                    facedifference=compareFace();
                    //textViewResult.setText(String.valueOf(facedifference));
                    if(facedifference<0.4){
                        makeButton2Visible();
                        textViewResult.setText(String.valueOf(facedifference));

                    }
                }
            }
        });

        btnToggleCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraView.toggleFacing();
            }
        });

        btnGetFace.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                isGettingface++;
                cameraView.captureImage();
            }
        });
        btnGetInto.setOnClickListener (new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, Main2Activity.class);
                startActivity(intent);
                MainActivity.this.finish();
            }
        });
        initTensorFlowAndLoadModel();
    }

    private void initView(){
        cameraView = (CameraView) findViewById(R.id.cameraView);
        face1Result = (ImageView) findViewById(R.id.face1);
        face2Result = (ImageView) findViewById(R.id.face2);
        textViewResult = (TextView) findViewById(R.id.text_view);

        btnToggleCamera = (Button) findViewById(R.id.btnToggleCamera);
        btnGetFace = (Button) findViewById(R.id.btnGetFace);
        btnGetInto = (Button) findViewById(R.id.btnGetInto);
    }

    @Override
    protected void onResume() {
        super.onResume();
        cameraView.start();
    }

    @Override
    protected void onPause() {
        cameraView.stop();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.execute(new Runnable() {
            @Override
            public void run() {
                classifier.close();
            }
        });
    }

    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = TensorflowFaceClassifier.create(
                            getAssets(),
                            MODEL_FILE,
                            LABEL_FILE,
                            INPUT_SIZE,
                            IMAGE_MEAN,
                            IMAGE_STD,
                            INPUT_NAME,
                            OUTPUT_NAME);
                    makeButtonVisible();
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    private void makeButtonVisible() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                btnGetFace.setVisibility(View.VISIBLE);
            }
        });
    }
    private void makeButton2Visible() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                btnGetInto.setVisibility(View.VISIBLE);
            }
        });
    }
    private double compareFace(){
        double runSum=0;
        if(null!=mface1 && null!=mface2) {
            Iterator<Classifier.Recognition> iter1= mface1.iterator();
            Iterator<Classifier.Recognition> iter2= mface2.iterator();
            while(iter1.hasNext()){
                //get the sum of power of the error
                runSum+=Math.pow(iter1.next().getConfidence()-iter2.next().getConfidence(),2);
            }
        }
        return Math.sqrt(runSum);
    }

    private void getFace(int index){
        //要使用Android内置的人脸识别，需要将Bitmap对象转为RGB_565格式，否则无法识别
        bitmap = bitmap.copy(Bitmap.Config.RGB_565, true);

        android.media.FaceDetector.Face[] mFace;
        android.media.FaceDetector mDector;
        int numOfMaxFaces=10;
        mFace=new android.media.FaceDetector.Face[numOfMaxFaces];
        mDector = new android.media.FaceDetector(bitmap.getWidth(),bitmap.getHeight(),numOfMaxFaces);
        int numResults = mDector.findFaces(bitmap,mFace);
        //获得人脸的数量
        Log.i("getFace", ""+numResults);
        if(numResults!=0) {
            PointF eyeMidPoint = new PointF();
            mFace[index].getMidPoint(eyeMidPoint);
            float eyesDistance = mFace[index].eyesDistance();

            bitmap = Bitmap.createBitmap(
                    bitmap,
                    (int) (eyeMidPoint.x - eyesDistance),
                    (int) (eyeMidPoint.y - eyesDistance),
                    (int) (eyesDistance * 2),
                    (int) (eyesDistance * 2.5)
            );
        }
    }




    //end of MainActivity
}
