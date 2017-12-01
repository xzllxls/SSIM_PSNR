package com.oceanai.main;

/**
 * Created by WangRupeng on 2017/11/27.
 */

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;

import javax.swing.*;

import java.io.File;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.Laplacian;


public class LaplaceTest {
    public static void main(String[] args) {
        LaplaceTest laplaceTest = new LaplaceTest();
        //double standardMean = laplaceTest.imageVar("D:\\JavaProjects\\JavaOpenCV\\src\\main\\resources\\1510889925564.jpg");
        //double comparedImage = laplaceTest.imageVar("D:\\JavaProjects\\JavaOpenCV\\src\\main\\resources\\1510889926692.jpg");
        String directory = "D:\\JavaProjects\\JavaOpenCV\\src\\main\\resources";
        File file = new File(directory);
        if (file.exists() && file.isDirectory()) {
            for (File f : file.listFiles()) {
                long start = System.currentTimeMillis();
                double result = laplaceTest.imageVar(f.getAbsolutePath());
                System.out.println(f.getName() + " std mean is " + result + " used time is " + (System.currentTimeMillis() - start));
            }
        }
        //System.out.println("Standard mean is " + standardMean + ", compared image is " + comparedImage);
    }

    public double imageVar(String imagePath) {
        Mat src = imread(imagePath);
        if (src.data() == null) {
            return -1;
        }
        Mat dest = new Mat();
        //Laplacian(src, dest, src.depth(), 1, 3, 0, BORDER_DEFAULT);
        Laplacian(src, dest, CV_64F);
        Mat tmp_m = new Mat();
        Mat tmp_sd = new Mat();
        //display(dest, "Laplacian");
        meanStdDev(dest, tmp_m, tmp_sd);
        /*BytePointer bytePointer = tmp_sd.data();
        byte[] bytes = bytePointer.getStringBytes();
        double result = LaplaceTest.byteToDouble(bytes);*/
        DoubleIndexer doubleIndexer = tmp_sd.createIndexer();
        double result = doubleIndexer.get(0,0);
        //System.out.println("Dest mean is " + mean + ", standard mean is " + result);
        return result;
    }

    public static double byteToDouble(byte[] b){
        long l;

        l=b[0];
        l&=0xff;
        l|=((long)b[1]<<8);
        l&=0xffff;
        l|=((long)b[2]<<16);
        l&=0xffffff;
        l|=((long)b[3]<<24);
        l&=0xffffffffl;
        l|=((long)b[4]<<32);
        l&=0xffffffffffl;

        l|=((long)b[5]<<40);
        l&=0xffffffffffffl;
        l|=((long)b[6]<<48);

        l|=((long)b[7]<<56);
        return Double.longBitsToDouble(l);
    }

    static void display(Mat image, String caption) {
        // Create image window named "My Image".
        final CanvasFrame canvas = new CanvasFrame(caption, 1.0);

        // Request closing of the application when the image window is closed.
        canvas.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

        // Convert from OpenCV Mat to Java Buffered image for display
        final OpenCVFrameConverter converter = new OpenCVFrameConverter.ToMat();
        // Show image on window.
        canvas.showImage(converter.convert(image));
    }
}