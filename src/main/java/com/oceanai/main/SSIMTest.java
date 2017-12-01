package com.oceanai.main;

/**
 * Created by WangRupeng on 2017/12/1 0001.
 */

import static java.lang.StrictMath.log10;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.GaussianBlur;

public class SSIMTest {
	public static void main(String[] args) {
		SSIMTest ssimTest = new SSIMTest();
		Mat image1 = imread("E:\\HUST\\HUST-BitData\\JavaOpenCV\\src\\main\\resources\\wangrupeng.png");
		Mat image2 = imread("E:\\HUST\\HUST-BitData\\JavaOpenCV\\src\\main\\resources\\wangrupeng2.png");
		long start = System.currentTimeMillis();
		double PSNR_Result = ssimTest.getPSNR(image1, image2);
		long psnrTime = System.currentTimeMillis() - start;
		start = System.currentTimeMillis();
		Scalar scalar = ssimTest.getMSSIM(image1, image2);
		long ssimTime = System.currentTimeMillis() - start;
		System.out.println("PSNR result is " + PSNR_Result + " time used " + psnrTime);
		System.out.println("SSIM result is " + scalar.get() + " time used " + ssimTime);
	}

	public double getPSNR(Mat I1, Mat I2) {
		Mat s1 = new Mat();
		MatExpr s2 = new MatExpr();
		absdiff(I1, I2, s1);         // |I1 - I2|
		s1.convertTo(s1, CV_32F);   // cannot make a square on 8 bits
		s2 = s1.mul(s1);             // |I1 - I2|^2
		s1 = s2.asMat();
		Scalar s = sumElems(s1);
		double sse =s.asBuffer().get(0) + s.asBuffer().get(1) + s.asBuffer().get(2); // sum channels
		if( sse <= 1e-10) { // for small values return zero
			return 0;
		} else {
			double  mse =sse /(double)(I1.channels() * I1.total());
			double psnr = 10.0*log10((255*255)/mse);
			return psnr;
		}
	}

	public Scalar getMSSIM(Mat i1, Mat i2) {
         double C1 = 6.5025, C2 = 58.5225;
		/***************************** INITS **********************************/
		int d     = CV_32F;

		Mat I1 = new Mat();
		Mat I2 = new Mat();
		i1.convertTo(I1, d);           // cannot calculate on one byte large values
		i2.convertTo(I2, d);

		Mat I2_2   = I2.mul(I2).asMat();        // I2^2
		Mat I1_2   = I1.mul(I1).asMat();        // I1^2
		Mat I1_I2  = I1.mul(I2).asMat();        // I1 * I2

		/*************************** END INITS **********************************/

		Mat mu1 = new Mat();
		Mat mu2 = new Mat();   // PRELIMINARY COMPUTING
		GaussianBlur(I1, mu1, new Size(11, 11), 1.5);
		GaussianBlur(I2, mu2, new Size(11, 11), 1.5);

		Mat mu1_2   =   mu1.mul(mu1).asMat();
		Mat mu2_2   =   mu2.mul(mu2).asMat();
		Mat mu1_mu2 =   mu1.mul(mu2).asMat();

		Mat sigma1_2 = new Mat();
		Mat sigma2_2 = new Mat();
		Mat sigma12 = new Mat();

		GaussianBlur(I1_2, sigma1_2, new Size(11, 11), 1.5);
		//sigma1_2 -= mu1_2;
		sigma1_2 = subtract(sigma1_2, mu1_2).asMat();

		GaussianBlur(I2_2, sigma2_2, new  Size(11, 11), 1.5);
		//sigma2_2 -= mu2_2;
		sigma2_2 = subtract(sigma2_2, mu2_2).asMat();

		GaussianBlur(I1_I2, sigma12, new Size(11, 11), 1.5);
		//sigma12 -= mu1_mu2;
		sigma12 = subtract(sigma12, mu1_mu2).asMat();

		///////////////////////////////// FORMULA ////////////////////////////////
		Mat t1, t2, t3;

		Scalar scalar = new Scalar(C1);

		t1 = add(multiply(2, mu1_mu2), scalar).asMat();;
		//t2 = 2 * sigma12 + C2;
		t2 = add(multiply(2, sigma12), new Scalar(C2)).asMat();
		t3 = t1.mul(t2).asMat();              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

		//t1 = mu1_2 + mu2_2 + C1;
		t1 = add(add(mu1_2, mu2_2), new Scalar(C1)).asMat();

		//t2 = sigma1_2 + sigma2_2 + C2;
		t2 = add(add(sigma1_2, sigma2_2), new Scalar(C2)).asMat();
		t1 = t1.mul(t2).asMat();               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))


		Mat ssim_map = new Mat();
		divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

		Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
		return mssim;
	}

}
